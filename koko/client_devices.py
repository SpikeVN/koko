"""PipeWire device discovery + selection for the koko client.

On PipeWire systems ``sounddevice`` only exposes raw ALSA port names
(``sof-hda-dsp: - (hw:1,7)``) that don't map stably to the friendly,
human-facing device names PipeWire keeps on each node.  This module reads
those friendly names straight off the server (``pw-dump`` gives each
Audio Sink/Source node's ``node.description`` plus its ``api.alsa.path``)
and maps each node back to the PortAudio device index that actually
carries its audio.

Selection is by PortAudio device index (so we never depend on PipeWire's
active-default metadata -- which a remote/containerised WirePlumber may
refuse to promote): "pick a node" becomes "open that device index".

App-audio capture (proc-tap) is handled separately and needs none of this.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass

# auditable for tests
_PROG = "pw-dump"
_WPCTL = "wpctl"

_SECTION_RESET = (
    "Sink endpoints:",
    "Source endpoints:",
    "Streams:",
    "Sink:",
    "Source:",
    "Devices:",
    "Video:",
    "Settings:",
    "Clients:",
)


@dataclass
class PwDevice:
    """One selectable PipeWire audio endpoint."""

    id: int                       # PipeWire global node id (also used by wpctl)
    name: str                     # friendly name (node.description)
    kind: str                     # "source" or "sink"
    is_default: bool = False
    alsa_path: str | None = None  # e.g. "hw:sofhdadsp,6" from api.alsa.path
    dev_index: int | None = None  # matching PortAudio device index (if resolved)


def pipewire_available() -> bool:
    """True when the PipeWire CLI tools we need are on PATH."""
    return shutil.which(_WPCTL) is not None and shutil.which(_PROG) is not None


def _pw_nodes() -> tuple[dict[int, dict], dict[int, dict]]:
    """Return ``({source_id: {name, alsa_path}}, {sink_id: ...})``.

    If PipeWire is unreachable both dicts are empty.
    """
    sources: dict[int, dict] = {}
    sinks: dict[int, dict] = {}
    try:
        data = json.loads(subprocess.run(
            [_PROG], capture_output=True, text=True, check=True).stdout)
    except (subprocess.CalledProcessError, FileNotFoundError,
            json.JSONDecodeError):
        return sources, sinks
    for obj in data:
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        props = obj.get("info", {}).get("props", {})
        media_class = props.get("media.class", "")
        if media_class not in ("Audio/Source", "Audio/Sink"):
            continue
        desc = (props.get("node.description") or props.get("node.nick")
                or props.get("node.name"))
        if not desc:
            continue
        entry = {"name": desc,
                 "alsa_path": props.get("api.alsa.path")}
        target = sources if media_class == "Audio/Source" else sinks
        target[obj.get("id")] = entry
    return sources, sinks


def _alsa_card_indexes() -> dict[str, int]:
    """``{alsa card name: card index}`` from ``/proc/asound/cards``."""
    idx: dict[str, int] = {}
    try:
        with open("/proc/asound/cards") as f:
            for line in f:
                m = re.match(r"\s*(\d+)\s+\[([^\]]+)\]", line)
                if m:
                    idx[m.group(2).strip()] = int(m.group(1))
    except OSError:
        pass
    return idx


def _portaudio_for_alsa(alsa_path: str, kind: str, devs) -> int | None:
    """PortAudio device index carrying ``alsa_path`` (``hw:<card>[,<pcm>]``).

    PipeWire's ``api.alsa.path`` names the card (``hw:sofhdadsp,6``) while
    PortAudio names it by index (``(hw:1,6)``), so the card name is first
    resolved to its index via ``/proc/asound/cards``.  ``kind`` picks the
    input or output side for duplex devices.  Returns None if unmapped.
    """
    if not alsa_path or not alsa_path.startswith("hw:"):
        return None
    rest = alsa_path[3:]
    card_name, _, pcm = rest.partition(",")
    pcm = pcm or "0"
    card_idx = _alsa_card_indexes().get(card_name)
    if card_idx is None:
        return None
    want = "(hw:%d,%s)" % (card_idx, pcm)
    for i, d in enumerate(devs):
        if want not in (d.get("name") or ""):
            continue
        if kind == "source" and d.get("max_input_channels", 0) > 0:
            return i
        if kind == "sink" and d.get("max_output_channels", 0) > 0:
            return i
    return None


def _enrich_dev_index(d: PwDevice, devs) -> PwDevice:
    """Fill in ``d.dev_index`` from ``d.alsa_path`` against PortAudio ``devs``."""
    if devs is not None and d.dev_index is None:
        d.dev_index = _portaudio_for_alsa(d.alsa_path or "", d.kind, devs)
    return d


def _pw_defaults() -> tuple[int | None, int | None]:
    """Return ``(default_source_id, default_sink_id)`` from ``wpctl status``."""
    try:
        out = subprocess.run([_WPCTL, "status"], capture_output=True,
                             text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    default_source = default_sink = None
    section = None
    for line in out.splitlines():
        if "Sinks:" in line:
            section = "sinks"
            continue
        if "Sources:" in line:
            section = "sources"
            continue
        if any(r in line for r in _SECTION_RESET):
            section = None
            continue
        m = re.search(r"\*\s+(\d+)\.\s+(.*)$", line)
        if m:
            nid = int(m.group(1))
            if section == "sinks":
                default_sink = nid
            elif section == "sources":
                default_source = nid
    return default_source, default_sink


def _devices(kind: str) -> list[PwDevice]:
    """All endpoints of ``kind`` (``"source"``/``"sink"``), default first."""
    nodes = _pw_nodes()
    table = nodes[0] if kind == "source" else nodes[1]
    flags = _pw_defaults()
    default_id = flags[0] if kind == "source" else flags[1]
    out = [PwDevice(i, e["name"], kind, i == default_id, e["alsa_path"])
           for i, e in table.items()]
    out.sort(key=lambda d: (not d.is_default, d.name.lower()))
    return out


def list_sources(devs=None) -> list[PwDevice]:
    """Friendly names of every PipeWire mic; default first.

    Pass ``sd.query_devices()`` as ``devs`` to also resolve each node's
    PortAudio ``dev_index``.
    """
    return [_enrich_dev_index(x, devs) for x in _devices("source")]


def list_sinks(devs=None) -> list[PwDevice]:
    """Friendly names of every PipeWire speaker; default first.

    Pass ``sd.query_devices()`` as ``devs`` to also resolve each node's
    PortAudio ``dev_index``.
    """
    return [_enrich_dev_index(x, devs) for x in _devices("sink")]


def default_source() -> PwDevice | None:
    for d in list_sources():
        if d.is_default:
            return d
    srcs = list_sources()
    return srcs[0] if srcs else None


def default_sink() -> PwDevice | None:
    for d in list_sinks():
        if d.is_default:
            return d
    sinks = list_sinks()
    return sinks[0] if sinks else None
