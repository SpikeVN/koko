"""Textual TUI frontend for the koko *client*.

Runs the client session (mic/proc-tap -> websocket -> translated audio
out) inside its own event loop, so the device pickers can restart the
audio streams live and the panels can render the transcript.

    +-------------------+---------+
    |    transcribed    |  info   |
    +-------------------+         |
    |    translated     | panel   |
    +-------------------+---------+

``transcribed`` shows what was heard (partial ASR words replace the
current working text in place, finished turns separated by a blank line)
and ``translated`` the Vietnamese interpretation, aligned turn for turn.
The right ``info`` panel spans both rows and carries two live dropdowns:

  * **Speech source** — a microphone, or proc-tap capture of a specific
    audio-playing process (default falls back to the system mic when
    proc-tap is not installed).
  * **Audio output** — the device the translated speech is played on.

Picking a new device tears down and reopens just that audio stream; the
websocket session keeps running.

Run:  uv run koko-client [ws://host:6942] [--language en]   (TUI when TTY)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import subprocess
import threading

import numpy as np
import sounddevice as sd
import websockets
from rich.text import Text

from . import client_devices as cdev

from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, VerticalScroll
from textual.widgets import Button, Header, Select, Static

IN_RATE = 16_000
OUT_RATE = 48_000
SLICE_MS = 20
REFRESH_S = 0.5
REFRESH_DEVICES_S = 3.0  # how often the device/app pickers re-poll live sources

ASR_LANGUAGE_OPTIONS = [
    ("English", "en"),
    ("Vietnamese", "vi"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("French", "fr"),
    ("German", "de"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Thai", "th"),
    ("Indonesian", "id"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
]

log = logging.getLogger("koko.client_ui")

RECONNECT_S = 2.0  # backoff between reconnect attempts

# ---- optional process capture (proc-tap + psutil) -------------------------
try:
    from proctap import ProcessAudioCapture

    PROC_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    ProcessAudioCapture = None
    PROC_AVAILABLE = False

try:
    import soxr
except Exception:  # pragma: no cover
    soxr = None

_AUDIO_KEYWORDS = (
    "chrome",
    "firefox",
    "edge",
    "spotify",
    "vlc",
    "mpv",
    "mpc",
    "discord",
    "teams",
    "zoom",
    "slack",
    "obs",
    "player",
    "music",
    "media",
    "audacious",
    "rhythmbox",
    "foobar",
    "aimp",
    "winamp",
)


def list_processes() -> list[tuple[int, str]]:
    """Candidate audio-source processes as ``(pid, name)`` pairs.

    Prefers the live PulseAudio/PipeWire sink-input list (via ``pulsectl``),
    which yields exactly the processes currently *playing* audio -- the ones
    proc-tap can actually capture (e.g. Firefox's per-tab child processes,
    not just the main browser).  Falls back to a name-keyword scan of psutil
    or ``/proc`` when pulsectl is unavailable.  An empty list means
    "mic only".
    """
    procs: dict[int, str] = {}
    try:
        import pulsectl

        with pulsectl.Pulse("koko-procs") as pulse:
            for si in pulse.sink_input_list():
                pid = si.proplist.get("application.process.id")
                if not pid or not str(pid).isdigit():
                    continue
                name = (
                    si.proplist.get("application.process.binary")
                    or si.proplist.get("application.name")
                    or str(pid)
                )
                procs[int(pid)] = name
    except Exception:
        pass

    if not procs:
        # fallback: name-keyword scan (psutil, else /proc) for platforms
        # without a shared Pulse/PipeWire socket.
        try:
            import psutil

            def _iter():
                for p in psutil.process_iter(["pid", "name"]):
                    try:
                        yield int(p.info["pid"]), (p.info["name"] or "")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                        continue

            for pid, name in _iter():
                if any(k in name.lower() for k in _AUDIO_KEYWORDS):
                    procs[pid] = name
        except Exception:
            # no psutil: Linux /proc fallback (best-effort)
            if os.path.isdir("/proc"):
                try:
                    for entry in os.listdir("/proc"):
                        if not entry.isdigit():
                            continue
                        try:
                            with open(f"/proc/{entry}/comm") as f:
                                name = f.read().strip()
                        except OSError:
                            continue
                        if any(k in name.lower() for k in _AUDIO_KEYWORDS):
                            procs[int(entry)] = name
                except Exception:
                    pass

    return sorted(procs.items(), key=lambda x: x[1].lower())


def list_monitors() -> list[tuple[str, str]]:
    """Recordable sink monitors as ``(monitor_source_name, friendly_label)``.

    A "monitor" is the live tap of a sink's output, so selecting one captures
    whatever is playing to that sink -- without moving or corking any stream
    (unlike proc-tap, which redirects the target app and can disrupt it).
    Each entry is keyed by the Pulse/PipeWire monitor source name.
    """
    out: list[tuple[str, str]] = []
    try:
        import pulsectl

        with pulsectl.Pulse("koko-monitors") as p:
            sinks = {
                s.index: (getattr(s, "description", None) or s.name)
                for s in p.sink_list()
            }
            for s in p.source_list():
                of = getattr(s, "monitor_of_sink", None)
                if of in sinks and "proctap" not in s.name:
                    out.append((s.name, sinks[of]))
    except Exception:
        pass
    return sorted(out, key=lambda x: x[1].lower())


def _proc_to_pcm16(pcm_bytes: bytes) -> bytes:
    """proc-tap gives 48 kHz stereo float32; return 16 kHz mono int16."""
    x = np.frombuffer(pcm_bytes, dtype=np.float32)
    n = x.size - (x.size % 2)  # drop a trailing sample if odd
    mono = x[:n].reshape(-1, 2).mean(axis=1)
    mono = soxr.resample(mono, 48_000, IN_RATE)
    return (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _resample_to_16k(pcm16_bytes: bytes) -> bytes:
    """Raw ALSA mics are rate-locked to 48 kHz; bring their int16 down to 16k."""
    if soxr is None:
        raise RuntimeError("soxr required to capture this mic at 48 kHz")
    x = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    y = soxr.resample(x, 48_000, IN_RATE)
    return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class ClientUI(App[None]):
    TITLE = "CTE Intelligence Labs - koko"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+shift+c", "copy_transcript", "Copy transcript"),
    ]

    CSS = """
    Screen { background: $surface; }

    #layout {
        grid-size: 2 2;
        grid-columns: 1fr 46;
        grid-gutter: 0 1;          /* tight: no row gap, 1-cell column gap */
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    .panel {
        border: round $primary 60%;
        height: 100%;
    }

    #info { row-span: 2; width: 46; }

    .cfg-label { padding: 1 1 0 1; text-style: bold; }

    .cfg-select { margin: 0 1; width: 100%; }

    #mutes { width: auto; margin: 1 1; height: auto; }

    #mutes Button { width: 1fr; margin: 0; }

    #mute-out { border-left: solid $primary; }

    #auto-language { width: 1fr; margin: 1 1; }

    #clear-context { width: 1fr; margin: 1 1; }

    #footer {
        dock: bottom;
        height: auto;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
        text-align: center;
        border-top: solid $primary 25%;
    }
    """

    def __init__(self, url: str, language: str, device=None, out_device=None, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.language = language
        self._in_device = device  # mic device idx or friendly string
        self._out_device = out_device  # output device idx or friendly string

    # ---- lifecycle ----
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Grid(id="layout"):
            with VerticalScroll(id="wrap-transcribed", classes="panel"):
                yield Static(id="transcribed")
            with VerticalScroll(id="info", classes="panel"):
                yield Static(id="source-label", classes="cfg-label")
                yield Select(
                    id="source-select",
                    options=[("…", "…")],
                    prompt="…",
                    allow_blank=False,
                )
                yield Static(id="language-label", classes="cfg-label")
                yield Select(
                    id="language-select", options=ASR_LANGUAGE_OPTIONS,
                    prompt="Whisper language", allow_blank=False,
                )
                yield Button("Auto detect: Off", id="auto-language", classes="mute", variant="primary")
                yield Static(id="voice-label", classes="cfg-label")
                yield Select(
                    id="voice-select", options=[("Loading voices...", "")],
                    prompt="Loading voices...", allow_blank=False, disabled=True,
                )
                yield Static(id="out-label", classes="cfg-label")
                yield Select(
                    id="out-select", options=[("…", "…")], prompt="…",
                    allow_blank=False
                )
                with Horizontal(id="mutes"):
                    yield Button("Mute input", id="mute-in", classes="mute", variant="primary")
                    yield Button("Mute output", id="mute-out", classes="mute", variant="primary")
                yield Button("Clear context", id="clear-context", classes="mute", variant="primary")
                yield Static(id="footer")
            with VerticalScroll(id="wrap-translated", classes="panel"):
                yield Static(id="translated")

    def on_mount(self) -> None:
        self._connected = False
        self._session_ready = False
        self._quitting = False
        self._conn_err: str | None = None
        self._proc_err: str | None = None
        self._in_pct = 0
        self._tx = 0
        self._rx = 0

        # session state
        self._ws = None
        self._session_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._stop = threading.Event()  # playback / session threads
        self._play_thread: threading.Thread | None = None
        self._cap_thread: threading.Thread | None = None
        self._cap_stop = threading.Event()

        # audio plumbing
        self.pcm_in: "queue.SimpleQueue[bytes]" = queue.SimpleQueue()
        self.pcm_out: "queue.SimpleQueue[np.ndarray]" = queue.SimpleQueue()
        self.out_rate = [OUT_RATE]
        self._source = ("mic", None)  # ("mic", idx|None) / ("proc", pid) / ("mon", src)
        self._out_dev = None  # PortAudio output index, or None=default/mix
        self._monitor_labels: dict[str, str] = {}  # monitor source -> friendly sink
        self._mute_in = False  # suppress sending captured PCM to the server
        self._mute_out = False  # suppress playback of received PCM
        self._tts_voice: str | None = None
        self._asr_language = self.language
        self._asr_auto_detect = False
        self._setting_asr_language = False
        self._tick_n = 0

        self.transcribed = self.query_one("#transcribed", Static)
        self.translated = self.query_one("#translated", Static)
        self.transcribed.update("… waiting for speaker")
        self.translated.update("…")

        self.query_one("#wrap-transcribed").border_title = "Transcribed"
        self.query_one("#wrap-translated").border_title = "Translated"
        self.query_one("#info").border_title = "Info"
        self.query_one("#source-label", Static).update("Speech source")
        self.query_one("#language-label", Static).update("Whisper language")
        self.query_one("#voice-label", Static).update("VieNeu voice")
        self.query_one("#out-label", Static).update("Audio output")
        self.query_one("#footer", Static).update("(c) 2026 CTE Intelligence Labs")

        # Only the two transcript panels are copyable: keep them selectable
        # (allow_select stays True) and opt every other static text out of the
        # mouse-selection/copy walk.
        for sel_id in ("#source-label", "#language-label", "#voice-label", "#out-label", "#footer"):
            self.query_one(sel_id, Static).ALLOW_SELECT = False

        # transcript: finished spoken turns + in-progress current turn/partial
        self._t_turns: list[list[str]] = []
        self._t_current: list[str] = []
        self._t_partial: str = ""
        # translation: finished turns + in-progress current turn
        self._l_turns: list[list[str]] = []
        self._l_current: list[str] = []

        self._build_options()
        self._set_asr_language(self.language)
        self.set_interval(REFRESH_S, self._tick)
        self._session_task = asyncio.create_task(self._run_session())

    def _compute_options(self) -> None:
        """Re-derive the source/output option lists (and the PipeWire node
        name maps used for display). Refreshed periodically so newly-started
        apps (proc-tap) and hotplugged devices show up live."""
        pw_ok = cdev.pipewire_available()
        devs = sd.query_devices() if pw_ok else None

        self._pw_sources = cdev.list_sources(devs) if pw_ok else []
        self._pw_sinks = cdev.list_sinks(devs) if pw_ok else []
        if pw_ok:
            source_opts = [("Mic: (system default)", "mic")]
            for d in self._pw_sources:
                if d.dev_index is None:
                    continue
                source_opts.append((f"Mic: {d.name}", f"idx:{d.dev_index}"))
        else:
            source_opts = [("Mic: (default)", "mic")]
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    source_opts.append((f"Mic: {d['name']}", f"idx:{i}"))
        # NOTE: proc-tap per-process capture is intentionally NOT offered here:
        # on PipeWire it redirects the app's stream to a null sink, which both
        # pauses the playback and captures the wrong (silent) stream. Sink
        # monitor capture below does neither.
        self._monitor_labels = {}
        for mon_name, label in list_monitors():
            self._monitor_labels[mon_name] = label
            source_opts.append((f"Monitor: {label}", f"mon:{mon_name}"))
        self._source_opts = source_opts

        if pw_ok:
            out_opts = [("Output: (system default / mix)", "out:")]
            for d in self._pw_sinks:
                if d.dev_index is None:
                    continue
                out_opts.append((f"Output: {d.name}", f"out:{d.dev_index}"))
        else:
            out_opts = [("Output: (default)", "out:")]
            for i, d in enumerate(sd.query_devices()):
                if d["max_output_channels"] > 0:
                    out_opts.append((f"Output: {d['name']}", f"out:{i}"))
        self._out_opts = out_opts

    @staticmethod
    def _value_to_source(value: str) -> tuple:
        """Decode a source Select value into ``self._source`` state."""
        if value.startswith("proc:"):
            return ("proc", int(value.split(":", 1)[1]))
        if value.startswith("idx:"):
            return ("mic", int(value.split(":", 1)[1]))
        if value.startswith("mon:"):
            return ("mon", value.split(":", 1)[1])
        return ("mic", None)

    @staticmethod
    def _value_to_out(value: str):
        """Decode an output Select value into ``self._out_dev``."""
        if value.startswith("out:") and value[4:].isdigit():
            return int(value[4:])
        return None

    def _build_options(self) -> None:
        """Populate the two dropdowns from live PipeWire / process lists.

        Select values are strings (Textual can't print arbitrary tuples in
        the prompt line): sources encode as ``"idx:<n>"`` (a specific
        PortAudio mic device), ``"proc:<pid>"`` (app capture) or ``"mic"``
        (system default), outputs as ``"out:<n>"`` (a specific PortAudio
        output device) or ``"out:"`` (system default / mix -- follows the
        PipeWire active default so per-app routing via pavucontrol works).

        Friendly names come from PipeWire itself and are mapped to the
        PortAudio device index that actually carries each node's audio, so
        selecting a node opens that device directly -- no reliance on the
        system default changing.  Raw ALSA names are only a fallback when
        PipeWire isn't available.
        """
        self._compute_options()
        src = self.query_one("#source-select", Select)
        out = self.query_one("#out-select", Select)
        src.set_options(self._source_opts)
        out.set_options(self._out_opts)

        # initial values from CLI args (a PortAudio index, or None = default)
        def _match(val, opts, prefix, fallback):
            if isinstance(val, int):
                cand = prefix + str(val)
                return cand if cand in [v for _, v in opts] else fallback
            return fallback

        sv = _match(self._in_device, self._source_opts, "idx:", "mic")
        ov = _match(self._out_device, self._out_opts, "out:", "out:")
        src.value = sv
        out.value = ov
        self._source = self._value_to_source(sv)
        self._out_dev = self._value_to_out(ov)

    def _refresh_options(self) -> None:
        """Re-poll sources/outputs, keeping the current selection stable."""
        src = self.query_one("#source-select", Select)
        out = self.query_one("#out-select", Select)
        old_src, old_out = src.value, out.value
        old_source, old_out_dev = self._source, self._out_dev
        self._compute_options()
        src.set_options(self._source_opts)
        out.set_options(self._out_opts)
        avail_src = [v for _, v in self._source_opts]
        avail_out = [v for _, v in self._out_opts]
        newsrc = old_src if old_src in avail_src else "mic"
        newout = old_out if old_out in avail_out else "out:"
        src.value = newsrc
        out.value = newout
        new_source = self._value_to_source(newsrc)
        if new_source != old_source:
            self._source = new_source
            if self._connected:
                self._apply_source()
        self._out_dev = self._value_to_out(newout)

    # ---- websocket session ----
    async def _run_session(self) -> None:
        while True:
            self._conn_err = None
            try:
                async with websockets.connect(self.url, max_size=None) as ws:
                    self._ws = ws
                    self._connected = True
                    await ws.send(
                        json.dumps({
                            "type": "hello", "language": self._asr_language,
                            "asr_auto_detect": self._asr_auto_detect,
                        })
                    )
                    self._start_audio()
                    try:
                        async for msg in ws:
                            await self._handle(ws, msg)
                    finally:
                        self._stop_audio()
            except (
                websockets.ConnectionClosed,
                websockets.WebSocketException,
                OSError,
            ) as exc:
                self._conn_err = f"disconnected: {exc}"
            except Exception as exc:  # surface anything else on screen
                log.exception("session crashed")
                self._conn_err = f"session error: {exc}"
                self._stop_audio()
            finally:
                self._connected = False
                self._session_ready = False
                self._ws = None

            if self._quitting:
                return
            log.info("reconnecting in %.0fs", RECONNECT_S)
            await asyncio.sleep(RECONNECT_S)

    def _start_audio(self) -> None:
        self._stop.clear()
        self._cap_stop.set()  # ensure clean prior state
        self._play_thread = threading.Thread(target=self._play, daemon=True)
        self._play_thread.start()
        self._apply_source()
        self._send_task = asyncio.create_task(self._sender())

    def _stop_audio(self) -> None:
        self._stop.set()
        if self._send_task is not None:
            self._send_task.cancel()
            self._send_task = None
        self._teardown_capture()
        if self._play_thread is not None:
            while not self.pcm_out.empty():  # unblock _play from stream.write
                try:
                    self.pcm_out.get_nowait()
                except queue.Empty:
                    break
            self._play_thread.join(timeout=1.0)  # portaudio segfaults if killed
            self._play_thread = None

    async def _sender(self) -> None:
        """Drain captured PCM16 -> websocket."""
        while not self._stop.is_set():
            try:
                raw = self.pcm_in.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)  # ~2x the capture block rate
                continue
            if self._mute_in:  # keep draining (live level) but don't transmit
                continue
            if self._ws is not None:
                self._tx += len(raw)
                await self._ws.send(raw)

    async def _handle(self, ws, msg) -> None:
        if isinstance(msg, bytes):
            self._rx += len(msg)
            x = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
            self.pcm_out.put(x)
            return
        ctl = json.loads(msg)
        t = ctl.get("type")
        if t == "audio":
            self.out_rate[0] = int(ctl.get("rate", OUT_RATE))
        elif t == "ready":
            self._session_ready = True
            self._conn_err = None
            if ctl.get("asr_language"):
                self._set_asr_language(ctl["asr_language"])
            if "asr_auto_detect" in ctl:
                self._set_asr_auto_detect(ctl["asr_auto_detect"])
            if "tts_voices" in ctl:
                self._set_voice_options(ctl["tts_voices"], ctl.get("tts_voice"))
        elif t == "asr_language":
            self._set_asr_language(ctl.get("language"))
            self._set_asr_auto_detect(ctl.get("asr_auto_detect", False))
        elif t == "asr_auto_detect":
            self._set_asr_auto_detect(ctl.get("enabled"))
        elif t == "tts_voice":
            self._tts_voice = ctl.get("voice")
        elif t == "partial":
            self._t_partial = ctl.get("text", "") or ""
            self._render_transcribed()
        elif t == "final":
            if ctl.get("text"):
                self._t_current.append(ctl["text"])
            self._t_partial = ""
            self._render_transcribed()
        elif t == "speak":
            # gate released a turn: commit it, open the next on both panels.
            if self._t_current or self._t_partial:
                self._t_turns.append(self._t_current)
                self._t_current = []
                self._t_partial = ""
                self._render_transcribed()
            if self._l_current:
                self._l_turns.append(self._l_current)
                self._l_current = []
                self._render_translated()
        elif t == "translation":
            if ctl.get("text"):
                self._l_current.append(ctl["text"])
            self._render_translated()
        elif t == "error":
            self._conn_err = "server: %s" % ctl.get("detail")

    # ---- audio capture (source switching) ----
    def on_select_changed(self, event: Select.Changed) -> None:
        widget = event.select.id
        value = event.value
        if value in (Select.BLANK, None):
            return
        if widget == "source-select":
            new = self._value_to_source(value)
            if new != self._source:
                self._source = new
                self._apply_source()
        elif widget == "language-select":
            self._asr_language = str(value)
            self._set_asr_auto_detect(False)
            if not self._setting_asr_language and self._connected and self._ws is not None:
                asyncio.create_task(self._ws.send(json.dumps({
                    "type": "asr_language", "language": self._asr_language,
                })))
        elif widget == "out-select":
            self._out_dev = self._value_to_out(value)
            # a silent nudge lets _play notice the new device promptly
            if self._connected:
                self.pcm_out.put(np.zeros(1, dtype=np.float32))
        elif widget == "voice-select":
            self._tts_voice = str(value)
            if self._connected and self._ws is not None:
                asyncio.create_task(self._ws.send(json.dumps({
                    "type": "tts_voice", "voice": self._tts_voice,
                })))

    def _set_asr_language(self, language) -> None:
        """Reflect a server-confirmed language without sending another control."""
        if not isinstance(language, str):
            return
        select = self.query_one("#language-select", Select)
        values = [value for _, value in ASR_LANGUAGE_OPTIONS]
        if language not in values:
            return
        self._setting_asr_language = True
        try:
            self._asr_language = language
            select.value = language
        finally:
            self._setting_asr_language = False

    def _set_asr_auto_detect(self, enabled) -> None:
        if not isinstance(enabled, bool):
            return
        self._asr_auto_detect = enabled
        button = self.query_one("#auto-language", Button)
        button.label = "Auto detect: On" if enabled else "Auto detect: Off"
        button.set_class(enabled, "muted")

    def _set_voice_options(self, voices, selected) -> None:
        """Populate the TTS selector from server-provided VieNeu metadata."""
        select = self.query_one("#voice-select", Select)
        options = []
        for voice in voices:
            if not isinstance(voice, (list, tuple)) or len(voice) != 2:
                continue
            name, description = voice
            options.append((f"{name} - {description}" if description else name, name))
        if not options:
            select.disabled = True
            return
        select.set_options(options)
        self._tts_voice = selected if selected in [value for _, value in options] else options[0][1]
        select.value = self._tts_voice
        select.disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mute-in":
            self._mute_in = not self._mute_in
            event.button.label = "Unmute input" if self._mute_in else "Mute input"
        elif event.button.id == "mute-out":
            self._mute_out = not self._mute_out
            event.button.label = "Unmute output" if self._mute_out else "Mute output"
        elif event.button.id == "auto-language":
            self._set_asr_auto_detect(not self._asr_auto_detect)
            if self._connected and self._ws is not None:
                asyncio.create_task(self._ws.send(json.dumps({
                    "type": "asr_auto_detect", "enabled": self._asr_auto_detect,
                })))
            return
        elif event.button.id == "clear-context":
            self._t_turns.clear()
            self._t_current.clear()
            self._t_partial = ""
            self._l_turns.clear()
            self._l_current.clear()
            self._render_transcribed()
            self._render_translated()
            if self._connected and self._ws is not None:
                asyncio.create_task(self._ws.send(json.dumps({
                    "type": "clear_context",
                })))
            self.notify("Context cleared")
            return
        else:
            return
        # reflect the muted state with a red-tinted style (no white text)
        if event.button.id == "mute-in":
            self.query_one("#mute-in").set_class(self._mute_in, "muted")
        else:
            self.query_one("#mute-out").set_class(self._mute_out, "muted")

    def action_copy_transcript(self) -> None:
        """Copy the current mouse text-selection (only transcript panels are
        selectable) to the clipboard."""
        try:
            sel = self.screen.get_selected_text()
        except Exception:
            sel = None
        if sel:
            self.copy_to_clipboard(sel)
            self.notify("Copied transcript selection")
        else:
            self.notify(
                "Drag to select text in a transcript panel first, then ctrl+shift+c",
                severity="warning",
                timeout=4,
            )

    def _apply_source(self) -> None:
        """Restart the capture thread to match ``self._source``."""
        self._teardown_capture()
        if not self._connected:
            return
        kind, value = self._source
        self._cap_stop = threading.Event()
        if kind == "proc":
            if not PROC_AVAILABLE or ProcessAudioCapture is None:
                self._proc_err = "proc-tap not installed"
                return
            self._cap_thread = threading.Thread(
                target=self._capture_proc, args=(value, self._cap_stop), daemon=True
            )
        elif kind == "mon":
            if shutil.which("parec") is None:
                self._proc_err = "parec not found (install pulseaudio-utils)"
                return
            self._cap_thread = threading.Thread(
                target=self._capture_monitor,
                args=(value, self._cap_stop),
                daemon=True,
            )
        else:
            # ("mic", idx|None): open that PortAudio device index, or the
            # system default (None) which follows the PipeWire active default.
            self._cap_thread = threading.Thread(
                target=self._capture_mic, args=(value, self._cap_stop), daemon=True
            )
        self._cap_thread.start()

    def _teardown_capture(self) -> None:
        if self._cap_thread is not None:
            self._cap_stop.set()
            self._cap_thread.join(timeout=1.0)
            self._cap_thread = None

    def _on_pcm16(self, raw: bytes) -> None:
        self._in_pct = int(np.max(np.abs(np.frombuffer(raw, dtype=np.int16))) / 327.67)
        self.pcm_in.put(raw)

    def _capture_mic(self, device, stop: threading.Event) -> None:
        # A None device opens the system default (via PipeWire, any rate);
        # an explicit index is a raw ALSA mic, rate-locked to 48 kHz, so it
        # is captured at 48k and resampled down to the 16 kHz wire rate.
        if device is None:
            rate, resample = IN_RATE, False
        else:
            rate, resample = 48_000, True

        def on_audio(indata, frames, time_info, status):
            raw = bytes(indata)
            if resample:
                try:
                    raw = _resample_to_16k(raw)
                except Exception as exc:
                    self._proc_err = repr(exc)
                    return
            self._on_pcm16(raw)

        stream = sd.RawInputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            blocksize=int(rate * SLICE_MS / 1000),
            callback=on_audio,
            device=device,
        )
        with stream:
            while not stop.is_set():
                sd.sleep(50)

    def _capture_monitor(self, monitor_source: str, stop: threading.Event) -> None:
        """Record a sink monitor via ``parec`` and stream 16 kHz mono PCM.

        Unlike proc-tap this never moves or corks the app's stream -- it just
        taps the sink the audio is already playing to, so playback is
        uninterrupted.  ``parec`` delivers 48 kHz stereo s16; we downmix to
        mono and resample to 16 kHz with soxr (matching the wire format)."""
        if soxr is None:
            self._proc_err = "soxr required for monitor capture"
            return
        proc = subprocess.Popen(
            [
                "parec",
                "--device=%s" % monitor_source,
                "--format=s16le",
                "--rate=48000",
                "--channels=2",
            ],
            stdout=subprocess.PIPE,
        )
        block = int(48_000 * SLICE_MS / 1000) * 2 * 2  # 20 ms stereo s16 bytes
        try:
            while not stop.is_set():
                raw = proc.stdout.read(block)
                if not raw:
                    break
                x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                n = x.size - (x.size % 2)
                mono = x[:n].reshape(-1, 2).mean(axis=1)
                mono16 = soxr.resample(mono, 48_000, IN_RATE)
                pcm = (np.clip(mono16, -1.0, 1.0) * 32767).astype(np.int16)
                self._on_pcm16(pcm.tobytes())
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    def _capture_proc(self, pid: int, stop: threading.Event) -> None:
        def on_data(pcm_bytes, frames):
            try:
                data = _proc_to_pcm16(pcm_bytes)
            except Exception as exc:
                self._proc_err = repr(exc)
                return
            if data:
                self._on_pcm16(data)

        tap = ProcessAudioCapture(pid, on_data=on_data)
        try:
            tap.start()
        except Exception as exc:
            self._proc_err = repr(exc)
            return
        try:
            while not stop.is_set():
                sd.sleep(50)
        finally:
            tap.stop()

    def _play(self) -> None:
        """Drain decoded PCM chunks out the speakers, reopening on change.

        ``_play`` polls the current output device + sample rate every
        iteration so a dropdown swap is picked up even with no audio
        pending; portaudio is never stopped mid-``stream.write`` because
        we only stop between frames here (and join before kill in
        ``_stop_audio``)."""
        key = None
        stream = None

        def _open() -> None:
            nonlocal key, stream
            # a chosen output opens its PortAudio device index directly;
            # None (default / mix) follows the PipeWire active default.
            dev = self._out_dev
            rate = self.out_rate[0]
            new_key = (dev, rate)
            if stream is not None and key != new_key:
                stream.stop()
                stream.close()
                stream = None
            if stream is None:
                stream = sd.OutputStream(
                    samplerate=rate, channels=1, dtype="float32", device=dev
                )
                stream.start()
                key = new_key

        while not self._stop.is_set():
            _open()
            if self._mute_out:
                # drop rather than play; keep the stream open for instant unmute
                try:
                    self.pcm_out.get_nowait()
                except queue.Empty:
                    pass
                sd.sleep(20)
                continue
            try:
                x = self.pcm_out.get_nowait()
            except queue.Empty:
                sd.sleep(20)
                continue
            stream.write(np.asarray(x, dtype=np.float32))
        if stream is not None:
            stream.stop()
            stream.close()

    # ---- rendering ----
    @staticmethod
    def _turn_text(turn: list[str]) -> str:
        return " ".join(t for t in turn if t)

    def _render_transcribed(self) -> None:
        # committed speech (already sent to translation) renders white; the
        # in-progress whisper partial renders dim/gray so it's clearly "not
        # sent yet". Finished turns and the current turn's finals are each
        # their own block; the live partial trails the current turn inline.
        segs: list[list[tuple[str, str]]] = []
        for turn in self._t_turns:
            txt = self._turn_text(turn)
            if txt:
                segs.append([(txt, "white")])
        cur: list[tuple[str, str]] = []
        if self._t_current:
            cur.append((" ".join(self._t_current), "white"))
        if self._t_partial:
            cur.append((self._t_partial, "dim"))
        if cur:
            segs.append(cur)
        text = Text()
        for bi, block in enumerate(segs):
            if bi:
                text.append("\n\n")
            for ci, (s, style) in enumerate(block):
                if ci:
                    text.append(" ")
                text.append(s, style=style)
        self.transcribed.update(text if segs else "… waiting for speaker")
        self._scroll_end()

    def _render_translated(self) -> None:
        # finished translations (already spoken as TTS) render white; the
        # in-progress LLM translation (not yet spoken) renders dim/gray.
        segs: list[list[tuple[str, str]]] = []
        for turn in self._l_turns:
            txt = self._turn_text(turn)
            if txt:
                segs.append([(txt, "white")])
        cur_txt = self._turn_text(self._l_current)
        if cur_txt:
            segs.append([(cur_txt, "dim")])
        text = Text()
        for bi, block in enumerate(segs):
            if bi:
                text.append("\n\n")
            for ci, (s, style) in enumerate(block):
                if ci:
                    text.append(" ")
                text.append(s, style=style)
        self.translated.update(text if segs else "…")
        self._scroll_end()

    def _scroll_end(self) -> None:
        for wrap in (
            self.query_one("#wrap-transcribed", VerticalScroll),
            self.query_one("#wrap-translated", VerticalScroll),
        ):
            wrap.scroll_end(animate=False)

    def _tick(self) -> None:
        self._tick_n += 1
        if self._tick_n % max(1, int(REFRESH_DEVICES_S / REFRESH_S)) == 0:
            self._refresh_options()

    def action_quit(self) -> None:
        self._quitting = True
        if self._session_task is not None:
            self._session_task.cancel()
        self._stop_audio()
        self.exit()
