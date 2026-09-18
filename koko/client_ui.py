"""Tkinter/ttk desktop client for the koko websocket interpreter.

Tk owns the UI thread.  The websocket and audio capture/playback work in
background threads so a slow network or PortAudio callback never blocks the
window.  Sun Valley is applied to all ttk widgets when available.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import queue
import signal
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd

from . import client_devices as cdev
from clients.koko_client import KokoAudio, KokoClient, KokoControl

IN_RATE = 16_000
OUT_RATE = 48_000
SLICE_MS = 20
REFRESH_DEVICES_S = 3_000
RECONNECT_S = 2.0

ASR_LANGUAGE_OPTIONS = [
    ("English", "en"), ("Vietnamese", "vi"), ("Chinese", "zh"),
    ("Japanese", "ja"), ("Korean", "ko"), ("French", "fr"),
    ("German", "de"), ("Spanish", "es"), ("Italian", "it"),
    ("Portuguese", "pt"), ("Russian", "ru"), ("Thai", "th"),
    ("Indonesian", "id"), ("Arabic", "ar"), ("Hindi", "hi"),
]

log = logging.getLogger("koko.client_ui")

try:
    import soxr
except Exception:  # pragma: no cover - optional at import time
    soxr = None


def list_monitors() -> list[tuple[str, str]]:
    """Return PulseAudio/PipeWire sink monitors as ``(source, label)``."""
    if os.name == "nt":
        return []
    out: list[tuple[str, str]] = []
    try:
        import pulsectl

        with pulsectl.Pulse("koko-monitors") as pulse:
            sinks = {s.index: (getattr(s, "description", None) or s.name)
                     for s in pulse.sink_list()}
            for source in pulse.source_list():
                sink_id = getattr(source, "monitor_of_sink", None)
                if sink_id in sinks and "proctap" not in source.name:
                    out.append((source.name, sinks[sink_id]))
    except Exception:
        pass
    return sorted(out, key=lambda item: item[1].lower())


def _resample_to_16k(pcm16_bytes: bytes) -> bytes:
    if soxr is None:
        raise RuntimeError("soxr required to capture this mic at 48 kHz")
    values = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    output = soxr.resample(values, 48_000, IN_RATE)
    return (np.clip(output, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class ClientUI(tk.Tk):
    """Sun Valley ttk frontend for a single koko client session."""

    def __init__(self, url: str, language: str, device=None, out_device=None):
        super().__init__()
        self.title("CTE Intelligence Labs - koko")
        self.geometry("1180x760")
        self.minsize(820, 560)

        try:
            import sv_ttk
            sv_ttk.set_theme("dark")
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError("koko-client requires the sv-ttk package") from exc

        self.url = url
        self.language = language
        self._in_device = device
        self._out_device = out_device
        self._quitting = False
        self._connected = False
        self._session_ready = False
        self._session_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: KokoClient | None = None
        self._server_proc: subprocess.Popen | None = None
        self._server_stop_thread: threading.Thread | None = None
        self._ui_events: queue.Queue[tuple] = queue.Queue()
        self._stop = threading.Event()
        self._cap_stop = threading.Event()
        self._cap_thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None
        self._send_task: asyncio.Task | None = None
        self._source = ("mic", None)
        self._out_dev = None
        self._mute_in = False
        self._mute_out = False
        self._out_rate = [OUT_RATE]
        self._tts_voice: str | None = None
        self._asr_language = language
        self._asr_auto_detect = False
        self._in_pct = 0
        self._proc_err = ""
        self.pcm_in: queue.SimpleQueue[bytes] = queue.SimpleQueue()
        self.pcm_out: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()

        self._t_turns: list[list[str]] = []
        self._t_current: list[str] = []
        self._t_partial = ""
        self._l_turns: list[list[str]] = []
        self._l_current: list[str] = []
        self._source_opts: list[tuple[str, str]] = []
        self._out_opts: list[tuple[str, str]] = []

        self._build_ui()
        self._build_options(initial=True)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(50, self._drain_ui_events)
        self.after(500, self._refresh_status)
        self.after(500, self._refresh_server)
        self.after(REFRESH_DEVICES_S, self._refresh_options)
        self._session_thread = threading.Thread(target=self._session_worker,
                                                name="koko-websocket", daemon=True)
        self._session_thread.start()

    def _build_ui(self) -> None:
        self.style = ttk.Style(self)
        self.style.configure(".", focuscolor=self.style.lookup(".", "background"))

        root = ttk.Frame(self, padding=16)
        root.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        heading = ttk.Frame(root)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        heading.columnconfigure(0, weight=1)
        ttk.Label(heading, text="koko", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="Connecting...")

        panes = ttk.Panedwindow(root, orient="horizontal")
        panes.grid(row=1, column=0, sticky="nsew")
        transcript = ttk.Frame(panes, padding=(0, 0, 12, 0))
        settings = ttk.Frame(panes, padding=(12, 0, 0, 0))
        panes.add(transcript, weight=4)
        panes.add(settings, weight=1)
        transcript.rowconfigure(0, weight=1)
        transcript.rowconfigure(1, weight=1)
        transcript.columnconfigure(0, weight=1)
        self.transcribed = self._text_panel(transcript, 0, "Transcribed")
        self.translated = self._text_panel(transcript, 1, "Translated")
        self.transcribed.insert("1.0", "Waiting for speaker...", "placeholder")
        self.translated.insert("1.0", "Waiting for translation...", "placeholder")

        settings.columnconfigure(0, weight=1)
        self.source_var = self._setting(settings, 0, "Speech source")
        self.language_var = self._setting(settings, 2, "Whisper language")
        self.voice_var = self._setting(settings, 4, "VieNeu voice")
        self.output_var = self._setting(settings, 6, "Audio output")
        self.source_combo = ttk.Combobox(settings, textvariable=self.source_var, state="readonly")
        self.source_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.language_combo = ttk.Combobox(
            settings, textvariable=self.language_var, state="readonly",
            values=[label for label, _ in ASR_LANGUAGE_OPTIONS])
        self.language_combo.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        self.language_combo.bind("<<ComboboxSelected>>", self._language_changed)
        self.auto_button = ttk.Button(
            settings, text="Auto detect", style="Toggle.TButton",
            command=self._toggle_auto,
        )
        self.auto_button.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(0, 5))
        self.voice_combo = ttk.Combobox(settings, textvariable=self.voice_var,
                                        state="disabled")
        self.voice_combo.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.voice_combo.bind("<<ComboboxSelected>>", self._voice_changed)
        self.output_combo = ttk.Combobox(settings, textvariable=self.output_var, state="readonly")
        self.output_combo.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.output_combo.bind("<<ComboboxSelected>>", self._output_changed)
        self.mute_in_button = ttk.Button(settings, text="Mute input",
                                         style="Toggle.TButton", command=self._toggle_input)
        self.mute_in_button.grid(row=8, column=0, sticky="ew", padx=(0, 4), pady=4)
        self.mute_out_button = ttk.Button(settings, text="Mute output",
                                          style="Toggle.TButton", command=self._toggle_output)
        self.mute_out_button.grid(row=8, column=1, sticky="ew", padx=(4, 0), pady=4)
        ttk.Button(settings, text="Clear context", command=self._clear_context).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        ttk.Button(settings, text="Reload mic & speakers", command=self._reload_audio).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        self.level_var = tk.StringVar(value="Input level  0%")
        ttk.Label(settings, textvariable=self.level_var).grid(row=11, column=0,
                                                              columnspan=2, sticky="w", pady=12)
        self.server_var = tk.StringVar(value="Server stopped")
        self.server_button = ttk.Button(settings, text="Run server",
                                        style="Toggle.TButton",
                                        command=self._toggle_server)
        self.server_button.grid(row=14, column=0, columnspan=2, sticky="ew")
        settings.columnconfigure(1, weight=1)

        self.footer_var = tk.StringVar(value="Disconnected")
        ttk.Label(root, textvariable=self.footer_var, style="Caption.TLabel").grid(
            row=2, column=0, sticky="w", pady=(10, 0))
        self.style.configure("Title.TLabel", font=("TkDefaultFont", 22, "bold"))
        self.style.configure("Caption.TLabel", foreground="#8a8a8a")

    @staticmethod
    def _setting(parent, row: int, label: str) -> tk.StringVar:
        ttk.Label(parent, text=label).grid(row=row, column=0, columnspan=2,
                                           sticky="w", pady=(4, 5))
        return tk.StringVar()

    @staticmethod
    def _text_panel(parent, row: int, title: str) -> tk.Text:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 12 if row == 0 else 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = tk.Text(frame, wrap="word", relief="flat", borderwidth=0,
                       padx=8, pady=8, state="disabled", font=("TkDefaultFont", 12),
                       highlightthickness=0)
        text.tag_configure("placeholder", foreground="#8a8a8a")
        text.tag_configure("gray", foreground="#8a8a8a")
        text.grid(row=0, column=0, sticky="nsew")
        return text

    def _build_options(self, initial=False) -> None:
        self._compute_options()
        source_values = [label for label, _ in self._source_opts]
        output_values = [label for label, _ in self._out_opts]
        self.source_combo["values"] = source_values
        self.output_combo["values"] = output_values
        if initial:
            source = self._find_requested(self._in_device, self._source_opts, "idx:", "mic")
            output = self._find_requested(self._out_device, self._out_opts, "out:", "out:")
            self.source_var.set(self._label_for(source, self._source_opts))
            self.output_var.set(self._label_for(output, self._out_opts))
            self._set_language(self.language)
            self._source = self._value_to_source(source)
            self._out_dev = self._value_to_out(output)

    def _compute_options(self) -> None:
        pw_ok = cdev.pipewire_available()
        devs = sd.query_devices() if pw_ok else None
        sources = cdev.list_sources(devs) if pw_ok else []
        sinks = cdev.list_sinks(devs) if pw_ok else []
        if pw_ok:
            source_opts = [("Mic: (system default)", "mic")]
            source_opts += [(f"Mic: {d.name}", f"idx:{d.dev_index}") for d in sources if d.dev_index is not None]
            output_opts = [("Output: (system default / mix)", "out:")]
            output_opts += [(f"Output: {d.name}", f"out:{d.dev_index}") for d in sinks if d.dev_index is not None]
        else:
            source_opts = [("Mic: (default)", "mic")]
            output_opts = [("Output: (default)", "out:")]
            for index, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    source_opts.append((f"Mic: {device['name']}", f"idx:{index}"))
                if device["max_output_channels"] > 0:
                    output_opts.append((f"Output: {device['name']}", f"out:{index}"))
        for name, label in list_monitors():
            source_opts.append((f"Monitor: {label}", f"mon:{name}"))
        self._source_opts, self._out_opts = source_opts, output_opts

    @staticmethod
    def _find_requested(request, options, prefix, fallback):
        if isinstance(request, int):
            value = f"{prefix}{request}"
            if value in [v for _, v in options]:
                return value
        return fallback

    @staticmethod
    def _label_for(value, options):
        return next((label for label, item in options if item == value), options[0][0])

    @staticmethod
    def _value_to_source(value):
        if value.startswith("idx:"):
            return "mic", int(value[4:])
        if value.startswith("mon:"):
            return "mon", value[4:]
        return "mic", None

    @staticmethod
    def _value_to_out(value):
        return int(value[4:]) if value.startswith("out:") and value[4:].isdigit() else None

    def _selected_value(self, combo, options):
        label = combo.get()
        return next((value for item, value in options if item == label), options[0][1])

    def _refresh_options(self) -> None:
        old_source, old_output = self._source, self._out_dev
        self._compute_options()
        current_source = self._selected_value(self.source_combo, self._source_opts) if self.source_combo.get() else "mic"
        current_output = self._selected_value(self.output_combo, self._out_opts) if self.output_combo.get() else "out:"
        self.source_combo["values"] = [label for label, _ in self._source_opts]
        self.output_combo["values"] = [label for label, _ in self._out_opts]
        if current_source not in [value for _, value in self._source_opts]:
            current_source = "mic"
        if current_output not in [value for _, value in self._out_opts]:
            current_output = "out:"
        self.source_var.set(self._label_for(current_source, self._source_opts))
        self.output_var.set(self._label_for(current_output, self._out_opts))
        self._source, self._out_dev = self._value_to_source(current_source), self._value_to_out(current_output)
        if self._source != old_source and self._connected:
            self._apply_source()
        if self._out_dev != old_output:
            self.pcm_out.put(np.zeros(1, dtype=np.float32))
        self.after(REFRESH_DEVICES_S, self._refresh_options)

    # ---- websocket worker -------------------------------------------------
    def _session_worker(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_session())
        finally:
            self._loop.close()

    async def _run_session(self) -> None:
        while not self._quitting:
            client = KokoClient(self.url)
            self._client = client
            try:
                await client.connect(self._asr_language, auto_detect=self._asr_auto_detect)
                self._connected = True
                self._post_ui(self._set_connection, True, "Connected")
                self._start_audio()
                async for message in client.messages():
                    if isinstance(message, KokoAudio):
                        self._out_rate[0] = message.rate
                        self.pcm_out.put(message.samples())
                    else:
                        self._post_ui(self._handle_control, message)
            except Exception as exc:
                if not self._quitting:
                    log.debug("websocket session ended", exc_info=True)
                    self._post_ui(self._set_connection, False, f"Disconnected: {exc}")
            finally:
                self._connected = False
                self._stop_audio()
                await client.close(send_bye=False)
                self._client = None
            if not self._quitting:
                await asyncio.sleep(RECONNECT_S)

    def _post_ui(self, callback, *args) -> None:
        self._ui_events.put((callback, args))

    def _drain_ui_events(self) -> None:
        try:
            while True:
                callback, args = self._ui_events.get_nowait()
                callback(*args)
        except queue.Empty:
            pass
        if not self._quitting:
            self.after(50, self._drain_ui_events)

    def _set_connection(self, connected: bool, status: str) -> None:
        self._connected = connected
        self._session_ready = connected
        self.status_var.set(status)
        self.footer_var.set(status)

    def _handle_control(self, message: KokoControl) -> None:
        data = message.data
        if message.type == "ready":
            self._session_ready = True
            if data.get("asr_language"):
                self._set_language(data["asr_language"])
            if "asr_auto_detect" in data:
                self._set_auto(data["asr_auto_detect"])
            if "tts_voices" in data:
                self._set_voices(data["tts_voices"], data.get("tts_voice"))
        elif message.type == "asr_language":
            self._set_language(data.get("language"))
        elif message.type == "asr_auto_detect":
            self._set_auto(data.get("enabled"))
        elif message.type == "tts_voice":
            self._tts_voice = data.get("voice")
        elif message.type == "partial":
            self._t_partial = data.get("text", "") or ""
            self._render_transcribed()
        elif message.type == "final":
            if data.get("text"):
                self._t_current.append(data["text"])
            self._t_partial = ""
            self._render_transcribed()
        elif message.type == "speak":
            if self._t_current or self._t_partial:
                self._t_turns.append(self._t_current)
                self._t_current = []
                self._t_partial = ""
                self._render_transcribed()
            if self._l_current:
                self._l_turns.append(self._l_current)
                self._l_current = []
                self._render_translated()
        elif message.type == "translation":
            if data.get("text"):
                self._l_current.append(data["text"])
            self._render_translated()
        elif message.type == "error":
            self.status_var.set(f"Server: {data.get('detail', '')}")

    # ---- controls ---------------------------------------------------------
    def _run_control(self, operation) -> None:
        if self._loop and self._client and self._connected:
            asyncio.run_coroutine_threadsafe(operation(self._client), self._loop)

    def _language_changed(self, _event=None) -> None:
        label = self.language_var.get()
        language = next((code for name, code in ASR_LANGUAGE_OPTIONS if name == label), label)
        self._asr_language = language
        self._set_auto(False)
        self._run_control(lambda client: client.set_language(language))

    def _voice_changed(self, _event=None) -> None:
        voice = self.voice_var.get()
        self._tts_voice = voice
        self._run_control(lambda client: client.set_voice(voice))

    def _output_changed(self, _event=None) -> None:
        self._out_dev = self._value_to_out(self._selected_value(self.output_combo, self._out_opts))
        self.pcm_out.put(np.zeros(1, dtype=np.float32))

    def _toggle_auto(self) -> None:
        self._set_auto(not self._asr_auto_detect)
        self._run_control(lambda client: client.set_auto_detect(self._asr_auto_detect))

    def _toggle_input(self) -> None:
        self._mute_in = not self._mute_in
        if self._mute_in:
            self._teardown_capture()
            self._in_pct = 0
            while True:
                try:
                    self.pcm_in.get_nowait()
                except queue.Empty:
                    break
        else:
            self._apply_source()
        self.mute_in_button.state(["selected"] if self._mute_in else ["!selected"])
        self.mute_in_button.configure(text="Unmute input" if self._mute_in else "Mute input")

    def _toggle_output(self) -> None:
        self._mute_out = not self._mute_out
        self.mute_out_button.state(["selected"] if self._mute_out else ["!selected"])
        self.mute_out_button.configure(text="Unmute output" if self._mute_out else "Mute output")

    def _clear_context(self) -> None:
        self._t_turns.clear(); self._t_current.clear(); self._t_partial = ""
        self._l_turns.clear(); self._l_current.clear()
        self._render_transcribed(); self._render_translated()
        self._run_control(lambda client: client.clear_context())

    def _reload_audio(self) -> None:
        """Reopen both PortAudio streams and refresh available devices."""
        self._stop_audio()
        self._refresh_options()
        if self._connected and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._start_audio)

    # ---- local server -----------------------------------------------------
    def _toggle_server(self) -> None:
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self) -> None:
        root = Path(__file__).resolve().parents[1]
        try:
            self._server_proc = subprocess.Popen(
                [sys.executable, "-m", "koko.server", "--config", str(root / "config.toml")],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            self.server_var.set(f"Server failed: {exc}")
            return
        self.server_var.set("Server starting...")
        self.server_button.configure(text="Stop server")
        self.server_button.state(["selected"])

    def _stop_server(self, wait: bool = False) -> None:
        proc = self._server_proc
        if proc is None or proc.poll() is not None:
            self._server_proc = None
            self.server_var.set("Server stopped")
            self.server_button.configure(text="Run server", state="normal")
            self.server_button.state(["!selected"])
            return
        self.server_var.set("Stopping server...")
        self.server_button.configure(state="disabled")
        if wait:
            self._terminate_server(proc)
        elif self._server_stop_thread is None or not self._server_stop_thread.is_alive():
            self._server_stop_thread = threading.Thread(
                target=self._terminate_server, args=(proc,),
                name="koko-server-stop", daemon=True,
            )
            self._server_stop_thread.start()

    def _terminate_server(self, proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                if os.name == "nt":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        finally:
            self._post_ui(self._server_stopped, proc)

    def _server_stopped(self, proc: subprocess.Popen) -> None:
        if self._server_proc is proc:
            self._server_proc = None
            self.server_var.set("Server stopped")
            self.server_button.configure(text="Run server", state="normal")
            self.server_button.state(["!selected"])

    def _refresh_server(self) -> None:
        proc = self._server_proc
        if proc is not None:
            return_code = proc.poll()
            if return_code is None:
                if self.server_var.get() == "Server starting...":
                    self.server_var.set("Server running")
                self.server_button.configure(text="Stop server")
                self.server_button.state(["selected"])
            elif self._server_stop_thread is None or not self._server_stop_thread.is_alive():
                self._server_proc = None
                self.server_var.set(f"Server exited ({return_code})")
                self.server_button.configure(text="Run server", state="normal")
                self.server_button.state(["!selected"])
        if not self._quitting:
            self.after(500, self._refresh_server)

    def _set_language(self, language) -> None:
        if not isinstance(language, str):
            return
        self._asr_language = language
        self.language_var.set(next((name for name, code in ASR_LANGUAGE_OPTIONS if code == language), language))

    def _set_auto(self, enabled) -> None:
        if not isinstance(enabled, bool):
            return
        self._asr_auto_detect = enabled
        self.auto_button.state(["selected"] if enabled else ["!selected"])

    def _set_voices(self, voices, selected) -> None:
        values = [name for item in voices if isinstance(item, (list, tuple)) and len(item) == 2
                  for name in [item[0]]]
        if not values:
            self.voice_combo.configure(state="disabled")
            return
        self.voice_combo.configure(values=values, state="readonly")
        self._tts_voice = selected if selected in values else values[0]
        self.voice_var.set(self._tts_voice)

    # ---- audio ------------------------------------------------------------
    def _start_audio(self) -> None:
        self._stop.clear()
        self._cap_stop.set()
        self._play_thread = threading.Thread(target=self._play, name="koko-playback", daemon=True)
        self._play_thread.start()
        if not self._mute_in:
            self._apply_source()
        self._send_task = asyncio.create_task(self._sender())

    def _stop_audio(self) -> None:
        self._stop.set()
        if self._send_task:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._send_task.cancel)
            self._send_task = None
        self._teardown_capture()
        if self._play_thread:
            while not self.pcm_out.empty():
                try:
                    self.pcm_out.get_nowait()
                except queue.Empty:
                    break
            self._play_thread.join(timeout=1.0)
            self._play_thread = None

    async def _sender(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self.pcm_in.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if not self._mute_in and self._client:
                await self._client.send_audio(raw)

    def _apply_source(self) -> None:
        self._teardown_capture()
        if not self._connected:
            return
        self._cap_stop = threading.Event()
        kind, value = self._source
        if kind == "mon":
            if shutil.which("parec") is None:
                self._proc_err = "parec not found (install pulseaudio-utils)"
                return
            target = self._capture_monitor
            args = (value, self._cap_stop)
        else:
            target = self._capture_mic
            args = (value, self._cap_stop)
        self._cap_thread = threading.Thread(target=target, args=args, daemon=True)
        self._cap_thread.start()

    def _teardown_capture(self) -> None:
        if self._cap_thread:
            self._cap_stop.set()
            self._cap_thread.join(timeout=1.0)
            self._cap_thread = None

    def _capture_mic(self, device, stop: threading.Event) -> None:
        rate, resample = (IN_RATE, False) if device is None or os.name != "posix" else (48_000, True)

        def on_audio(indata, frames, time_info, status):
            raw = bytes(indata)
            if resample:
                try:
                    raw = _resample_to_16k(raw)
                except Exception as exc:
                    self._proc_err = str(exc)
                    return
            values = np.frombuffer(raw, dtype=np.int16)
            if values.size:
                self._in_pct = int(np.max(np.abs(values)) / 327.67)
            self.pcm_in.put(raw)

        try:
            stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16",
                                       blocksize=int(rate * SLICE_MS / 1000),
                                       callback=on_audio, device=device)
            with stream:
                while not stop.is_set():
                    sd.sleep(50)
        except Exception as exc:
            self._proc_err = str(exc)

    def _capture_monitor(self, monitor_source: str, stop: threading.Event) -> None:
        if soxr is None:
            self._proc_err = "soxr required for monitor capture"
            return
        proc = subprocess.Popen(["parec", f"--device={monitor_source}", "--format=s16le",
                                 "--rate=48000", "--channels=2"], stdout=subprocess.PIPE)
        block = int(48_000 * SLICE_MS / 1000) * 4
        try:
            while not stop.is_set():
                raw = proc.stdout.read(block)
                if not raw:
                    break
                values = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                count = values.size - values.size % 2
                mono = values[:count].reshape(-1, 2).mean(axis=1)
                output = soxr.resample(mono, 48_000, IN_RATE)
                self.pcm_in.put((np.clip(output, -1, 1) * 32767).astype(np.int16).tobytes())
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    def _play(self) -> None:
        stream = None
        key = None
        try:
            while not self._stop.is_set():
                new_key = (self._out_dev, self._out_rate[0])
                if stream is None or key != new_key:
                    if stream is not None:
                        stream.stop(); stream.close()
                    stream = sd.OutputStream(samplerate=new_key[1], channels=1,
                                             dtype="float32", device=new_key[0])
                    stream.start(); key = new_key
                if self._mute_out:
                    try:
                        self.pcm_out.get_nowait()
                    except queue.Empty:
                        pass
                    sd.sleep(20)
                    continue
                try:
                    samples = self.pcm_out.get_nowait()
                except queue.Empty:
                    sd.sleep(20)
                    continue
                stream.write(np.asarray(samples, dtype=np.float32))
        except Exception as exc:
            self._proc_err = str(exc)
        finally:
            if stream is not None:
                stream.stop(); stream.close()

    # ---- rendering and shutdown ------------------------------------------
    @staticmethod
    def _turn_text(turn):
        return " ".join(part for part in turn if part)

    def _render_transcribed(self) -> None:
        blocks = [self._turn_text(turn) for turn in self._t_turns if self._turn_text(turn)]
        current = " ".join(filter(None, [self._turn_text(self._t_current), self._t_partial]))
        if current:
            blocks.append(current)
        value = "\n\n".join(blocks)
        self._replace_text(
            self.transcribed, value or "Waiting for speaker...",
            placeholder=not value, gray_text=self._t_partial,
        )

    def _render_translated(self) -> None:
        blocks = [self._turn_text(turn) for turn in self._l_turns if self._turn_text(turn)]
        current = self._turn_text(self._l_current)
        if current:
            blocks.append(current)
        value = "\n\n".join(blocks)
        self._replace_text(
            self.translated, value or "Waiting for translation...",
            placeholder=not value, gray_text=current,
        )

    @staticmethod
    def _replace_text(widget: tk.Text, value: str, placeholder: bool = False,
                      gray_text: str = "") -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if placeholder:
            widget.insert("1.0", value, "placeholder")
        else:
            widget.insert("1.0", value)
            if gray_text:
                start = len(value) - len(gray_text)
                widget.tag_add("gray", f"1.0 + {start} chars", "end-1c")
        widget.see("end")
        widget.configure(state="disabled")

    def _refresh_status(self) -> None:
        self.level_var.set(f"Input level  {min(100, self._in_pct):3d}%")
        if self._proc_err:
            self.footer_var.set(self._proc_err)
        if not self._quitting:
            self.after(500, self._refresh_status)

    def close(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._stop_audio()
        if self._loop and self._loop.is_running() and self._client:
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
        if self._session_thread:
            self._session_thread.join(timeout=2.0)
        self._stop_server(wait=True)
        self.destroy()


def run_client_ui(url: str, language: str, device=None, out_device=None) -> None:
    app = ClientUI(url, language, device, out_device)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.close()
