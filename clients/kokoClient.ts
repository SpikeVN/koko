/**
 * API-only Bun client for the koko websocket protocol.
 *
 * This module handles only transport and protocol concerns. It does not
 * capture a microphone, play audio, resample samples, expose an HTTP API, or
 * provide a user interface. The host application owns those concerns and
 * supplies raw PCM16 input through {@link KokoClient.sendAudio}.
 *
 * Koko multiplexes JSON text frames and binary frames on one websocket:
 *
 * - Client-to-server binary frames are mono signed PCM16 little-endian at
 *   16 kHz. A 20 ms frame is normally 640 bytes.
 * - Client-to-server JSON frames are control commands.
 * - Server-to-client JSON frames are events.
 * - Server-to-client binary frames are raw mono PCM16 output audio.
 *
 * Every server audio frame is immediately preceded by an
 * `{"type":"audio","rate":N}` JSON frame. The rate applies to the next
 * binary frame and may change during a session. `messages()` yields the
 * announcement as a normal {@link KokoControl}, then yields the following
 * bytes as {@link KokoAudio}.
 *
 * Minimal usage:
 *
 * ```ts
 * const client = new KokoClient("ws://127.0.0.1:6942");
 * await client.connect("en");
 * try {
 *   await client.sendAudio(microphonePcm16);
 *   for await (const message of client.messages()) {
 *     if (message.kind === "audio") {
 *       output.write(message.pcm16, message.rate);
 *     } else if (message.type === "translation") {
 *       console.log(message.data.text);
 *     }
 *   }
 * } finally {
 *   await client.close();
 * }
 * ```
 *
 * Keep one `messages()` consumer running for the lifetime of the connection.
 * Audio and text events are interleaved. Each connection is an independent
 * server-side session, but an instance owns only one connection at a time.
 */

/** A JSON event received from the koko server. */
export type KokoControl = {
  /** Distinguishes a JSON protocol event from a PCM payload. */
  kind: "control";
  /** Event name, such as `ready`, `partial`, `translation`, or `error`. */
  type: string;
  /** Complete JSON event, retained for forward-compatible field access. */
  data: Record<string, unknown>;
};

/** One server-to-client raw mono PCM16 little-endian audio frame. */
export type KokoAudio = {
  /** Discriminator for narrowing a {@link KokoMessage}. */
  kind: "audio";
  /** Raw PCM16 bytes. This is not a WAV file and has no header. */
  pcm16: Uint8Array;
  /** Sample rate announced for this frame, normally 48000. */
  rate: number;
};

/** Any message yielded by {@link KokoClient.messages}. */
export type KokoMessage = KokoControl | KokoAudio;

type Pending =
  | { kind: "message"; value: KokoMessage }
  | { kind: "error"; error: Error };

export class KokoClient {
  /** Websocket endpoint used by this client. */
  readonly url: string;
  /** The browser/Bun websocket; absent before connect and after close. */
  private socket?: WebSocket;
  /** Rate announced by the most recent server `audio` control event. */
  private audioRate = 48_000;
  /** Messages received before an async-generator consumer asks for them. */
  private pending: Pending[] = [];
  /** Resolvers waiting for the next message, in consumer order. */
  private waiters: Array<(pending: Pending) => void> = [];
  /** Terminal error delivered to current and future message consumers. */
  private ended?: Error;
  /** Serializes Blob decoding with later websocket events. */
  private receiveChain: Promise<void> = Promise.resolve();

  /**
   * Create a disconnected client.
   *
   * @param url Koko websocket endpoint. Defaults to the local port 6942.
   */
  constructor(url = "ws://127.0.0.1:6942") {
    this.url = url;
  }

  /**
   * Whether the underlying websocket is currently OPEN.
   *
   * This is a point-in-time state check. A network failure can occur after it
   * returns `true`; send methods and `messages()` remain authoritative.
   */
  get connected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  /**
   * Open the websocket and send the initial `hello` command.
   *
   * @param language Initial Whisper source-language code, for example `en`.
   * @param autoDetect Enable Whisper automatic source-language detection.
   * @throws If already connected or if the websocket handshake fails.
   *
   * Successful completion means that `hello` was sent. The server's `ready`
   * response arrives through `messages()` afterward.
   */
  async connect(language = "en", autoDetect = false): Promise<void> {
    if (this.socket) throw new Error("koko client is already connected");
    this.ended = undefined;
    this.receiveChain = Promise.resolve();
    const socket = new WebSocket(this.url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    try {
      await new Promise<void>((resolve, reject) => {
        const cleanup = () => {
          socket.onopen = null;
          socket.onerror = null;
          socket.onclose = null;
        };
        socket.onopen = () => {
          cleanup();
          resolve();
        };
        socket.onerror = () => {
          cleanup();
          reject(new Error("koko websocket connection failed"));
        };
        socket.onclose = () => {
          cleanup();
          reject(new Error("koko websocket closed before opening"));
        };
      });
      socket.onmessage = (event) => this.enqueueMessage(socket, event.data);
      socket.onclose = () => {
        if (this.socket === socket) this.end(new Error("koko websocket closed"));
      };
      socket.onerror = () => {
        if (this.socket === socket) this.end(new Error("koko websocket error"));
      };
      await this.sendControl({
        type: "hello",
        language,
        asr_auto_detect: autoDetect,
      });
    } catch (error) {
      await this.close(false);
      throw error;
    }
  }

  /**
   * Close the socket and end pending message iteration.
   *
   * @param sendBye Send the protocol `bye` command before closing. Pass
   * `false` while recovering from a failed connection.
   *
   * Closing is idempotent. It does not drain audio already received by the
   * application; drain `messages()` first if final playback matters.
   */
  async close(sendBye = true): Promise<void> {
    const socket = this.socket;
    this.socket = undefined;
    if (!socket) return;
    if (sendBye && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "bye" }));
    }
    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close();
    }
    this.end(new Error("koko client closed"));
  }

  /**
   * Send one raw microphone frame.
   *
   * @param pcm16 Mono signed PCM16 little-endian samples at 16 kHz. A 20 ms
   * frame is typically 640 bytes. Do not include a WAV header, JSON, or rate
   * prefix.
   * @throws If disconnected or if the websocket cannot send.
   *
   * The server may drop input under downstream pressure. Keep the capture
   * queue shallow to preserve live latency.
   */
  async sendAudio(pcm16: Uint8Array | ArrayBuffer): Promise<void> {
    const socket = this.requireSocket();
    await this.waitUntilOpen(socket);
    const bytes = pcm16 instanceof Uint8Array ? pcm16 : new Uint8Array(pcm16);
    socket.send(bytes.slice().buffer);
  }

  /** Request a new Whisper source language for the active session. */
  setLanguage(language: string): Promise<void> {
    return this.sendControl({ type: "asr_language", language });
  }

  /** Enable or disable server-side automatic source-language detection. */
  setAutoDetect(enabled: boolean): Promise<void> {
    return this.sendControl({ type: "asr_auto_detect", enabled });
  }

  /** Request a TTS voice by its exact name from the server's `ready` list. */
  setVoice(voice: string): Promise<void> {
    return this.sendControl({ type: "tts_voice", voice });
  }

  /** Clear buffered source text and the server's LLM conversation history. */
  clearContext(): Promise<void> {
    return this.sendControl({ type: "clear_context" });
  }

  /**
   * Yield server events and audio in websocket arrival order.
   *
   * JSON frames are yielded as `KokoControl`. A binary frame is yielded as
   * `KokoAudio` using the most recent `audio.rate` announcement. The
   * announcement itself is also yielded as `KokoControl`, so consumers only
   * interested in playable frames can ignore controls whose `type` is
   * `"audio"`; use `kind === "audio"` to identify PCM payloads.
   *
   * Only one consumer should iterate this generator at a time. The generator
   * reports connection closure by throwing rather than silently returning.
   */
  async *messages(): AsyncGenerator<KokoMessage> {
    while (true) {
      const pending = await this.next();
      if (pending.kind === "error") throw pending.error;
      yield pending.value;
    }
  }

  private async sendControl(data: Record<string, unknown>): Promise<void> {
    // Control commands are JSON text frames. Audio uses a separate binary
    // frame so the server never has to inspect or decode an envelope.
    const socket = this.requireSocket();
    await this.waitUntilOpen(socket);
    socket.send(JSON.stringify(data));
  }

  private requireSocket(): WebSocket {
    // Fail synchronously and clearly instead of producing a less useful
    // "cannot read properties of undefined" error at the call site.
    if (!this.socket) throw new Error("koko client is not connected");
    return this.socket;
  }

  private waitUntilOpen(socket: WebSocket): Promise<void> {
    // This also covers calls made immediately after construction while a
    // caller is sharing the connection setup with another async task.
    if (socket.readyState === WebSocket.OPEN) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const onOpen = () => {
        cleanup();
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error("koko websocket is not open"));
      };
      const onClose = () => {
        cleanup();
        reject(new Error("koko websocket is not open"));
      };
      const cleanup = () => {
        socket.removeEventListener("open", onOpen);
        socket.removeEventListener("error", onError);
        socket.removeEventListener("close", onClose);
      };
      socket.addEventListener("open", onOpen);
      socket.addEventListener("error", onError);
      socket.addEventListener("close", onClose);
    });
  }

  private enqueueMessage(socket: WebSocket, data: unknown): void {
    // Blob.arrayBuffer() is asynchronous. Chain all frames to retain websocket
    // order, otherwise a following audio header could change the Blob's rate.
    this.receiveChain = this.receiveChain
      .then(() => this.handleMessage(socket, data))
      .catch((error: unknown) => {
        if (this.socket === socket) this.end(toError(error));
      });
  }

  private async handleMessage(socket: WebSocket, data: unknown): Promise<void> {
    if (this.socket !== socket) return;
    // Bun normally supplies ArrayBuffer because binaryType is set above. Blob
    // handling remains here so the API also works with WebSocket runtimes
    // that choose Blob for binary events.
    if (typeof data === "string") {
      const control = JSON.parse(data) as Record<string, unknown>;
      const type = control.type;
      if (typeof type !== "string") throw new Error("invalid koko control message");
      if (type === "audio" && typeof control.rate === "number") {
        this.audioRate = control.rate;
      }
      this.push({ kind: "message", value: { kind: "control", type, data: control } });
      return;
    }
    if (typeof Blob !== "undefined" && data instanceof Blob) {
      this.pushAudio(await data.arrayBuffer());
      return;
    }
    if (data instanceof ArrayBuffer) {
      this.pushAudio(data);
      return;
    }
    if (ArrayBuffer.isView(data)) {
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      this.pushAudio(bytes.slice().buffer);
      return;
    }
    throw new Error("invalid koko binary message");
  }

  private pushAudio(buffer: ArrayBuffer): void {
    // Copy the bytes into a Uint8Array view with no PCM interpretation. The
    // consumer decides whether to play bytes directly or decode samples.
    this.push({
      kind: "message",
      value: { kind: "audio", pcm16: new Uint8Array(buffer), rate: this.audioRate },
    });
  }

  private push(pending: Pending): void {
    // Resolve an awaiting generator before buffering. This keeps the normal
    // streaming path allocation-light while preserving arrival order.
    if (this.ended) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter(pending);
    else this.pending.push(pending);
  }

  private next(): Promise<Pending> {
    // Once ended, every future iterator read observes the same terminal
    // failure instead of waiting forever.
    const pending = this.pending.shift();
    if (pending) return Promise.resolve(pending);
    if (this.ended) return Promise.resolve({ kind: "error", error: this.ended });
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  private end(error: Error): void {
    // WebSocket events can report close/error more than once. Only the first
    // terminal condition is meaningful to API consumers.
    if (this.ended) return;
    this.ended = error;
    for (const waiter of this.waiters.splice(0)) waiter({ kind: "error", error });
  }
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}
