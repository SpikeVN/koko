import { For, Show, createEffect, createSignal, onCleanup, onMount } from 'solid-js';
import { ArrowRight, ChevronDown, Mic, MicOff, Volume2, VolumeX } from 'lucide-solid';
import * as ort from 'onnxruntime-web/wasm';
import nanoWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';

const SERVER_URL = 'wss://koko-api.falcolabs.org';
const INPUT_RATE = 16_000;
const INPUT_BLOCK_SIZE = 1024;

type Voice = [string, string];
type NanoVoice = { description: string; speaker_emb: number[]; style: number[][] };
type CaptureSource = 'microphone' | 'tab';
type SpeechSource = 'cloud' | 'nano';

const MOTION_STORAGE_KEY = 'koko-motion-enabled';
const SPEECH_SOURCE_STORAGE_KEY = 'koko-speech-source';
const NANO_REVISION = 'aba295eb96a6fa6003ebe417cc1f2802a7adc1dc';
const NANO_CACHE = `koko-vieneu-nano-${NANO_REVISION}`;
const NANO_BASE_URL = `https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/${NANO_REVISION}`;
const NANO_VOICES_URL = '/assets/voices_v3_nano.json';
const NANO_FILES = [
  'config.json', 'constants.npz', 'text_encoder.onnx', 'duration_predictor.onnx',
  'vector_estimator.onnx', 'codec_decoder.onnx', 'codec_encoder.onnx', 'denoiser.onnx',
  'reference_encoder.onnx', 'speaker_encoder.onnx',
];
const NANO_FILE_SIZES: Record<string, number> = {
  'config.json': 2_927, 'constants.npz': 53_448, 'text_encoder.onnx': 26_519_943,
  'duration_predictor.onnx': 727_809, 'vector_estimator.onnx': 155_132_418,
  'codec_decoder.onnx': 99_319_941, 'codec_encoder.onnx': 56_419_417,
  'denoiser.onnx': 42_661_414, 'reference_encoder.onnx': 10_778_145,
  'speaker_encoder.onnx': 28_303_423,
};

function Icon(props: { name: 'mic' | 'mic-off' | 'volume' | 'volume-off' | 'chevron' | 'arrow'; class?: string }) {
  return <Show when={props.name === 'mic'} fallback={<Show when={props.name === 'mic-off'} fallback={<Show when={props.name === 'volume'} fallback={<Show when={props.name === 'volume-off'} fallback={<Show when={props.name === 'chevron'} fallback={<ArrowRight class={props.class} aria-hidden="true" />}><ChevronDown class={props.class} aria-hidden="true" /></Show>}><VolumeX class={props.class} aria-hidden="true" /></Show>}><Volume2 class={props.class} aria-hidden="true" /></Show>}><MicOff class={props.class} aria-hidden="true" /></Show>}><Mic class={props.class} aria-hidden="true" /></Show>;
}

const languages = [
  { code: 'en', label: 'English' },
  { code: 'vi', label: 'Tiếng Việt' },
  { code: 'zh', label: '中文' },
  { code: 'ja', label: '日本語' },
  { code: 'ko', label: '한국어' },
  { code: 'fr', label: 'Français' },
  { code: 'de', label: 'Deutsch' },
  { code: 'es', label: 'Español' },
  { code: 'it', label: 'Italiano' },
  { code: 'pt', label: 'Português' },
  { code: 'ru', label: 'Русский' },
  { code: 'th', label: 'ไทย' },
  { code: 'id', label: 'Bahasa Indonesia' },
  { code: 'ar', label: 'العربية' },
  { code: 'hi', label: 'हिन्दी' },
];
const targetLanguageNames: Record<string, string> = {
  en: 'English', vi: 'tiếng Việt', zh: 'Chinese', ja: 'Japanese', ko: 'Korean',
  fr: 'French', de: 'German', es: 'Spanish', it: 'Italian', pt: 'Portuguese',
  ru: 'Russian', th: 'Thai', id: 'Indonesian', ar: 'Arabic', hi: 'Hindi',
};

function LoadingScreen(props: { motionEnabled?: boolean; onDismiss?: () => void }) {
  return (
    <main
      class="relative mx-auto h-dvh w-screen max-w-[402px] overflow-hidden bg-white select-none"
      onClick={() => props.onDismiss?.()}
    >
      <div class="absolute top-1/2 left-0 h-[496px] w-full -translate-y-1/2">
        <div class="absolute top-[142px] left-1/2 h-[354px] w-[395px] -translate-x-1/2" aria-hidden="true">
          {/* Gentle sway wrapper: sways both globe and flags together */}
          <div
            class="relative size-full origin-[197.5px_158px]"
            classList={{
              'animate-koko-sway': props.motionEnabled !== false,
            }}
          >
            {/* Globe: rotates smoothly on its own center */}
            <div
              class="absolute top-[28px] left-[67px] size-[260px] origin-center"
              classList={{
                'animate-koko-spin': props.motionEnabled !== false,
              }}
            >
              <img class="size-full object-contain pointer-events-none select-none" src="/assets/earth.png" alt="Globe" />
            </div>

            {/* Orbit container: flags rotate along with the globe around earth's center */}
            <div
              class="absolute inset-0 origin-[197.5px_158px]"
              classList={{
                'animate-koko-orbit': props.motionEnabled !== false,
              }}
            >
              {/* US Flag */}
              <div
                class="absolute top-[13px] left-0 h-[83px] w-[111px] origin-center"
                classList={{
                  'animate-koko-counter-spin': props.motionEnabled !== false,
                }}
              >
                <div
                  class="size-full origin-center"
                  classList={{
                    'animate-koko-flag-sway': props.motionEnabled !== false,
                  }}
                  style={{ "animation-delay": "0s" }}
                >
                  <img class="size-full object-contain pointer-events-none select-none" src="/assets/us-flag.png" alt="US Flag" />
                </div>
              </div>

              {/* Vietnam Flag */}
              <div
                class="absolute top-0 left-[280px] h-[106px] w-[111px] origin-center"
                classList={{
                  'animate-koko-counter-spin': props.motionEnabled !== false,
                }}
              >
                <div
                  class="size-full origin-center"
                  classList={{
                    'animate-koko-flag-sway': props.motionEnabled !== false,
                  }}
                  style={{ "animation-delay": "0.6s" }}
                >
                  <img class="size-full object-contain pointer-events-none select-none" src="/assets/vietnam-flag.png" alt="Vietnam Flag" />
                </div>
              </div>

              {/* China Flag */}
              <div
                class="absolute top-[234px] left-[14px] h-[114px] w-[115px] origin-center"
                classList={{
                  'animate-koko-counter-spin': props.motionEnabled !== false,
                }}
              >
                <div
                  class="size-full origin-center"
                  classList={{
                    'animate-koko-flag-sway': props.motionEnabled !== false,
                  }}
                  style={{ "animation-delay": "1.2s" }}
                >
                  <img class="size-full object-contain pointer-events-none select-none" src="/assets/china-flag.png" alt="China Flag" />
                </div>
              </div>

              {/* Japan Flag */}
              <div
                class="absolute top-[232px] left-[251px] h-[122px] w-[144px] origin-center"
                classList={{
                  'animate-koko-counter-spin': props.motionEnabled !== false,
                }}
              >
                <div
                  class="size-full origin-center"
                  classList={{
                    'animate-koko-flag-sway': props.motionEnabled !== false,
                  }}
                  style={{ "animation-delay": "1.8s" }}
                >
                  <img class="size-full object-contain pointer-events-none select-none" src="/assets/japan-flag.png" alt="Japan Flag" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="absolute top-[58px] left-1/2 w-[126px] -translate-x-1/2 text-center">
        <img class="mx-auto mb-[11px] block h-[71px] w-[102px] object-contain" src="/assets/mascot-56cba3.png" />
        <strong class="text-[20px] leading-[25px]">Project Koko</strong>
      </div>
      <small class="absolute bottom-[22px] left-0 w-full text-center text-[14px]">© 2026 Project Koko</small>
    </main>
  );
}

function WiggleBorder(props: { type?: 'box' | 'pill'; motionEnabled: boolean }) {
  let path!: SVGPathElement;
  let svg!: SVGSVGElement;
  let animationFrame = 0;
  let lastFrame = 0;
  let pillRadiusX = 8.5;
  const points = props.type === 'pill' ? 36 : 28;
  const inset = 0.75;
  const span = 100 - inset * 2;

  const noise = (seed: number) => {
    const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
    return (value - Math.floor(value)) - 0.5;
  };

  const draw = (timestamp: number) => {
    animationFrame = requestAnimationFrame(draw);
    if (timestamp - lastFrame < 1000 / 6) return;
    lastFrame = timestamp;
    const frame = Math.floor(timestamp / (1000 / 6));
    const jitter = (index: number) => props.motionEnabled ? noise(frame * 17.13 + index * 9.71) * 0.7 : 0;
    let d = '';

    if (props.type === 'pill') {
      const radiusY = 50 - inset;
      const rightCenter = 100 - inset - pillRadiusX;
      const leftCenter = inset + pillRadiusX;
      const corners = [
        [rightCenter, 50, -Math.PI / 2, 0],
        [rightCenter, 50, 0, Math.PI / 2],
        [leftCenter, 50, Math.PI / 2, Math.PI],
        [leftCenter, 50, Math.PI, Math.PI * 1.5],
      ];
      corners.forEach(([centerX, centerY, start, end], cornerIndex) => {
        for (let index = 0; index <= 6; index += 1) {
          const angle = start + ((end - start) * index) / 6;
          const pointIndex = cornerIndex * 7 + index;
          const x = centerX + pillRadiusX * Math.cos(angle) + jitter(pointIndex);
          const y = centerY + radiusY * Math.sin(angle) + jitter(pointIndex + points);
          d += `${d ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)} `;
        }
      });
      d += 'Z';
    } else {
      const edge = Math.floor(points / 4);
      const sides = [
        (index: number) => [inset + (span * index) / edge, inset + jitter(index)],
        (index: number) => [inset + span + jitter(index + edge), inset + (span * index) / edge],
        (index: number) => [inset + span - (span * index) / edge, inset + span + jitter(index + edge * 2)],
        (index: number) => [inset + jitter(index + edge * 3), inset + span - (span * index) / edge],
      ];
      sides.forEach((side, sideIndex) => {
        for (let index = sideIndex === 0 ? 0 : 1; index <= edge; index += 1) {
          const [x, y] = side(index);
          d += `${d ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)} `;
        }
      });
      d += 'Z';
    }
    path.setAttribute('d', d);
  };

  onMount(() => {
    if (props.type === 'pill') {
      const updatePillRadius = () => {
        const { width, height } = svg.getBoundingClientRect();
        if (width && height) pillRadiusX = Math.min(50 - inset, (50 - inset) * height / width);
      };
      updatePillRadius();
      const resizeObserver = new ResizeObserver(updatePillRadius);
      resizeObserver.observe(svg);
      onCleanup(() => resizeObserver.disconnect());
    }
    animationFrame = requestAnimationFrame(draw);
  });
  onCleanup(() => cancelAnimationFrame(animationFrame));

  return <svg ref={svg} class="pointer-events-none absolute inset-0 z-[1] size-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
    <path ref={path} fill="none" stroke="#000" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke" />
  </svg>;
}

function App() {
  const [loading, setLoading] = createSignal(true);
  const [connected, setConnected] = createSignal(false);
  const [recording, setRecording] = createSignal(false);
  const [sourceText, setSourceText] = createSignal('');
  const [sourcePartial, setSourcePartial] = createSignal('');
  const [translation, setTranslation] = createSignal('');
  const [translationPartial, setTranslationPartial] = createSignal('');
  const [speakerEnabled, setSpeakerEnabled] = createSignal(true);
  const [speechSource, setSpeechSource] = createSignal<SpeechSource>('cloud');
  const [nanoProgress, setNanoProgress] = createSignal<number>();
  const [nanoStatus, setNanoStatus] = createSignal('');
  const [motionEnabled, setMotionEnabled] = createSignal(true);
  const [captureSource, setCaptureSource] = createSignal<CaptureSource>('microphone');
  const [sourceLanguage, setSourceLanguage] = createSignal('en');
  const [targetLanguage, setTargetLanguage] = createSignal('vi');
  const [voices, setVoices] = createSignal<Voice[]>([]);
  const [voice, setVoice] = createSignal('');
  const [nanoVoices, setNanoVoices] = createSignal<Record<string, NanoVoice>>({});
  const [nanoVoice, setNanoVoice] = createSignal('Adam');
  const [error, setError] = createSignal('');

  let socket: WebSocket | undefined;
  let audioContext: AudioContext | undefined;
  let inputContext: AudioContext | undefined;
  let processor: ScriptProcessorNode | undefined;
  let inputSilencer: GainNode | undefined;
  let inputStream: MediaStream | undefined;
  let connection: Promise<void> | undefined;
  let sourcePanel: HTMLParagraphElement | undefined;
  let translationPanel: HTMLParagraphElement | undefined;
  let nextPlaybackTime = 0;
  let outputRate = 48_000;
  const playbackSources = new Set<AudioBufferSourceNode>();
  let nanoSessions: Record<string, ort.InferenceSession> | undefined;
  let nanoConfig: { vocab: Record<string, number>; bos_id: number; eos_id: number; pad_id: number; flow_fps?: number } | undefined;
  let nanoPhonemeId = 0;
  const nanoPhonemeRequests = new Map<string, { resolve: (phonemes: string) => void; reject: (error: Error) => void }>();

  const stopPlayback = () => {
    playbackSources.forEach((source) => source.stop());
    playbackSources.clear();
    nextPlaybackTime = 0;
  };

  const playFloatAudio = (samples: Float32Array, rate: number) => {
    if (!speakerEnabled() || !samples.length) return;
    audioContext ??= new AudioContext();
    const buffer = audioContext.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(new Float32Array(samples), 0);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    playbackSources.add(source);
    source.onended = () => playbackSources.delete(source);
    nextPlaybackTime = Math.max(nextPlaybackTime, audioContext.currentTime);
    source.start(nextPlaybackTime);
    nextPlaybackTime += buffer.duration;
  };

  const getNanoFile = async (cache: Cache, file: string, onProgress: (received: number, total: number) => void) => {
    const url = `${NANO_BASE_URL}/${file}`;
    const cached = await cache.match(url);
    if (cached) {
      const size = Number(cached.headers.get('content-length')) || (await cached.clone().blob()).size;
      onProgress(size, size);
      return cached;
    }
    const response = await fetch(url);
    if (!response.ok || !response.body) throw new Error(`Could not download ${file}.`);
    const total = Number(response.headers.get('content-length')) || 0;
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      onProgress(received, total);
    }
    const stored = new Response(new Blob(chunks), { headers: { 'content-type': response.headers.get('content-type') ?? 'application/octet-stream', 'content-length': String(received) } });
    await cache.put(url, stored.clone());
    return stored;
  };

  const prepareNano = async () => {
    if (nanoSessions) return;
    setNanoStatus('Preparing VieNeu Nano...');
    setNanoProgress(0);
    const cache = await caches.open(NANO_CACHE);
    // Sizes belong to the pinned revision, so total progress is known before downloading.
    const sizes = new Map(Object.entries(NANO_FILE_SIZES));
    const received = new Map<string, number>();
    const updateProgress = (file: string, loaded: number, total: number) => {
      received.set(file, loaded);
      if (total) sizes.set(file, total);
      const loadedBytes = [...received.values()].reduce((sum, value) => sum + value, 0);
      const totalBytes = [...sizes.values()].reduce((sum, value) => sum + value, 0);
      const progress = totalBytes ? Math.min(100, Math.round((loadedBytes / totalBytes) * 100)) : undefined;
      setNanoProgress((current) => progress === undefined ? current : Math.max(current ?? 0, progress));
    };
    const files = new Map<string, Response>();
    for (const file of NANO_FILES) files.set(file, await getNanoFile(cache, file, (loaded, total) => updateProgress(file, loaded, total)));
    setNanoStatus('Starting VieNeu Nano...');
    ort.env.wasm.wasmPaths = { wasm: nanoWasmUrl };
    const sessions = await Promise.all(['text_encoder.onnx', 'duration_predictor.onnx', 'vector_estimator.onnx', 'codec_decoder.onnx'].map(async (file) => [file, await ort.InferenceSession.create(await files.get(file)!.arrayBuffer(), { executionProviders: ['wasm'] })] as const));
    nanoSessions = Object.fromEntries(sessions);
    nanoConfig = await files.get('config.json')!.json();
    const voiceResponse = await fetch(NANO_VOICES_URL);
    if (!voiceResponse.ok) throw new Error('Could not download VieNeu Nano voice presets.');
    const voiceData = await voiceResponse.json() as { default_voice?: string; presets?: Record<string, NanoVoice> };
    const presets = voiceData.presets ?? {};
    setNanoVoices(presets);
    if (!presets[nanoVoice()]) setNanoVoice(voiceData.default_voice ?? Object.keys(presets)[0] ?? '');
    setNanoProgress(100);
    setNanoStatus('VieNeu Nano model cached');
  };

  const phonemizeNano = (text: string) => new Promise<string>((resolve, reject) => {
    if (socket?.readyState !== WebSocket.OPEN) {
      reject(new Error('Connect to Project Koko before using VieNeu Nano.'));
      return;
    }
    const id = String(++nanoPhonemeId);
    nanoPhonemeRequests.set(id, { resolve, reject });
    send({ type: 'nano_phonemize', id, text });
  });

  const normal = () => {
    let spare: number | undefined;
    return () => {
      if (spare !== undefined) {
        const value = spare;
        spare = undefined;
        return value;
      }
      let u = 0;
      let v = 0;
      while (!u) u = Math.random();
      while (!v) v = Math.random();
      const radius = Math.sqrt(-2 * Math.log(u));
      spare = radius * Math.sin(2 * Math.PI * v);
      return radius * Math.cos(2 * Math.PI * v);
    };
  };

  const speakNano = async (text: string) => {
    if (speechSource() !== 'nano' || !text) return;
    if (targetLanguage() !== 'vi') {
      setError('VieNeu Nano currently speaks Vietnamese only. Set the target language to Tiếng Việt.');
      return;
    }
    try {
      await prepareNano();
      const sessions = nanoSessions;
      const config = nanoConfig;
      const preset = nanoVoices()[nanoVoice()];
      if (!sessions || !config || !preset) throw new Error('VieNeu Nano is not ready.');
      const phonemes = await phonemizeNano(text);
      const ids = [config.bos_id, ...[...phonemes].flatMap((character) => config.vocab[character] === undefined ? [] : [config.vocab[character]]), config.eos_id];
      const length = ids.length;
      const idTensor = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, length]);
      const mask = new ort.Tensor('bool', new Uint8Array(length).fill(1), [1, length]);
      const style = new ort.Tensor('float32', Float32Array.from(preset.style.flat()), [1, 50, 256]);
      const speaker = new ort.Tensor('float32', Float32Array.from(preset.speaker_emb), [1, 192]);
      const ctx = (await sessions['text_encoder.onnx'].run({ ids: idTensor, style })).ctx as ort.Tensor;
      const logSeconds = (await sessions['duration_predictor.onnx'].run({ ctx, ctx_mask: mask, spk: speaker })).log_seconds.data as Float32Array;
      const frames = Math.max(2, Math.round(Math.min(Math.exp(logSeconds[0]), 15) * (config.flow_fps ?? 15.625)));
      const samples = new Float32Array(144 * frames);
      const random = normal();
      for (let index = 0; index < samples.length; index += 1) samples[index] = random();
      let x = new ort.Tensor('float32', samples, [1, 144, frames]);
      const steps = 8;
      for (let index = 0; index < steps; index += 1) {
        const time = index / steps;
        const result = await sessions['vector_estimator.onnx'].run({ x, t: new ort.Tensor('float32', Float32Array.of(time), [1]), ctx, ctx_mask: mask, spk: speaker, style });
        const velocity = result.v.data as Float32Array;
        const next = new Float32Array(samples.length);
        for (let sample = 0; sample < next.length; sample += 1) next[sample] = (x.data as Float32Array)[sample] + velocity[sample] / steps;
        x = new ort.Tensor('float32', next, [1, 144, frames]);
      }
      const waveform = (await sessions['codec_decoder.onnx'].run({ x })).wav.data as Float32Array;
      for (let index = 0; index < waveform.length; index += 1) waveform[index] = Math.max(-1, Math.min(1, waveform[index]));
      playFloatAudio(waveform, 24_000);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'VieNeu Nano could not synthesize this translation.');
    }
  };

  onMount(() => {
    const savedMotion = window.localStorage.getItem(MOTION_STORAGE_KEY);
    if (savedMotion !== null) setMotionEnabled(savedMotion === 'true');
    const savedSpeechSource = window.localStorage.getItem(SPEECH_SOURCE_STORAGE_KEY);
    if (savedSpeechSource === 'cloud' || savedSpeechSource === 'nano') setSpeechSource(savedSpeechSource);
    const savedNanoVoice = window.localStorage.getItem('koko-nano-voice');
    if (savedNanoVoice) setNanoVoice(savedNanoVoice);
    const timer = window.setTimeout(() => setLoading(false), 1500);
    onCleanup(() => window.clearTimeout(timer));
  });

  const send = (message: object) => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  };

  const handleMessage = async (event: MessageEvent) => {
    if (typeof event.data !== 'string') {
      const bytes = event.data instanceof ArrayBuffer ? event.data : await event.data.arrayBuffer();
      if (speechSource() !== 'cloud' || !speakerEnabled()) return;
      if (!audioContext) audioContext = new AudioContext({ sampleRate: outputRate });
      const samples = new Int16Array(bytes);
      const buffer = audioContext.createBuffer(1, samples.length, outputRate);
      const channel = buffer.getChannelData(0);
      for (let i = 0; i < samples.length; i += 1) channel[i] = samples[i] / 32768;
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      playbackSources.add(source);
      source.onended = () => playbackSources.delete(source);
      nextPlaybackTime = Math.max(nextPlaybackTime, audioContext.currentTime);
      source.start(nextPlaybackTime);
      nextPlaybackTime += buffer.duration;
      return;
    }
    const message = JSON.parse(event.data) as Record<string, unknown>;
    if (message.type === 'audio') {
      outputRate = Number(message.rate) || 48_000;
    } else if (message.type === 'nano_phonemes') {
      const request = nanoPhonemeRequests.get(String(message.id ?? ''));
      if (request) {
        nanoPhonemeRequests.delete(String(message.id));
        request.resolve(String(message.phonemes ?? ''));
      }
    } else if (message.type === 'ready') {
      setConnected(true);
      const serverVoices = (message.tts_voices as Voice[] | undefined) ?? [];
      setVoices(serverVoices);
      setVoice((message.tts_voice as string | undefined) ?? serverVoices[0]?.[0] ?? '');
    } else if (message.type === 'partial') {
      setSourcePartial(String(message.text ?? ''));
    } else if (message.type === 'final') {
      const text = String(message.text ?? '');
      if (text) setSourceText((current) => current ? `${current} ${text}` : text);
      setSourcePartial('');
    }
    else if (message.type === 'speak') {
      setTranslationPartial('');
    } else if (message.type === 'translation_partial') {
      setTranslationPartial(String(message.text ?? ''));
    } else if (message.type === 'translation') {
      const text = String(message.text ?? '');
      if (text) setTranslation((current) => current ? `${current} ${text}` : text);
      setTranslationPartial('');
      if (text) void speakNano(text);
    }
    else if (message.type === 'error') setError(String(message.detail ?? 'Server error'));
  };

  const connect = () => {
    if (socket?.readyState === WebSocket.OPEN) return Promise.resolve();
    if (socket?.readyState === WebSocket.CONNECTING && connection) return connection;
    setError('');
    socket = new WebSocket(SERVER_URL);
    socket.binaryType = 'arraybuffer';
    connection = new Promise((resolve, reject) => {
      socket!.onopen = () => {
        send({
          type: 'hello',
          language: sourceLanguage(),
          target_language: targetLanguageNames[targetLanguage()],
        });
        resolve();
      };
      socket!.onerror = () => {
        setError('Unable to connect to Project Koko.');
        reject(new Error('Unable to connect to Project Koko.'));
      };
    });
    socket.onmessage = (event) => void handleMessage(event);
    socket.onclose = () => { connection = undefined; setConnected(false); setRecording(false); };
    return connection;
  };

  const floatToPcm = (input: Float32Array, inputRate: number) => {
    const ratio = inputRate / INPUT_RATE;
    const length = Math.round(input.length / ratio);
    const output = new ArrayBuffer(length * 2);
    const view = new DataView(output);
    for (let i = 0; i < length; i += 1) {
      const sample = Math.max(-1, Math.min(1, input[Math.min(input.length - 1, Math.floor(i * ratio))]));
      view.setInt16(i * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
    }
    return output;
  };

  const startRecording = async () => {
    try {
      await connect();
      if (!audioContext) audioContext = new AudioContext({ sampleRate: outputRate });
      if (audioContext.state === 'suspended') await audioContext.resume();
      inputStream ??= captureSource() === 'microphone'
        ? await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
        : await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
      if (!inputStream.getAudioTracks().length) throw new DOMException('No audio track', 'NotFoundError');
      inputContext = new AudioContext();
      const source = inputContext.createMediaStreamSource(inputStream);
      processor = inputContext.createScriptProcessor(INPUT_BLOCK_SIZE, 1, 1);
      inputSilencer = inputContext.createGain();
      inputSilencer.gain.value = 0;
      processor.onaudioprocess = (event) => {
        if (recording()) socket?.readyState === WebSocket.OPEN && socket.send(floatToPcm(event.inputBuffer.getChannelData(0), inputContext?.sampleRate ?? 48_000));
      };
      source.connect(processor);
      processor.connect(inputSilencer);
      inputSilencer.connect(inputContext.destination);
      setRecording(true);
    } catch (error) {
      setError(error instanceof DOMException
        ? captureSource() === 'tab' ? 'Share a browser tab and enable its audio to start interpreting.' : 'Microphone access is needed to start interpreting.'
        : 'Unable to connect to Project Koko.');
    }
  };

  const stopRecording = () => {
    processor?.disconnect();
    inputSilencer?.disconnect();
    inputStream?.getTracks().forEach((track) => track.stop());
    void inputContext?.close();
    processor = undefined;
    inputSilencer = undefined;
    inputStream = undefined;
    setRecording(false);
  };

  const toggleRecording = () => recording() ? stopRecording() : void startRecording();
  const toggleSpeaker = () => {
    setSpeakerEnabled((enabled) => !enabled);
    if (!speakerEnabled()) {
      stopPlayback();
    }
    if (speakerEnabled() && audioContext?.state === 'suspended') void audioContext.resume();
  };
  const updateSpeechSource = async (selected: SpeechSource) => {
    stopPlayback();
    if (selected === 'nano') {
      try {
        await prepareNano();
        setNanoStatus('');
        setNanoProgress(undefined);
      } catch (error) {
        setNanoProgress(undefined);
        setNanoStatus('');
        setError(error instanceof Error ? error.message : 'Unable to prepare VieNeu Nano.');
        return;
      }
    }
    setError('');
    setSpeechSource(selected);
  };
  const updateSourceLanguage = (language: string) => { setSourceLanguage(language); send({ type: 'asr_language', language }); };
  const updateTargetLanguage = (language: string) => {
    setTargetLanguage(language);
    send({ type: 'target_language', language: targetLanguageNames[language] });
  };
  const updateVoice = (selected: string) => { setVoice(selected); send({ type: 'tts_voice', voice: selected }); };
  const updateCaptureSource = async (selected: CaptureSource) => {
    const wasRecording = recording();
    if (wasRecording) stopRecording();
    if (selected === 'tab') {
      try {
        inputStream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
        if (!inputStream.getAudioTracks().length) {
          inputStream.getTracks().forEach((track) => track.stop());
          inputStream = undefined;
          throw new DOMException('No audio track', 'NotFoundError');
        }
        setCaptureSource('tab');
      } catch {
        setError('Share a browser tab and enable its audio to use tab capture.');
        return;
      }
    } else {
      inputStream?.getTracks().forEach((track) => track.stop());
      inputStream = undefined;
      setCaptureSource('microphone');
    }
    if (wasRecording) void startRecording();
  };
  const clearContext = () => {
    setSourceText('');
    setSourcePartial('');
    setTranslation('');
    setTranslationPartial('');
    send({ type: 'clear_context' });
  };

  const sourceLabel = () => languages.find((l) => l.code === sourceLanguage())?.label ?? sourceLanguage();
  const targetLabel = () => languages.find((l) => l.code === targetLanguage())?.label ?? targetLanguage();
  const voiceLabel = () => voice() || 'Giọng mặc định';
  const captureLabel = () => captureSource() === 'microphone' ? 'Micro' : 'Browser tab';

  createEffect(() => {
    if (!loading() && !socket) void connect().catch(() => undefined);
  });

  createEffect(() => {
    window.localStorage.setItem(MOTION_STORAGE_KEY, String(motionEnabled()));
  });

  createEffect(() => {
    window.localStorage.setItem(SPEECH_SOURCE_STORAGE_KEY, speechSource());
  });

  createEffect(() => {
    window.localStorage.setItem('koko-nano-voice', nanoVoice());
  });

  createEffect(() => {
    sourceText();
    sourcePartial();
    sourcePanel?.scrollTo({ top: sourcePanel.scrollHeight });
  });

  createEffect(() => {
    translation();
    translationPartial();
    translationPanel?.scrollTo({ top: translationPanel.scrollHeight });
  });

  onCleanup(() => {
    stopRecording();
    stopPlayback();
    send({ type: 'bye' });
    socket?.close();
    void audioContext?.close();
  });

  return <Show when={!loading()} fallback={<LoadingScreen motionEnabled={motionEnabled()} onDismiss={() => setLoading(false)} />}>
    <main class="relative mx-auto flex min-h-dvh w-full flex-col items-center gap-6 overflow-y-auto bg-white px-6 py-10 md:h-dvh md:flex-row md:gap-12 md:overflow-hidden md:px-12">
      <section class="flex w-full max-w-sm flex-col gap-6 md:h-[calc(100dvh-4rem)] md:max-w-none md:flex-1">
        <section class="relative flex min-h-52 flex-1 flex-col md:min-h-0">
          <div class="pointer-events-none absolute -top-20 -right-2 z-0 h-44 w-40"><img class="absolute top-0 left-6 h-24 w-28 object-contain" src="/assets/dashboard-flag.png" /><img class="absolute bottom-0 left-0 h-32 w-36 object-contain" src="/assets/dashboard-mascot-688b49.png" /></div>
          <h2 class="relative z-[2] mb-2 text-sm font-semibold">Transcription</h2>
          <article class="relative z-[1] min-h-0 flex-1 rounded-2xl pb-4">
            <WiggleBorder motionEnabled={motionEnabled()} />
            <div class="pointer-events-none absolute inset-0 z-0 rounded-2xl bg-[url('/assets/grid_paper.png')] bg-repeat opacity-95" />
            <p ref={sourcePanel} class="relative z-[2] h-full overflow-y-auto px-8 py-5 text-sm leading-relaxed"><Show when={sourceText() || sourcePartial()} fallback={<span class="text-neutral-500">{recording() ? 'Listening...' : 'Your words will appear here'}</span>}><span>{sourceText()}</span><Show when={sourcePartial()}><span>{sourceText() && ' '}</span><span class="text-neutral-500">{sourcePartial()}</span></Show></Show></p>
          </article>
        </section>
        <section class="relative flex min-h-52 flex-1 flex-col md:min-h-0">
          <h2 class="relative z-[2] mb-2 text-sm font-semibold">Translation</h2>
          <article class="relative z-[1] min-h-0 flex-1 rounded-2xl pb-4">
            <WiggleBorder motionEnabled={motionEnabled()} />
            <div class="pointer-events-none absolute inset-0 z-0 rounded-2xl bg-[url('/assets/grid_paper.png')] bg-repeat opacity-95" />
            <p ref={translationPanel} class="relative z-[2] h-full overflow-y-auto px-8 py-5 text-sm leading-relaxed"><Show when={translation() || translationPartial()} fallback={<span class="text-neutral-500">Your translation will appear here</span>}><span>{translation()}</span><Show when={translationPartial()}><span>{translation() && ' '}</span><span class="text-neutral-500">{translationPartial()}</span></Show></Show></p>
          </article>
        </section>
      </section>
      <aside class="flex w-full max-w-sm flex-col gap-6 md:w-84 md:shrink-0">
        <section class="grid grid-cols-[1fr_auto_1fr] items-center gap-3" aria-label="Language switcher">
          <label class="relative flex min-h-12 items-center rounded-full"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-white/70" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-sm text-black outline-none" value={sourceLanguage()} onChange={(event) => updateSourceLanguage(event.currentTarget.value)}><For each={languages}>{(language) => <option value={language.code}>{language.label}</option>}</For></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label>
          <Icon name="arrow" class="size-5 shrink-0" />
          <label class="relative flex min-h-12 items-center rounded-full"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-white/70" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-sm text-black outline-none" value={targetLanguage()} onChange={(event) => updateTargetLanguage(event.currentTarget.value)}><For each={languages}>{(language) => <option value={language.code}>{language.label}</option>}</For></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label>
        </section>
        <section class="flex flex-col gap-4">
          <h2 class="text-sm font-semibold">Phát phiên dịch</h2>
          <label class="relative flex min-h-12 items-center rounded-full text-left text-sm"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-white/70" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-black outline-none" value={speechSource()} onChange={(event) => void updateSpeechSource(event.currentTarget.value as SpeechSource)}><option value="cloud">Cloud</option><option value="nano">Browser (VieNeu Nano)</option></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label>
          <Show when={nanoStatus()}><div class="text-xs" role="status"><div class="mb-1 flex justify-between"><span>{nanoStatus()}</span><Show when={nanoProgress() !== undefined}><span>{nanoProgress()}%</span></Show></div><Show when={nanoProgress() !== undefined}><div class="h-1 overflow-hidden rounded bg-neutral-200"><div class="h-full bg-black transition-[width]" style={{ width: `${nanoProgress()}%` }} /></div></Show></div></Show>
          <Show when={speechSource() === 'cloud'}><label class="relative flex min-h-12 items-center rounded-full text-left text-sm"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-[url('/assets/voice_bg.png')] bg-cover bg-center" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-black outline-none" value={voice()} onChange={(event) => updateVoice(event.currentTarget.value)}><option value="">Giọng mặc định</option><For each={voices()}>{(item) => <option value={item[0]}>{item[0]}</option>}</For></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label></Show>
          <Show when={speechSource() === 'nano'}><label class="relative flex min-h-12 items-center rounded-full text-left text-sm"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-[url('/assets/voice_bg.png')] bg-cover bg-center" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-black outline-none" value={nanoVoice()} onChange={(event) => setNanoVoice(event.currentTarget.value)}><For each={Object.entries(nanoVoices())}>{([name, preset]) => <option value={name}>{preset.description ? `${name} - ${preset.description}` : name}</option>}</For></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label></Show>
          <label class="relative flex min-h-12 items-center rounded-full text-left text-sm"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-white/70" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-black outline-none" value={captureSource()} onChange={(event) => void updateCaptureSource(event.currentTarget.value as CaptureSource)}><option value="microphone">Microphone</option><option value="tab">Browser tab</option></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label>
        </section>
        <section class="grid grid-cols-2 gap-4">
          <button class="flex h-14 items-center justify-center gap-2 rounded-full border border-black bg-neutral-900 text-sm text-white transition-colors hover:bg-neutral-700 [&.active]:bg-white [&.active]:text-black" classList={{ active: recording() }} onClick={toggleRecording} type="button" aria-label={recording() ? 'Mute microphone' : 'Enable microphone'}><Icon class="size-6" name={recording() ? 'mic' : 'mic-off'} /><span>Mic</span></button>
          <button class="flex h-14 items-center justify-center gap-2 rounded-full border border-black bg-neutral-900 text-sm text-white transition-colors hover:bg-neutral-700 [&.active]:bg-white [&.active]:text-black" classList={{ active: speakerEnabled() }} onClick={toggleSpeaker} type="button" aria-label={speakerEnabled() ? 'Mute speaker' : 'Enable speaker'}><Icon class="size-6" name={speakerEnabled() ? 'volume' : 'volume-off'} /><span>Speaker</span></button>
        </section>
        <button class="h-12 rounded-full border border-black text-sm transition-colors hover:bg-neutral-100" type="button" onClick={clearContext}>Clear context</button>
        <label class="flex cursor-pointer items-center gap-2 self-end text-xs text-black"><input class="size-4 accent-black" type="checkbox" checked={!motionEnabled()} onChange={(event) => setMotionEnabled(!event.currentTarget.checked)} /><span>Disable motion</span></label>
      </aside>
      <Show when={error()}><p class="max-w-sm text-center text-xs text-red-800 md:absolute md:bottom-4 md:left-1/2 md:-translate-x-1/2">{error()}</p></Show>
      <div class="absolute top-4 right-6 flex items-center gap-2 text-xs opacity-70 md:right-12"><span class="size-2 rounded-full bg-neutral-400" classList={{ 'bg-emerald-600': connected() }}></span>{connected() ? 'Connected' : 'Connecting...'}</div>
    </main>
  </Show>;
}

export default App;
