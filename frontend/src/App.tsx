import { For, Show, createEffect, createSignal, onCleanup, onMount } from 'solid-js';
import { ArrowRight, ChevronDown, Mic, MicOff, Volume2, VolumeX } from 'lucide-solid';

const SERVER_URL = 'wss://koko-api.falcolabs.org';
const INPUT_RATE = 16_000;
const INPUT_BLOCK_SIZE = 1024;

type Voice = [string, string];
type CaptureSource = 'microphone' | 'tab';

const MOTION_STORAGE_KEY = 'koko-motion-enabled';

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

function LoadingScreen() {
  return (
    <main class="relative mx-auto h-dvh w-screen max-w-[402px] overflow-hidden bg-white">
      <div class="absolute top-1/2 left-0 h-[496px] w-full -translate-y-1/2">
        <div class="absolute top-[142px] left-1/2 h-[354px] w-[395px] -translate-x-1/2" aria-hidden="true">
          <img class="absolute top-[29px] left-[65px] h-[258px] w-[265px] rounded-[65px] object-cover" src="/assets/earth.png" />
          <img class="absolute top-[13px] left-0 h-[83px] w-[111px] object-contain" src="/assets/us-flag.png" />
          <img class="absolute top-0 left-[280px] h-[106px] w-[111px] object-contain" src="/assets/vietnam-flag.png" />
          <img class="absolute top-[234px] left-[14px] h-[114px] w-[115px] object-contain" src="/assets/china-flag.png" />
          <img class="absolute top-[232px] left-[251px] h-[122px] w-[144px] object-contain" src="/assets/japan-flag.png" />
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
  const [motionEnabled, setMotionEnabled] = createSignal(true);
  const [captureSource, setCaptureSource] = createSignal<CaptureSource>('microphone');
  const [sourceLanguage, setSourceLanguage] = createSignal('en');
  const [targetLanguage, setTargetLanguage] = createSignal('vi');
  const [voices, setVoices] = createSignal<Voice[]>([]);
  const [voice, setVoice] = createSignal('');
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

  onMount(() => {
    const savedMotion = window.localStorage.getItem(MOTION_STORAGE_KEY);
    if (savedMotion !== null) setMotionEnabled(savedMotion === 'true');
    const timer = window.setTimeout(() => setLoading(false), 1500);
    onCleanup(() => window.clearTimeout(timer));
  });

  const send = (message: object) => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  };

  const handleMessage = async (event: MessageEvent) => {
    if (typeof event.data !== 'string') {
      const bytes = event.data instanceof ArrayBuffer ? event.data : await event.data.arrayBuffer();
      if (!audioContext) audioContext = new AudioContext({ sampleRate: outputRate });
      if (!speakerEnabled()) return;
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
        send({ type: 'hello', language: sourceLanguage() });
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
      playbackSources.forEach((source) => source.stop());
      nextPlaybackTime = 0;
    }
    if (speakerEnabled() && audioContext?.state === 'suspended') void audioContext.resume();
  };
  const updateSourceLanguage = (language: string) => { setSourceLanguage(language); send({ type: 'asr_language', language }); };
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

  createEffect(() => {
    if (!loading() && !socket) void connect().catch(() => undefined);
  });

  createEffect(() => {
    window.localStorage.setItem(MOTION_STORAGE_KEY, String(motionEnabled()));
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
    send({ type: 'bye' });
    socket?.close();
    void audioContext?.close();
  });

  return <Show when={!loading()} fallback={<LoadingScreen />}>
    <main class="relative mx-auto flex min-h-dvh w-full flex-col items-center gap-6 overflow-y-auto bg-white px-6 py-10 md:h-dvh md:flex-row md:gap-12 md:overflow-hidden md:px-12">
      <section class="flex w-full max-w-sm flex-col gap-6 md:h-[calc(100dvh-4rem)] md:max-w-none md:flex-1">
        <section class="relative flex min-h-52 flex-1 flex-col md:min-h-0">
          <div class="pointer-events-none absolute -top-20 -right-2 z-0 h-44 w-40"><img class="absolute top-0 left-6 h-24 w-28 object-contain" src="/assets/dashboard-flag.png" /><img class="absolute bottom-0 left-0 h-32 w-36 object-contain" src="/assets/dashboard-mascot-688b49.png" /></div>
          <h2 class="relative z-[2] mb-2 text-sm font-semibold">Transcription</h2>
          <article class="relative z-[1] min-h-0 flex-1 rounded-2xl pb-4">
            <WiggleBorder motionEnabled={motionEnabled()} />
            <div class="pointer-events-none absolute inset-0 z-0 rounded-2xl bg-white/85" />
            <p ref={sourcePanel} class="relative z-[2] h-full overflow-y-auto px-8 py-5 text-sm leading-relaxed"><Show when={sourceText() || sourcePartial()} fallback={<span class="text-neutral-500">{recording() ? 'Listening...' : 'Your words will appear here'}</span>}><span>{sourceText()}</span><Show when={sourcePartial()}><span>{sourceText() && ' '}</span><span class="text-neutral-500">{sourcePartial()}</span></Show></Show></p>
          </article>
        </section>
        <section class="relative flex min-h-52 flex-1 flex-col md:min-h-0">
          <h2 class="relative z-[2] mb-2 text-sm font-semibold">Translation</h2>
          <article class="relative z-[1] min-h-0 flex-1 rounded-2xl pb-4">
            <WiggleBorder motionEnabled={motionEnabled()} />
            <div class="pointer-events-none absolute inset-0 z-0 rounded-2xl bg-white/85" />
            <p ref={translationPanel} class="relative z-[2] h-full overflow-y-auto px-8 py-5 text-sm leading-relaxed"><Show when={translation() || translationPartial()} fallback={<span class="text-neutral-500">Your translation will appear here</span>}><span>{translation()}</span><Show when={translationPartial()}><span>{translation() && ' '}</span><span class="text-neutral-500">{translationPartial()}</span></Show></Show></p>
          </article>
        </section>
      </section>
      <aside class="flex w-full max-w-sm flex-col gap-6 md:w-84 md:shrink-0">
        <section class="grid grid-cols-[1fr_auto_1fr] items-center gap-3" aria-label="Language switcher">
          <label class="relative flex min-h-12 items-center rounded-full"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-white/70" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-sm text-black outline-none" value={sourceLanguage()} onChange={(event) => updateSourceLanguage(event.currentTarget.value)}><For each={languages}>{(language) => <option value={language.code}>{language.label}</option>}</For></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label>
          <Icon name="arrow" class="size-5 shrink-0" />
          <label class="relative flex min-h-12 items-center rounded-full"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-white/70" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-sm text-black outline-none" value={targetLanguage()} onChange={(event) => setTargetLanguage(event.currentTarget.value)}><For each={languages}>{(language) => <option value={language.code}>{language.label}</option>}</For></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label>
        </section>
        <section class="flex flex-col gap-4">
          <h2 class="text-sm font-semibold">Phát phiên dịch</h2>
          <label class="relative flex min-h-12 items-center rounded-full text-left text-sm"><WiggleBorder type="pill" motionEnabled={motionEnabled()} /><div class="pointer-events-none absolute inset-0 z-0 rounded-full bg-white/70" /><select class="absolute inset-0 z-[3] h-full w-full cursor-pointer appearance-none bg-transparent px-5 pr-10 text-black outline-none" value={voice()} onChange={(event) => updateVoice(event.currentTarget.value)}><option value="">Giọng mặc định</option><For each={voices()}>{(item) => <option value={item[0]}>{item[0]}</option>}</For></select><Icon name="chevron" class="pointer-events-none absolute right-3 z-[2] size-5" /></label>
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
