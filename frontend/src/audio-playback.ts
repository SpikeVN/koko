export class AudioPlayback {
  private context: AudioContext | undefined;
  private nextTime = 0;
  private readonly sources = new Set<AudioBufferSourceNode>();

  stop() {
    this.sources.forEach((source) => source.stop());
    this.sources.clear();
    this.nextTime = 0;
  }

  resume(rate = 48_000) {
    // Create the context during a user gesture so browser autoplay rules permit output.
    this.context ??= new AudioContext({ sampleRate: rate });
    return this.context.state === 'suspended' ? this.context.resume() : Promise.resolve();
  }

  close() {
    return this.context?.close();
  }

  playFloat(samples: Float32Array, rate: number, enabled: boolean) {
    if (!enabled || !samples.length) return;
    this.context ??= new AudioContext({ sampleRate: rate });
    const buffer = this.context.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(Float32Array.from(samples), 0);
    this.schedule(buffer);
  }

  playPcm16(bytes: ArrayBuffer, rate: number, enabled: boolean) {
    if (!enabled) return;
    this.context ??= new AudioContext({ sampleRate: rate });
    const pcm = new Int16Array(bytes);
    const buffer = this.context.createBuffer(1, pcm.length, rate);
    const channel = buffer.getChannelData(0);
    for (let index = 0; index < pcm.length; index += 1) channel[index] = pcm[index] / 32768;
    this.schedule(buffer);
  }

  private schedule(buffer: AudioBuffer) {
    const source = this.context!.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context!.destination);
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
    this.nextTime = Math.max(this.nextTime, this.context!.currentTime);
    source.start(this.nextTime);
    this.nextTime += buffer.duration;
  }
}
