import { CONFIG } from '../core/config';

export class Beeper {
  private readonly ctx = new AudioContext();
  private readonly gain: GainNode;
  private osc: OscillatorNode | null = null;
  private timer: number | null = null;
  private sounding = false;
  private volume = 1;

  constructor() {
    this.gain = this.ctx.createGain();
    this.gain.gain.value = 0;
    this.gain.connect(this.ctx.destination);
    void this.ctx.resume().catch(() => {});
  }

  setVolume(v: number): void {
    this.volume = Math.min(1, Math.max(0, v));
    if (this.osc && this.timer !== null && this.gain.gain.value > 0) {
      this.gain.gain.value = this.volume * CONFIG.beepPeakGain;
    }
  }

  setSounding(on: boolean): void {
    if (on === this.sounding) return;
    this.sounding = on;
    if (on) this.start();
    else this.stop();
  }

  private start(): void {
    try {
      void this.ctx.resume().catch(() => {});
      const osc = this.ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = CONFIG.beepHz;
      osc.connect(this.gain);
      osc.start();
      this.osc = osc;
      this.beepOn();
    } catch (err) {
      console.error('Alarm sound failed', err);
      this.sounding = false;
    }
  }

  private beepOn(): void {
    this.gain.gain.value = this.volume * CONFIG.beepPeakGain;
    this.timer = window.setTimeout(() => this.beepOff(), CONFIG.beepOnMs);
  }

  private beepOff(): void {
    this.gain.gain.value = 0;
    this.timer = window.setTimeout(() => this.beepOn(), CONFIG.beepOffMs);
  }

  private stop(): void {
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    this.gain.gain.value = 0;
    if (this.osc) {
      try { this.osc.stop(); } catch { /* already stopped */ }
      this.osc.disconnect();
      this.osc = null;
    }
  }
}
