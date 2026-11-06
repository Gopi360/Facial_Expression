import { CONFIG } from './config';
import type { RawFace, Signals } from './types';

export function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

/** Head pitch in degrees from a column-major 4x4 matrix. Positive means tipped forward. */
export function pitchFromMatrix(m: ArrayLike<number>): number {
  const radians = Math.atan2(m[6] ?? 0, m[10] ?? 0);
  return (CONFIG.pitchSign * radians * 180) / Math.PI;
}

function pairAverage(scores: Record<string, number>, name: string): number {
  return ((scores[`${name}Left`] ?? 0) + (scores[`${name}Right`] ?? 0)) / 2;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const upper = sorted[mid] ?? 0;
  return sorted.length % 2 === 1 ? upper : ((sorted[mid - 1] ?? 0) + upper) / 2;
}

function percentile(values: readonly number[], fraction: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(fraction * (sorted.length - 1))] ?? 0;
}

interface Sample { ms: number; closure: number; pitch: number }

/** Learns one person's normal eye openness and head pitch, then judges changes against them. */
export class Baseline {
  private startMs: number | null = null;
  private readonly closures: number[] = [];
  private readonly pitches: number[] = [];
  private recent: Sample[] = []; // every sample in the last baselineMaxWaitMs
  private frozen: { closure: number; pitch: number } | null = null;

  toSignals(face: RawFace, nowMs: number): Signals {
    const rawClosure = pairAverage(face.scores, 'eyeBlink');
    if (!this.frozen) this.collect(rawClosure, face.pitchDeg, nowMs);

    const baseClosure = this.frozen?.closure ?? 0;
    return {
      eyeClosure: clamp01((rawClosure - baseClosure) / (1 - baseClosure)),
      lookDown: pairAverage(face.scores, 'eyeLookDown'),
      pitchDelta: this.frozen ? face.pitchDeg - this.frozen.pitch : 0,
      smile: pairAverage(face.scores, 'mouthSmile'),
      jawOpen: face.scores['jawOpen'] ?? 0,
      browDown: pairAverage(face.scores, 'browDown'),
      browInnerUp: face.scores['browInnerUp'] ?? 0,
      frown: pairAverage(face.scores, 'mouthFrown'),
    };
  }

  private collect(rawClosure: number, pitchDeg: number, nowMs: number): void {
    this.startMs ??= nowMs;
    if (rawClosure < CONFIG.baselineOpenEyesBelow) {
      this.closures.push(rawClosure);
      this.pitches.push(pitchDeg);
    }
    const enoughTime = nowMs - this.startMs >= CONFIG.baselineMs;
    if (enoughTime && this.closures.length >= CONFIG.baselineMinSamples) {
      this.frozen = { closure: median(this.closures), pitch: median(this.pitches) };
      return;
    }
    this.tryFallback(rawClosure, pitchDeg, nowMs);
  }

  /**
   * Some eyes never read as open, so after a long wait their low percentile is
   * taken as their normal. Eyes truly shut stay above the limit and keep waiting,
   * or a sleeper would become the new normal and read as awake forever.
   */
  private tryFallback(rawClosure: number, pitchDeg: number, nowMs: number): void {
    this.recent.push({ ms: nowMs, closure: rawClosure, pitch: pitchDeg });
    this.recent = this.recent.filter((s) => nowMs - s.ms <= CONFIG.baselineMaxWaitMs);
    if (this.startMs === null || nowMs - this.startMs < CONFIG.baselineMaxWaitMs) return;

    const closures = this.recent.map((s) => s.closure);
    const closure = percentile(closures, CONFIG.baselineFallbackPercentile);
    if (closure <= CONFIG.baselineFallbackMaxClosure) {
      this.frozen = { closure, pitch: median(this.recent.map((s) => s.pitch)) };
    }
  }
}
