import { CONFIG } from './config';
import type { ClassifierDebug, Label, Letter, Signals } from './types';

export interface DebugSample extends ClassifierDebug {
  tMs: number;
  letter: Letter;
  signals: Signals;
  label: Label;
  confidence: number;
}

export interface Range { min: number; max: number }

const WATCHED = ['eyeClosure', 'lookDown', 'pitchDelta'] as const;
export type WatchedRanges = Record<(typeof WATCHED)[number], Range>;

function rangeOf(values: number[]): Range {
  return { min: Math.min(...values), max: Math.max(...values) };
}

/** A rolling record of what each person's signals did, for tuning without a live screen. */
export class DebugBuffer {
  private items: DebugSample[] = [];

  add(sample: DebugSample): void {
    this.items.push(sample);
    const firstKept = this.items.findIndex((s) => sample.tMs - s.tMs <= CONFIG.debugBufferMs);
    if (firstKept > 0) this.items = this.items.slice(firstKept);
  }

  samples(): readonly DebugSample[] {
    return this.items;
  }

  recentRanges(letter: Letter, nowMs: number): WatchedRanges | null {
    const recent = this.items.filter(
      (s) => s.letter === letter && nowMs - s.tMs <= CONFIG.debugRangeMs,
    );
    if (recent.length === 0) return null;
    const rangeFor = (name: (typeof WATCHED)[number]): Range =>
      rangeOf(recent.map((s) => s.signals[name]));
    return {
      eyeClosure: rangeFor('eyeClosure'),
      lookDown: rangeFor('lookDown'),
      pitchDelta: rangeFor('pitchDelta'),
    };
  }
}

function drowsyText(debug: ClassifierDebug): string {
  if (debug.strongSustained) return 'strong expression blocking';
  const sign = debug.drowsySign ? `sign ${debug.drowsySign}` : 'no sign';
  return debug.strongHeld ? `${sign}, strong expression delays sleeping` : sign;
}

export function debugLines(
  letter: Letter,
  signals: Signals,
  ranges: WatchedRanges | null,
  debug: ClassifierDebug,
): string[] {
  const live = Object.entries(signals).map(([name, value]) => `${name} ${value.toFixed(2)}`);
  const lines = [`${letter}: ${live.join('  ')}`];
  if (ranges) {
    const parts = WATCHED.map((name) => {
      const { min, max } = ranges[name];
      return `${name} ${min.toFixed(2)} to ${max.toFixed(2)}`;
    });
    lines.push(`${letter} last ${CONFIG.debugRangeMs / 1000}s: ${parts.join('  ')}`);
  }
  lines.push(`${letter} drowsy: ${drowsyText(debug)}`);
  return lines;
}

export function debugLogJson(samples: readonly DebugSample[], savedAt: Date): string {
  return JSON.stringify({ savedAt: savedAt.toISOString(), config: CONFIG, samples }, null, 1);
}
