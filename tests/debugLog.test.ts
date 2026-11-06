import { describe, expect, it } from 'vitest';
import { CONFIG } from '../src/core/config';
import { DebugBuffer, debugLines, debugLogJson } from '../src/core/debugLog';
import type { DebugSample } from '../src/core/debugLog';
import type { ClassifierDebug, Letter, Signals } from '../src/core/types';

const SIGNALS: Signals = {
  eyeClosure: 0.1,
  lookDown: 0.2,
  pitchDelta: 3,
  smile: 0,
  jawOpen: 0,
  browDown: 0,
  browInnerUp: 0,
  frown: 0,
};
const QUIET: ClassifierDebug = { drowsySign: null, strongHeld: false, strongSustained: false };

function sample(tMs: number, letter: Letter = 'A', changes: Partial<Signals> = {}): DebugSample {
  return {
    tMs,
    letter,
    signals: { ...SIGNALS, ...changes },
    label: 'neutral',
    confidence: 90,
    ...QUIET,
  };
}

describe('DebugBuffer', () => {
  it('keeps about the last 30 seconds', () => {
    const buffer = new DebugBuffer();
    for (let t = 0; t <= 40000; t += 1000) buffer.add(sample(t));

    const times = buffer.samples().map((s) => s.tMs);
    expect(times[0]).toBe(10000);
    expect(times.at(-1)).toBe(40000);
  });

  it('keeps every person in the window', () => {
    const buffer = new DebugBuffer();
    buffer.add(sample(0, 'A'));
    buffer.add(sample(0, 'B'));

    expect(buffer.samples().map((s) => s.letter)).toEqual(['A', 'B']);
  });

  it('gives min and max over the last 10 seconds for one person', () => {
    const buffer = new DebugBuffer();
    buffer.add(sample(0, 'A', { eyeClosure: 0.99 }));
    buffer.add(sample(5000, 'A', { eyeClosure: 0.8, lookDown: 0.1, pitchDelta: -4 }));
    buffer.add(sample(8000, 'A', { eyeClosure: 0.3, lookDown: 0.6, pitchDelta: 12 }));
    buffer.add(sample(8000, 'B', { eyeClosure: 0 }));

    expect(buffer.recentRanges('A', 12000)).toEqual({
      eyeClosure: { min: 0.3, max: 0.8 },
      lookDown: { min: 0.1, max: 0.6 },
      pitchDelta: { min: -4, max: 12 },
    });
  });

  it('has no ranges for a person with no recent samples', () => {
    const buffer = new DebugBuffer();
    buffer.add(sample(0, 'A'));

    expect(buffer.recentRanges('B', 100)).toBeNull();
    expect(buffer.recentRanges('A', 20000)).toBeNull();
  });
});

describe('debugLines', () => {
  const ranges = {
    eyeClosure: { min: 0, max: 0.95 },
    lookDown: { min: 0.1, max: 0.5 },
    pitchDelta: { min: -2, max: 18.5 },
  };

  it('shows live signals, the 10 second range and the drowsy state', () => {
    const lines = debugLines('A', SIGNALS, ranges, QUIET);

    expect(lines[0]).toContain('A: eyeClosure 0.10  lookDown 0.20  pitchDelta 3.00');
    expect(lines[1]).toBe(
      'A last 10s: eyeClosure 0.00 to 0.95  lookDown 0.10 to 0.50  pitchDelta -2.00 to 18.50',
    );
    expect(lines[2]).toBe('A drowsy: no sign');
  });

  it.each<[ClassifierDebug, string]>([
    [{ ...QUIET, drowsySign: 'droopy' }, 'A drowsy: sign droopy'],
    [{ ...QUIET, drowsySign: 'headDroop' }, 'A drowsy: sign headDroop'],
    [{ ...QUIET, strongHeld: true }, 'A drowsy: no sign, strong expression delays sleeping'],
    [{ ...QUIET, strongHeld: true, strongSustained: true }, 'A drowsy: strong expression blocking'],
  ])('names what is driving drowsy %#', (debug, line) => {
    expect(debugLines('A', SIGNALS, ranges, debug)[2]).toBe(line);
  });

  it('leaves out the range line when there are no ranges', () => {
    expect(debugLines('A', SIGNALS, null, QUIET)).toHaveLength(2);
  });
});

describe('debugLogJson', () => {
  it('holds the samples and the settings they were judged against', () => {
    const parsed = JSON.parse(debugLogJson([sample(5)], new Date('2026-11-06T12:00:00Z')));

    expect(parsed.savedAt).toBe('2026-11-06T12:00:00.000Z');
    expect(parsed.config.eyesClosedAt).toBe(CONFIG.eyesClosedAt);
    expect(parsed.samples).toEqual([sample(5)]);
  });
});
