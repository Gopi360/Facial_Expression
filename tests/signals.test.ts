import { describe, expect, it } from 'vitest';
import { Baseline, pitchFromMatrix } from '../src/core/signals';
import type { RawFace } from '../src/core/types';

const STEP_MS = 66;

function face(rawClosure: number, pitchDeg = 0): RawFace {
  return {
    box: { x: 0, y: 0, width: 0.2, height: 0.2 },
    scores: { eyeBlinkLeft: rawClosure, eyeBlinkRight: rawClosure },
    pitchDeg,
  };
}

function feed(baseline: Baseline, input: RawFace, fromMs: number, toMs: number): void {
  for (let t = fromMs; t < toMs; t += STEP_MS) baseline.toSignals(input, t);
}

function xRotation(degrees: number): number[] {
  const r = (degrees * Math.PI) / 180;
  const [c, s] = [Math.cos(r), Math.sin(r)];
  // Column-major 4x4.
  return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
}

describe('pitchFromMatrix', () => {
  it('identity matrix gives 0 pitch', () => {
    expect(pitchFromMatrix(xRotation(0))).toBeCloseTo(0);
  });

  it('20 degree X rotation matrix gives 20 pitch', () => {
    expect(pitchFromMatrix(xRotation(20))).toBeCloseTo(20);
  });
});

describe('Baseline', () => {
  it('averages left and right scores', () => {
    const signals = new Baseline().toSignals(
      {
        box: { x: 0, y: 0, width: 0.2, height: 0.2 },
        scores: {
          eyeBlinkLeft: 0.2,
          eyeBlinkRight: 0.4,
          eyeLookDownLeft: 0.1,
          eyeLookDownRight: 0.3,
          mouthSmileLeft: 0.5,
          mouthSmileRight: 0.7,
          browDownLeft: 0.6,
          browDownRight: 0.8,
          mouthFrownLeft: 0.0,
          mouthFrownRight: 0.2,
          jawOpen: 0.45,
          browInnerUp: 0.55,
        },
        pitchDeg: 0,
      },
      0,
    );

    expect(signals.eyeClosure).toBeCloseTo(0.3);
    expect(signals.lookDown).toBeCloseTo(0.2);
    expect(signals.smile).toBeCloseTo(0.6);
    expect(signals.browDown).toBeCloseTo(0.7);
    expect(signals.frown).toBeCloseTo(0.1);
    expect(signals.jawOpen).toBeCloseTo(0.45);
    expect(signals.browInnerUp).toBeCloseTo(0.55);
  });

  it('baseline raises closure scale for naturally narrow eyes', () => {
    const baseline = new Baseline();
    feed(baseline, face(0.3), 0, 2500);

    expect(baseline.toSignals(face(0.3), 2500).eyeClosure).toBeCloseTo(0);
    expect(baseline.toSignals(face(0.65), 2566).eyeClosure).toBeCloseTo(0.5);
    expect(baseline.toSignals(face(0.1), 2632).eyeClosure).toBe(0);
  });

  it('arriving with closed eyes does not freeze a bad baseline', () => {
    const baseline = new Baseline();
    feed(baseline, face(0.9), 0, 3000);

    expect(baseline.toSignals(face(0.9), 3000).eyeClosure).toBeCloseTo(0.9);
  });

  it('waits for enough open-eye samples after the minimum time', () => {
    const baseline = new Baseline();
    feed(baseline, face(0.9), 0, 3000);
    feed(baseline, face(0.2), 3000, 3000 + 8 * STEP_MS);

    expect(baseline.toSignals(face(0.2), 3600).eyeClosure).toBeCloseTo(0.2); // 9th open sample
    expect(baseline.toSignals(face(0.2), 3666).eyeClosure).toBeCloseTo(0); // 10th freezes
    expect(baseline.toSignals(face(0.6), 3732).eyeClosure).toBeCloseTo(0.5);
  });

  it("pitchDelta is measured from the person's normal pitch", () => {
    const baseline = new Baseline();

    expect(baseline.toSignals(face(0.1, 10), 0).pitchDelta).toBe(0);
    feed(baseline, face(0.1, 10), 66, 2500);
    expect(baseline.toSignals(face(0.1, 28), 2500).pitchDelta).toBeCloseTo(18);
  });
});
