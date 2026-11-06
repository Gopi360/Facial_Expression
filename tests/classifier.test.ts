import { describe, expect, it } from 'vitest';
import { PersonClassifier } from '../src/core/classifier';
import type { Label, Reading, Signals } from '../src/core/types';

const STEP_MS = 66;

const OPEN: Signals = {
  eyeClosure: 0,
  lookDown: 0,
  pitchDelta: 0,
  smile: 0,
  jawOpen: 0,
  browDown: 0,
  browInnerUp: 0,
  frown: 0,
};
const CLOSED: Signals = { ...OPEN, eyeClosure: 1 };
const LAUGH: Signals = { ...OPEN, smile: 0.8, jawOpen: 0.5 };
const ANGRY: Signals = { ...OPEN, browDown: 0.7 };
const CRYING: Signals = { ...OPEN, browInnerUp: 0.6, frown: 0.5 };

/** Steps 66ms from fromMs and always finishes on an update at exactly toMs. */
function run(
  classifier: PersonClassifier,
  signals: Signals,
  fromMs: number,
  toMs: number,
  delaySec = 3,
): Reading {
  for (let t = fromMs; t < toMs; t += STEP_MS) classifier.update(signals, t, delaySec);
  return classifier.update(signals, toMs, delaySec);
}

/** Every label shown from fromMs to toMs, for "never shows" checks. */
function labelsShown(
  classifier: PersonClassifier,
  signalsAt: (t: number) => Signals,
  fromMs: number,
  toMs: number,
  delaySec = 3,
): Set<Label> {
  const labels = new Set<Label>();
  for (let t = fromMs; t <= toMs; t += STEP_MS) {
    labels.add(classifier.update(signalsAt(t), t, delaySec).label);
  }
  return labels;
}

describe('sleeping', () => {
  it('eyes closed 3 seconds gives sleeping', () => {
    const classifier = new PersonClassifier();

    expect(run(classifier, CLOSED, 0, 2500).label).not.toBe('sleeping');
    const reading = run(classifier, CLOSED, 2500, 3000);
    expect(reading).toEqual({ label: 'sleeping', confidence: 100 });
  });

  it('delay setting of 1 gives sleeping after 1 second', () => {
    const classifier = new PersonClassifier();

    expect(run(classifier, CLOSED, 0, 900, 1).label).not.toBe('sleeping');
    expect(run(classifier, CLOSED, 900, 1000, 1).label).toBe('sleeping');
  });

  it('a brief open-eye glitch under 200ms does not reset the count', () => {
    const classifier = new PersonClassifier();
    run(classifier, CLOSED, 0, 2000);
    run(classifier, OPEN, 2066, 2200);

    expect(run(classifier, CLOSED, 2266, 3000).label).toBe('sleeping');
  });

  it('opening eyes for longer resets the count', () => {
    const classifier = new PersonClassifier();
    run(classifier, CLOSED, 0, 2000);
    run(classifier, OPEN, 2066, 2400);

    expect(run(classifier, CLOSED, 2466, 4000).label).not.toBe('sleeping');
    expect(run(classifier, CLOSED, 4000, 5466).label).toBe('sleeping');
  });

  it('smile with closed eyes stays laughing at 4 seconds and becomes sleeping at 5', () => {
    const classifier = new PersonClassifier();
    const squeezed = { ...LAUGH, eyeClosure: 1 };

    expect(run(classifier, squeezed, 0, 4000).label).toBe('laughing');
    expect(run(classifier, squeezed, 4000, 4900).label).toBe('laughing');
    expect(run(classifier, squeezed, 4900, 5000).label).toBe('sleeping');
  });

  it('jaw open with closed eyes still becomes sleeping at 3 seconds', () => {
    const classifier = new PersonClassifier();

    expect(run(classifier, { ...CLOSED, jawOpen: 0.6 }, 0, 3000).label).toBe('sleeping');
  });

  it('angry with closed eyes becomes sleeping at 5 seconds', () => {
    const classifier = new PersonClassifier();
    const angryClosed = { ...ANGRY, eyeClosure: 1 };

    expect(run(classifier, angryClosed, 0, 4000).label).toBe('angry');
    expect(run(classifier, angryClosed, 4000, 5000).label).toBe('sleeping');
  });

  it('a strong expression waits for the longer of the delay and the 5 second limit', () => {
    const classifier = new PersonClassifier();
    const squeezed = { ...LAUGH, eyeClosure: 1 };

    expect(run(classifier, squeezed, 0, 6000, 7).label).toBe('laughing');
    expect(run(classifier, squeezed, 6000, 7000, 7).label).toBe('sleeping');
  });

  it('closed eyes at delay 7 show neither sleeping nor drowsy until 7 seconds', () => {
    const classifier = new PersonClassifier();
    const labels = labelsShown(classifier, () => CLOSED, 0, 6900, 7);

    expect(labels.has('sleeping')).toBe(false);
    expect(labels.has('drowsy')).toBe(false);
    expect(run(classifier, CLOSED, 6900, 7000, 7).label).toBe('sleeping');
  });

  it('waking from sleeping does not pass through drowsy', () => {
    const classifier = new PersonClassifier();
    run(classifier, CLOSED, 0, 3500);
    const labels = labelsShown(classifier, () => OPEN, 3566, 7000);

    expect(labels.has('drowsy')).toBe(false);
    expect(run(classifier, OPEN, 7000, 7066).label).toBe('neutral');
  });

  it('a one-frame smile dip with squeezed eyes stays out of sleeping before 5 seconds', () => {
    const squeezed = { ...LAUGH, eyeClosure: 1 };
    const dip = { ...squeezed, smile: 0.45 };
    const classifier = new PersonClassifier();
    const labels = labelsShown(classifier, (t) => (t === 3300 ? dip : squeezed), 0, 4950);

    expect(labels.has('sleeping')).toBe(false);
    expect(labels.has('drowsy')).toBe(false);
    expect(run(classifier, squeezed, 4950, 5000).label).toBe('sleeping');
  });

  it('a smile gone for 500ms no longer holds sleeping back', () => {
    const classifier = new PersonClassifier();
    run(classifier, { ...LAUGH, eyeClosure: 1 }, 0, 3300);

    expect(run(classifier, CLOSED, 3366, 3700).label).not.toBe('sleeping');
    expect(run(classifier, CLOSED, 3700, 3800).label).toBe('sleeping');
  });

  it('a 5 second frame gap does not count as closed time', () => {
    const classifier = new PersonClassifier();
    run(classifier, CLOSED, 0, 2000);

    expect(classifier.update(CLOSED, 7000, 3).label).not.toBe('sleeping');
  });

  it('a face lost for 1.5 seconds keeps its closed count', () => {
    const classifier = new PersonClassifier();
    run(classifier, CLOSED, 0, 2000);

    expect(classifier.update(CLOSED, 3500, 3).label).toBe('sleeping');
  });
});

describe('drowsy', () => {
  function slowBlink(classifier: PersonClassifier, fromMs: number): Reading {
    run(classifier, CLOSED, fromMs, fromMs + 500);
    return run(classifier, OPEN, fromMs + 566, fromMs + 900);
  }

  it('normal blinks for one minute never give sleeping or drowsy', () => {
    const labels = labelsShown(
      new PersonClassifier(),
      (t) => (t % 4000 < 200 ? CLOSED : OPEN),
      0,
      60000,
    );

    expect(labels.has('sleeping')).toBe(false);
    expect(labels.has('drowsy')).toBe(false);
  });

  it('two slow blinks within 10 seconds give drowsy', () => {
    const classifier = new PersonClassifier();

    expect(slowBlink(classifier, 0).label).not.toBe('drowsy');
    run(classifier, OPEN, 900, 5000);
    expect(slowBlink(classifier, 5000).label).toBe('drowsy');
  });

  it('two slow blinks further apart than 10 seconds do not give drowsy', () => {
    const classifier = new PersonClassifier();
    slowBlink(classifier, 0);
    run(classifier, OPEN, 900, 11000);

    expect(slowBlink(classifier, 11000).label).not.toBe('drowsy');
  });

  it('a sleeping-length closure does not count as a slow blink', () => {
    const classifier = new PersonClassifier();
    run(classifier, CLOSED, 0, 3500);
    run(classifier, OPEN, 3566, 6500);

    expect(slowBlink(classifier, 6500).label).not.toBe('drowsy');
  });

  it('double blinks every 5 seconds for one minute never give drowsy', () => {
    const doubleBlink = (t: number): Signals => {
      const phase = t % 5000;
      return phase < 180 || (phase >= 300 && phase < 480) ? CLOSED : OPEN;
    };
    const labels = labelsShown(new PersonClassifier(), doubleBlink, 0, 60000);

    expect(labels.has('drowsy')).toBe(false);
    expect(labels.has('sleeping')).toBe(false);
  });

  it('a glitch inside a slow blink keeps its longest unbroken run', () => {
    const classifier = new PersonClassifier();
    const slowWithGlitch = (fromMs: number): Reading => {
      run(classifier, CLOSED, fromMs, fromMs + 500);
      run(classifier, OPEN, fromMs + 566, fromMs + 632);
      run(classifier, CLOSED, fromMs + 698, fromMs + 800);
      return run(classifier, OPEN, fromMs + 866, fromMs + 1200);
    };

    expect(slowWithGlitch(0).label).not.toBe('drowsy');
    expect(slowWithGlitch(1266).label).toBe('drowsy');
  });

  it('drooping eyes for 3 seconds give drowsy', () => {
    const classifier = new PersonClassifier();
    const droopy = { ...OPEN, eyeClosure: 0.4 };

    expect(run(classifier, droopy, 0, 2900).label).not.toBe('drowsy');
    expect(run(classifier, droopy, 2900, 3000)).toEqual({ label: 'drowsy', confidence: 50 });
  });

  it('head droop with half-closed eyes for 1.5 seconds gives drowsy', () => {
    const classifier = new PersonClassifier();
    const nodding = { ...OPEN, eyeClosure: 0.5, pitchDelta: 20 };

    expect(run(classifier, nodding, 0, 1400).label).not.toBe('drowsy');
    expect(run(classifier, nodding, 1400, 1500)).toEqual({ label: 'drowsy', confidence: 50 });
  });

  it('head droop with closed eyes gives drowsy at 1.5 seconds, whatever the delay', () => {
    const classifier = new PersonClassifier();
    const nodded = { ...CLOSED, pitchDelta: 20 };

    expect(run(classifier, nodded, 0, 1400, 7).label).not.toBe('drowsy');
    expect(run(classifier, nodded, 1400, 1500, 7)).toEqual({ label: 'drowsy', confidence: 100 });
  });

  it('looking down with eyes open never triggers', () => {
    const lookingDown = { ...OPEN, pitchDelta: 25, lookDown: 0.7, eyeClosure: 0.5 };
    const labels = labelsShown(new PersonClassifier(), () => lookingDown, 0, 30000);

    expect(labels.has('sleeping')).toBe(false);
    expect(labels.has('drowsy')).toBe(false);
  });

  it('stays until its signs have been gone for 2 seconds', () => {
    const classifier = new PersonClassifier();
    run(classifier, { ...OPEN, eyeClosure: 0.4 }, 0, 3000);

    expect(run(classifier, OPEN, 3066, 4900).label).toBe('drowsy');
    expect(run(classifier, OPEN, 4900, 5600).label).toBe('neutral');
  });

  it('a strong expression suppresses drowsy', () => {
    const smilingDroopy = { ...OPEN, eyeClosure: 0.4, smile: 0.8 };
    const labels = labelsShown(new PersonClassifier(), () => smilingDroopy, 0, 10000);

    expect(labels.has('drowsy')).toBe(false);
  });

  it('a laugh with repeated eye squeezes leaves no drowsy behind', () => {
    const squeezing = (t: number): Signals =>
      t < 6000 ? { ...LAUGH, eyeClosure: t % 1000 < 600 ? 1 : 0 } : OPEN;
    const labels = labelsShown(new PersonClassifier(), squeezing, 0, 9000);

    expect(labels.has('drowsy')).toBe(false);
  });
});

describe('expressions', () => {
  const cases: [Label, Signals, number][] = [
    ['laughing', LAUGH, 80],
    ['smiling', { ...OPEN, smile: 0.4 }, 40],
    ['angry', ANGRY, 70],
    ['crying', CRYING, 55],
    ['neutral', { ...OPEN, smile: 0.1, browDown: 0.25, browInnerUp: 0.2 }, 75],
  ];

  it.each(cases)('%s is reached from its scores', (label, signals, confidence) => {
    expect(run(new PersonClassifier(), signals, 0, 600)).toEqual({ label, confidence });
  });

  it('a smile above the angry limit is not angry', () => {
    const reading = run(new PersonClassifier(), { ...ANGRY, smile: 0.25 }, 0, 600);

    expect(reading.label).toBe('neutral');
  });
});

describe('display', () => {
  it('a label flicker under 500ms is not displayed', () => {
    const smile = { ...OPEN, smile: 0.4 };
    const flicker = (t: number): Signals => (t >= 1000 && t < 1400 ? smile : OPEN);

    expect(labelsShown(new PersonClassifier(), flicker, 0, 3000)).toEqual(new Set(['neutral']));
  });

  it('a held label keeps its last confidence until the new label shows', () => {
    const classifier = new PersonClassifier();
    run(classifier, ANGRY, 0, 600);

    expect(run(classifier, OPEN, 666, 1000)).toEqual({ label: 'angry', confidence: 70 });
    expect(run(classifier, OPEN, 1000, 1200).label).toBe('neutral');
  });

  it('confidence stays an integer from 0 to 100', () => {
    const classifier = new PersonClassifier();
    const inputs: Signals[] = [
      OPEN,
      { ...OPEN, smile: 0.333 },
      { ...OPEN, smile: 1.2, jawOpen: 1.5 },
      { ...OPEN, browDown: 1.4 },
      { ...OPEN, browInnerUp: 1.3, frown: 1.3 },
      { ...OPEN, eyeClosure: 1.2 },
      { ...OPEN, eyeClosure: 0.4 },
    ];

    let t = 0;
    for (const signals of inputs) {
      for (const end = t + 6000; t < end; t += STEP_MS) {
        const { confidence } = classifier.update(signals, t, 3);
        expect(Number.isInteger(confidence)).toBe(true);
        expect(confidence).toBeGreaterThanOrEqual(0);
        expect(confidence).toBeLessThanOrEqual(100);
      }
    }
  });
});
