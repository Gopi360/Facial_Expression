import { CONFIG } from './config';
import { clamp01 } from './signals';
import type { Reading, Signals } from './types';

const MS_PER_SEC = 1000;

function percent(strength: number): number {
  return Math.round(clamp01(strength) * 100);
}

function isEyesClosed(s: Signals): boolean {
  const threshold =
    s.lookDown >= CONFIG.lookDownAt ? CONFIG.eyesClosedWhenLookingDownAt : CONFIG.eyesClosedAt;
  return s.eyeClosure >= threshold;
}

function isAngry(s: Signals): boolean {
  return s.browDown >= CONFIG.angryBrowAt && s.smile < CONFIG.angrySmileMax;
}

function isCrying(s: Signals): boolean {
  return s.browInnerUp >= CONFIG.cryBrowAt && s.frown >= CONFIG.cryFrownAt;
}

function isStrongExpression(s: Signals): boolean {
  return s.smile >= CONFIG.strongSmileAt || isAngry(s) || isCrying(s);
}

function neutralReading(s: Signals): Reading {
  return { label: 'neutral', confidence: percent(1 - Math.max(s.smile, s.browDown, s.browInnerUp)) };
}

function expressionReading(s: Signals): Reading {
  if (s.smile >= CONFIG.strongSmileAt && s.jawOpen >= CONFIG.laughJawAt) {
    return { label: 'laughing', confidence: percent(s.smile) };
  }
  if (s.smile >= CONFIG.smileAt) return { label: 'smiling', confidence: percent(s.smile) };
  if (isAngry(s)) return { label: 'angry', confidence: percent(s.browDown) };
  if (isCrying(s)) {
    return { label: 'crying', confidence: percent((s.browInnerUp + s.frown) / 2) };
  }
  return neutralReading(s);
}

interface State {
  lastUpdateMs: number | null;
  closedSinceMs: number | null; // start of the closed count
  lastClosedMs: number; // last frame with eyes closed, while the count runs
  lastClosedStrength: number; // eyeClosure on that frame
  closedRunSinceMs: number | null; // start of the current unbroken closed run
  longestClosedRunMs: number; // longest unbroken run within the closed count
  openSinceMs: number | null; // first open frame while the count still runs
  slowBlinkEndsMs: number[];
  lastStrongMs: number | null;
  strongRunSinceMs: number | null; // start of the current unbroken strong run
  droopySinceMs: number | null;
  headDroopSinceMs: number | null;
  drowsySignMs: number | null; // last time any drowsy sign was true
  shown: Reading | null;
  differsSinceMs: number | null; // when the candidate first stopped matching the shown label
}

function freshState(): State {
  return {
    lastUpdateMs: null,
    closedSinceMs: null,
    lastClosedMs: 0,
    lastClosedStrength: 0,
    closedRunSinceMs: null,
    longestClosedRunMs: 0,
    openSinceMs: null,
    slowBlinkEndsMs: [],
    lastStrongMs: null,
    strongRunSinceMs: null,
    droopySinceMs: null,
    headDroopSinceMs: null,
    drowsySignMs: null,
    shown: null,
    differsSinceMs: null,
  };
}

/** Turns one person's signals over time into a steady label and confidence. */
export class PersonClassifier {
  private state = freshState();

  update(s: Signals, nowMs: number, delaySec: number): Reading {
    const { lastUpdateMs } = this.state;
    if (lastUpdateMs !== null && nowMs - lastUpdateMs > CONFIG.frameGapResetMs) {
      this.state = freshState();
    }
    this.state.lastUpdateMs = nowMs;

    const delayMs = delaySec * MS_PER_SEC;
    const strong = this.heldStrong(s, nowMs);
    const sustained = this.sustainedStrong(s, nowMs, strong);
    this.trackClosure(s, nowMs, delayMs);
    this.trackDrowsySigns(s, nowMs, sustained);
    return this.display(s, this.candidate(s, nowMs, delayMs, strong), nowMs);
  }

  /** A strong expression counts as present until it has been absent for labelHoldMs. */
  private heldStrong(s: Signals, nowMs: number): boolean {
    const st = this.state;
    if (isStrongExpression(s)) st.lastStrongMs = nowMs;
    return st.lastStrongMs !== null && nowMs - st.lastStrongMs < CONFIG.labelHoldMs;
  }

  /**
   * True once a strong expression has run unbroken for labelHoldMs, and until it
   * is no longer held. A lone noisy frame never gets there.
   */
  private sustainedStrong(s: Signals, nowMs: number, held: boolean): boolean {
    const st = this.state;
    const reached = (): boolean =>
      st.strongRunSinceMs !== null && nowMs - st.strongRunSinceMs >= CONFIG.labelHoldMs;
    if (isStrongExpression(s)) st.strongRunSinceMs ??= nowMs;
    else if (!held || !reached()) st.strongRunSinceMs = null;
    return reached();
  }

  private trackClosure(s: Signals, nowMs: number, delayMs: number): void {
    const st = this.state;
    if (isEyesClosed(s)) {
      st.closedSinceMs ??= nowMs;
      st.closedRunSinceMs ??= nowMs;
      st.lastClosedMs = nowMs;
      st.lastClosedStrength = s.eyeClosure;
      st.longestClosedRunMs = Math.max(st.longestClosedRunMs, nowMs - st.closedRunSinceMs);
      st.openSinceMs = null;
      return;
    }
    st.closedRunSinceMs = null;
    if (st.closedSinceMs === null) return;

    st.openSinceMs ??= nowMs;
    if (nowMs - st.openSinceMs < CONFIG.openGraceMs) return;

    // The grace bridges the sleep count, but two quick blinks it joins are not
    // one slow blink, so a slow blink needs one unbroken run of that length.
    const countedMs = st.lastClosedMs - st.closedSinceMs;
    if (st.longestClosedRunMs >= CONFIG.slowBlinkMs && countedMs < delayMs) {
      st.slowBlinkEndsMs.push(nowMs);
    }
    st.closedSinceMs = null;
    st.openSinceMs = null;
    st.longestClosedRunMs = 0;
  }

  private trackDrowsySigns(s: Signals, nowMs: number, sustainedStrong: boolean): void {
    const st = this.state;
    if (sustainedStrong) {
      // Clear everything, not just the label, so that eyes squeezed shut in a
      // laugh cannot set off drowsy the moment the laugh ends.
      st.slowBlinkEndsMs = [];
      st.droopySinceMs = null;
      st.headDroopSinceMs = null;
      st.drowsySignMs = null;
      return;
    }

    st.slowBlinkEndsMs = st.slowBlinkEndsMs.filter(
      (endMs) => nowMs - endMs <= CONFIG.slowBlinkWindowMs,
    );
    const closed = isEyesClosed(s);
    const notLookingDown = s.lookDown < CONFIG.lookDownAt;
    const droopy = notLookingDown && s.eyeClosure >= CONFIG.droopyEyesAt;
    const headDroop =
      notLookingDown &&
      s.pitchDelta >= CONFIG.headDroopDeg &&
      s.eyeClosure >= CONFIG.headDroopEyesAt;
    // A blink passes through the droopy hold, but a longer closure ends it.
    const longClosure =
      st.closedRunSinceMs !== null && nowMs - st.closedRunSinceMs >= CONFIG.slowBlinkMs;
    if (!droopy || longClosure) st.droopySinceMs = null;
    else if (!closed) st.droopySinceMs ??= nowMs;
    st.headDroopSinceMs = headDroop ? (st.headDroopSinceMs ?? nowMs) : null;

    // Fully closed eyes are not the droopy sign, so the delay setting governs them.
    // Only someone already drowsy stays drowsy through them, until sleeping takes over.
    const droopyHeld =
      !closed && st.droopySinceMs !== null && nowMs - st.droopySinceMs >= CONFIG.droopyHoldMs;
    const slowBlinks = st.slowBlinkEndsMs.length >= CONFIG.slowBlinksForDrowsy;
    const signPresent =
      slowBlinks ||
      droopyHeld ||
      (st.headDroopSinceMs !== null && nowMs - st.headDroopSinceMs >= CONFIG.headDroopHoldMs) ||
      (closed && st.shown?.label === 'drowsy');
    // A strong frame neither ends drowsy nor starts it, so a laugh that begins
    // just as a hold runs out does not sound the alarm before it is sustained.
    if (!signPresent || isStrongExpression(s)) return;
    st.drowsySignMs = nowMs;
    if (slowBlinks) st.slowBlinkEndsMs = []; // one event, so drowsy ends after drowsyClearMs
  }

  private candidate(s: Signals, nowMs: number, delayMs: number, strong: boolean): Reading {
    const { closedSinceMs, drowsySignMs, lastClosedStrength } = this.state;
    const neededMs = strong ? Math.max(delayMs, CONFIG.expressionLimitMs) : delayMs;
    if (closedSinceMs !== null && nowMs - closedSinceMs >= neededMs) {
      // Through the open grace this stays the strength of the last closed frame.
      return { label: 'sleeping', confidence: percent(lastClosedStrength) };
    }
    if (drowsySignMs !== null && nowMs - drowsySignMs < CONFIG.drowsyClearMs) {
      const strength = Math.max(s.eyeClosure, CONFIG.drowsyMinConfidence);
      return { label: 'drowsy', confidence: percent(strength) };
    }
    return expressionReading(s);
  }

  /** A held label keeps the confidence it was last shown with. */
  private display(s: Signals, candidate: Reading, nowMs: number): Reading {
    const st = this.state;
    st.shown ??= neutralReading(s);

    if (candidate.label === st.shown.label) {
      st.differsSinceMs = null;
      st.shown = candidate;
      return candidate;
    }
    // Timed against the shown label, not one candidate, so that two other labels
    // alternating faster than the hold cannot pin an old label, or the alarm, on.
    st.differsSinceMs ??= nowMs;
    const immediate = candidate.label === 'sleeping' || candidate.label === 'drowsy';
    if (immediate || nowMs - st.differsSinceMs >= CONFIG.labelHoldMs) {
      st.differsSinceMs = null;
      st.shown = candidate;
    }
    return st.shown;
  }
}
