import { CONFIG } from './config';
import { clamp01 } from './signals';
import type { Label, Reading, Signals } from './types';

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
  openSinceMs: number | null; // first open frame while the count still runs
  slowBlinkEndsMs: number[];
  droopySinceMs: number | null;
  headDroopSinceMs: number | null;
  drowsySignMs: number | null; // last time any drowsy sign was true
  shown: Reading | null;
  pendingLabel: Label | null;
  pendingSinceMs: number;
}

function freshState(): State {
  return {
    lastUpdateMs: null,
    closedSinceMs: null,
    lastClosedMs: 0,
    openSinceMs: null,
    slowBlinkEndsMs: [],
    droopySinceMs: null,
    headDroopSinceMs: null,
    drowsySignMs: null,
    shown: null,
    pendingLabel: null,
    pendingSinceMs: 0,
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
    const strong = isStrongExpression(s);
    this.trackClosure(s, nowMs, delayMs);
    this.trackDrowsySigns(s, nowMs, strong);
    return this.display(s, this.candidate(s, nowMs, delayMs, strong), nowMs);
  }

  private trackClosure(s: Signals, nowMs: number, delayMs: number): void {
    const st = this.state;
    if (isEyesClosed(s)) {
      st.closedSinceMs ??= nowMs;
      st.lastClosedMs = nowMs;
      st.openSinceMs = null;
      return;
    }
    if (st.closedSinceMs === null) return;

    st.openSinceMs ??= nowMs;
    if (nowMs - st.openSinceMs < CONFIG.openGraceMs) return;

    // The open grace period is not part of the closure's length.
    const lengthMs = st.lastClosedMs - st.closedSinceMs;
    if (lengthMs >= CONFIG.slowBlinkMs && lengthMs < delayMs) st.slowBlinkEndsMs.push(nowMs);
    st.closedSinceMs = null;
    st.openSinceMs = null;
  }

  private trackDrowsySigns(s: Signals, nowMs: number, strong: boolean): void {
    const st = this.state;
    if (strong) {
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
    const notLookingDown = s.lookDown < CONFIG.lookDownAt;
    const droopy = notLookingDown && s.eyeClosure >= CONFIG.droopyEyesAt;
    const headDroop =
      notLookingDown &&
      s.pitchDelta >= CONFIG.headDroopDeg &&
      s.eyeClosure >= CONFIG.headDroopEyesAt;
    st.droopySinceMs = droopy ? (st.droopySinceMs ?? nowMs) : null;
    st.headDroopSinceMs = headDroop ? (st.headDroopSinceMs ?? nowMs) : null;

    const signPresent =
      st.slowBlinkEndsMs.length >= CONFIG.slowBlinksForDrowsy ||
      (st.droopySinceMs !== null && nowMs - st.droopySinceMs >= CONFIG.droopyHoldMs) ||
      (st.headDroopSinceMs !== null && nowMs - st.headDroopSinceMs >= CONFIG.headDroopHoldMs);
    if (signPresent) st.drowsySignMs = nowMs;
  }

  private candidate(s: Signals, nowMs: number, delayMs: number, strong: boolean): Reading {
    const { closedSinceMs, drowsySignMs } = this.state;
    const neededMs = strong ? Math.max(delayMs, CONFIG.expressionLimitMs) : delayMs;
    if (closedSinceMs !== null && nowMs - closedSinceMs >= neededMs) {
      return { label: 'sleeping', confidence: percent(s.eyeClosure) };
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
      st.pendingLabel = null;
      st.shown = candidate;
      return candidate;
    }
    if (candidate.label !== st.pendingLabel) {
      st.pendingLabel = candidate.label;
      st.pendingSinceMs = nowMs;
    }
    const immediate = candidate.label === 'sleeping' || candidate.label === 'drowsy';
    if (immediate || nowMs - st.pendingSinceMs >= CONFIG.labelHoldMs) {
      st.pendingLabel = null;
      st.shown = candidate;
    }
    return st.shown;
  }
}
