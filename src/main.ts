import './styles.css';
import { Beeper } from './browser/beeper';
import { startCamera } from './browser/camera';
import { Detector } from './browser/detector';
import { appendLog, diffEvents } from './core/activityLog';
import { shouldSound } from './core/alarmPolicy';
import { PersonClassifier } from './core/classifier';
import { CONFIG } from './core/config';
import { cameraErrorMessage, modelErrorMessage } from './core/errors';
import { Baseline } from './core/signals';
import { Tracker } from './core/tracker';
import type { Label, Letter, LogEntry, Person, RawFace, Signals } from './core/types';
import { saveSnapshot } from './ui/snapshot';
import { drawOverlay, renderLog, renderStrip, showMessage } from './ui/view';

const LOADING_MESSAGE = 'Loading...';
const SOUND_ERROR_MESSAGE = 'The alarm sound could not be started. Reload the page to try again.';
const DETECT_ERROR_MESSAGE = 'Face detection stopped working. Reload the page to try again.';
const SNAPSHOT_ERROR_MESSAGE = 'The snapshot could not be saved. Please try again.';

/** What is remembered for one letter. A face in its grace keeps its last person. */
interface FaceState {
  baseline: Baseline;
  classifier: PersonClassifier;
  person: Person;
  lastSeenMs: number;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}

const video = byId<HTMLVideoElement>('video');
const overlay = byId<HTMLCanvasElement>('overlay');
const startButton = byId<HTMLButtonElement>('start');
const message = byId('message');
const strip = byId('strip');
const debug = byId('debug');
const logList = byId('log');
const muteButton = byId<HTMLButtonElement>('mute');
const volumeSlider = byId<HTMLInputElement>('volume');
const volumeValue = byId<HTMLOutputElement>('volume-value');
const delaySlider = byId<HTMLInputElement>('delay');
const delayValue = byId<HTMLOutputElement>('delay-value');
const snapshotButton = byId<HTMLButtonElement>('snapshot');

const debugOn = new URLSearchParams(window.location.search).get('debug') === '1';
debug.hidden = !debugOn;

const tracker = new Tracker();
const faceStates = new Map<Letter, FaceState>();
let labels = new Map<Letter, Label>();
let log: LogEntry[] = [];

let beeper: Beeper | null = null;
let cameraRunning = false;
let detectFailed = false;
let muted = false;
let delaySec: number = CONFIG.defaultDelaySec;

function showVolume(): void {
  volumeValue.textContent = `${Math.round(volumeSlider.valueAsNumber * 100)}%`;
}

function showDelay(): void {
  delayValue.textContent = `${delaySec}s`;
}

function detectFaces(detector: Detector, nowMs: number): RawFace[] {
  try {
    const faces = detector.detect(video, nowMs);
    if (detectFailed) {
      detectFailed = false;
      showMessage(message, null);
    }
    return faces;
  } catch (err) {
    // Carry on with no faces, so people leave after their grace and the alarm cannot stick on.
    if (!detectFailed) console.error('Face detection failed', err);
    detectFailed = true;
    showMessage(message, DETECT_ERROR_MESSAGE);
    return [];
  }
}

function matchOverlayToVideo(): void {
  if (overlay.width !== video.videoWidth) overlay.width = video.videoWidth;
  if (overlay.height !== video.videoHeight) overlay.height = video.videoHeight;
}

function debugLine(letter: Letter, signals: Signals): string {
  const values = Object.entries(signals).map(([name, value]) => `${name} ${value.toFixed(2)}`);
  return `${letter}: ${values.join('  ')}`;
}

function processFrame(detector: Detector, sound: Beeper, nowMs: number): void {
  const tracked = tracker.update(detectFaces(detector, nowMs), nowMs);

  const visible: Person[] = [];
  const debugLines: string[] = [];
  for (const { letter, face } of tracked) {
    // A letter released and retaken in one update never leaves activeLetters,
    // for example after the tab was hidden, so old state is caught here.
    let state = faceStates.get(letter);
    if (state && nowMs - state.lastSeenMs > CONFIG.letterReleaseMs) state = undefined;
    const baseline = state?.baseline ?? new Baseline();
    const classifier = state?.classifier ?? new PersonClassifier();

    const signals = baseline.toSignals(face, nowMs);
    const reading = classifier.update(signals, nowMs, delaySec);
    const person: Person = { letter, box: face.box, reading };
    faceStates.set(letter, { baseline, classifier, person, lastSeenMs: nowMs });
    visible.push(person);
    if (debugOn) debugLines.push(debugLine(letter, signals));
  }

  const active = new Set(tracker.activeLetters());
  for (const letter of faceStates.keys()) {
    if (!active.has(letter)) faceStates.delete(letter);
  }

  const people = [...faceStates.values()]
    .map((state) => state.person)
    .sort((a, b) => a.letter.localeCompare(b.letter));
  const nextLabels = new Map(people.map((p) => [p.letter, p.reading.label]));
  const events = diffEvents(labels, nextLabels, new Date());
  labels = nextLabels;

  matchOverlayToVideo();
  drawOverlay(overlay, visible);
  renderStrip(strip, people);
  if (events.length > 0) {
    log = appendLog(log, events);
    renderLog(logList, log);
  }
  if (debugOn) debug.textContent = debugLines.join('\n');

  sound.setSounding(shouldSound([...nextLabels.values()], muted));
}

function runLoop(detector: Detector, sound: Beeper): void {
  let lastFrameMs = -Infinity;
  const tick = (nowMs: number): void => {
    if (!cameraRunning) return;
    requestAnimationFrame(tick);
    if (nowMs - lastFrameMs < CONFIG.minFrameGapMs) return;
    lastFrameMs = nowMs;
    processFrame(detector, sound, nowMs);
  };
  requestAnimationFrame(tick);
}

function onCameraEnded(sound: Beeper, err: unknown): void {
  cameraRunning = false;
  sound.setSounding(false);
  snapshotButton.disabled = true;
  drawOverlay(overlay, []);
  renderStrip(strip, []);
  showMessage(message, cameraErrorMessage(err));
}

async function start(): Promise<void> {
  startButton.hidden = true;

  let sound: Beeper;
  try {
    sound = new Beeper();
    sound.setVolume(volumeSlider.valueAsNumber);
    beeper = sound;
  } catch (err) {
    console.error('Alarm sound failed', err);
    showMessage(message, SOUND_ERROR_MESSAGE);
    return;
  }

  showMessage(message, LOADING_MESSAGE);

  let detector: Detector;
  try {
    detector = await Detector.create();
  } catch (err) {
    console.error('Face model failed to load', err);
    showMessage(message, modelErrorMessage(err));
    return;
  }

  cameraRunning = true;
  try {
    await startCamera(video, () => onCameraEnded(sound, new Error('Camera ended')));
  } catch (err) {
    console.error('Camera failed to start', err);
    cameraRunning = false;
    showMessage(message, cameraErrorMessage(err));
    return;
  }
  if (!cameraRunning) return; // ended while starting; its message is already showing

  showMessage(message, null);
  snapshotButton.disabled = false;
  runLoop(detector, sound);
}

startButton.addEventListener('click', () => void start());

muteButton.addEventListener('click', () => {
  muted = !muted;
  muteButton.textContent = muted ? 'Unmute' : 'Mute';
  muteButton.setAttribute('aria-pressed', String(muted));
});

volumeSlider.addEventListener('input', () => {
  beeper?.setVolume(volumeSlider.valueAsNumber);
  showVolume();
});

delaySlider.min = String(CONFIG.minDelaySec);
delaySlider.max = String(CONFIG.maxDelaySec);
delaySlider.value = String(CONFIG.defaultDelaySec);
delaySlider.addEventListener('input', () => {
  delaySec = delaySlider.valueAsNumber;
  showDelay();
});

snapshotButton.addEventListener('click', () => {
  try {
    saveSnapshot(video, overlay);
  } catch (err) {
    console.error('Snapshot failed', err);
    showMessage(message, SNAPSHOT_ERROR_MESSAGE);
  }
});

showVolume();
showDelay();
