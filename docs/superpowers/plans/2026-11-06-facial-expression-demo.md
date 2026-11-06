# Live Facial Expression Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web page that reads the laptop camera and shows, per person, a live expression label with a confidence score, and sounds an alarm when anyone is sleeping or about to sleep.

**Architecture:** Everything runs in the browser. MediaPipe Face Landmarker gives each face a box, blendshape scores and a head matrix. Pure TypeScript modules in `src/core/` turn those into stable letters and labels and are unit tested. Thin browser modules in `src/browser/` and `src/ui/` handle the camera, sound and screen.

**Tech Stack:** TypeScript 7.0.2, Vite 8.3.6, Vitest 5.1.0, `@mediapipe/tasks-vision` 1.2.2. Node 24 is installed. No UI framework.

**Spec:** `docs/superpowers/specs/2026-11-06-facial-expression-demo-design.md`

## Global Constraints

- No text on the camera view: only a thin box and a small letter per face.
- Nothing leaves the laptop: no video, image or analytics network calls. No CDN at runtime.
- WASM and model are served from the project. Package versions are pinned exactly (no `^`).
- At most 4 faces (`numFaces: 4`). Detection capped at about 15 per second.
- Video is mirrored with CSS only. Frames passed to MediaPipe are never mirrored.
- Every camera and model call has error handling that ends in a readable on-page message, never a blank page.
- `src/core/` must not touch `window`, `document` or any browser API.
- All tunable numbers live in `src/core/config.ts`. No magic numbers elsewhere.
- Target: current Chrome on Windows, served from `localhost`.
- Git: never commit to `main`. All work is on `feat/expression-demo`, with conventional commit messages.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0175kJheuKPY7sg8wM26zVt6
  ```

## Review Focus

1. Tab hidden or laptop sleeps, then frames resume after a long gap: the gap must not count as closed eyes. Test in Task 3.
2. Person who is alarming leaves the view: once the letter is released, the sound stops. Test in Task 4.
3. Face vanishes briefly and returns within 2 seconds: keeps its letter and its sleep count. Tests in Tasks 2 and 3.
4. Camera unplugged or blocked mid-demo: a clear message appears and the alarm stops. Test in Task 5 (message mapping), manual check in Task 7.
5. Model or WASM files missing because setup was skipped: the page says to run `npm run setup:assets`. Test in Task 5.

---

## File Map

| File | Responsibility |
|---|---|
| `src/core/types.ts` | Shared types |
| `src/core/config.ts` | All thresholds and timings |
| `src/core/tracker.ts` | Stable letters across frames |
| `src/core/signals.ts` | Raw scores to signals, per-face baseline, head pitch |
| `src/core/classifier.ts` | Per-person label and confidence over time |
| `src/core/text.ts` | Strip and log wording |
| `src/core/activityLog.ts` | Change detection and log list |
| `src/core/alarmPolicy.ts` | Should the alarm sound |
| `src/core/errors.ts` | Error to on-page message |
| `src/browser/camera.ts` | Webcam start and stop |
| `src/browser/detector.ts` | Face Landmarker wrapper |
| `src/browser/beeper.ts` | Web Audio alarm sound |
| `src/ui/view.ts` | Overlay drawing, strip, right panel |
| `src/ui/snapshot.ts` | PNG download |
| `src/main.ts` | Wires the loop together |
| `scripts/fetch-assets.mjs` | Copies WASM and downloads the model |

---

### Task 1: Repo, tooling and shared types

**Files:**
- Create: `.gitignore`, `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `scripts/fetch-assets.mjs`, `src/core/types.ts`, `src/core/config.ts`

**Interfaces:**
- Produces, in `types.ts`:
  ```ts
  export type Label = 'sleeping' | 'drowsy' | 'angry' | 'laughing' | 'smiling' | 'crying' | 'neutral';
  export type Letter = 'A' | 'B' | 'C' | 'D';
  export interface Box { x: number; y: number; width: number; height: number } // 0..1 of the frame
  export interface RawFace { box: Box; scores: Record<string, number>; pitchDeg: number }
  export interface TrackedFace { letter: Letter; face: RawFace }
  export interface Signals { eyeClosure: number; lookDown: number; pitchDelta: number; smile: number; jawOpen: number; browDown: number; browInnerUp: number; frown: number }
  export interface Reading { label: Label; confidence: number } // confidence 0..100
  export interface Person { letter: Letter; box: Box; reading: Reading }
  export interface LogEntry { at: Date; text: string }
  ```
- Produces, in `config.ts`, `export const CONFIG` with these starting values (Task 7 tunes them):
  `maxFaces 4, minFrameGapMs 66, letterReleaseMs 2000, maxMatchDistance 0.25, baselineMs 2000, baselineMinSamples 10, eyesClosedAt 0.55, eyesClosedWhenLookingDownAt 0.8, lookDownAt 0.4, openGraceMs 200, slowBlinkMs 400, slowBlinkWindowMs 10000, slowBlinksForDrowsy 2, droopyEyesAt 0.35, droopyHoldMs 3000, headDroopDeg 15, headDroopEyesAt 0.45, headDroopHoldMs 1500, drowsyClearMs 2000, expressionLimitMs 5000, labelHoldMs 500, frameGapResetMs 2500, defaultDelaySec 3, minDelaySec 1, maxDelaySec 10, logCap 100, smileAt 0.35, strongSmileAt 0.5, laughJawAt 0.25, angryBrowAt 0.4, cryBrowAt 0.4, cryFrownAt 0.3, pitchSign 1`

- [ ] **Step 1: Create the repo and branch**

```bash
cd E:/FACIAL_EXPRESSION
git init -b main
git checkout -b feat/expression-demo
```

- [ ] **Step 2: Add `.gitignore`** with `node_modules/`, `dist/`, `.superpowers/`, `public/mediapipe/`, `.env`.

- [ ] **Step 3: Add `package.json`** with exact versions above and these scripts: `dev` = `vite`, `build` = `tsc --noEmit && vite build`, `test` = `vitest run`, `setup:assets` = `node scripts/fetch-assets.mjs`. Run `npm install`.

- [ ] **Step 4: Write `scripts/fetch-assets.mjs`**

Copies `node_modules/@mediapipe/tasks-vision/wasm/` to `public/mediapipe/wasm/`, and downloads `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` to `public/mediapipe/face_landmarker.task`. Skips the download if the file exists. Exits non-zero with a readable message on failure.

- [ ] **Step 5: Add `index.html`, `tsconfig.json` (strict), `vite.config.ts`, `types.ts`, `config.ts`**

`index.html` holds a bare page with `<script type="module" src="/src/main.ts">`; add an empty `src/main.ts`.

- [ ] **Step 6: Verify**

Run: `npm run setup:assets && npm run build`
Expected: `public/mediapipe/face_landmarker.task` exists, build exits 0.

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "chore: set up Vite, TypeScript and MediaPipe assets"
```

Confirm first that `.superpowers/` is not staged. This commit includes the spec and plan.

---

### Task 2: Face tracker

**Files:**
- Create: `src/core/tracker.ts`
- Test: `tests/tracker.test.ts`

**Interfaces:**
- Consumes: `RawFace`, `TrackedFace`, `Letter`, `CONFIG`.
- Produces:
  ```ts
  export class Tracker {
    update(faces: RawFace[], nowMs: number): TrackedFace[]; // faces visible this frame
    activeLetters(): Letter[];                              // includes faces in their 2s grace
  }
  ```

- [ ] **Step 1: Write failing tests**

Helper `face(x, y)` builds a `RawFace` with a 0.1 by 0.1 box.

- `first face gets A, second gets B`
- `a face that moves slightly keeps its letter`: update at (0.2,0.2), then (0.22,0.2), expect `A`.
- `two faces keep letters when input order swaps`
- `face gone under 2 seconds keeps its letter on return`: update at t=0, empty at t=1000, same spot at t=1900, expect `A` and `activeLetters()` still has `A` at t=1000.
- `face gone over 2 seconds releases its letter`: empty at t=2100, then `activeLetters()` is `[]`.
- `new face takes the lowest free letter`: A and B active, A released, new face gets `A`.
- `a fifth face is ignored and steals no letter`: 5 far-apart faces gives 4 results lettered A to D.
- `a face beyond maxMatchDistance is treated as new`: A at (0.1,0.1), next frame one face at (0.9,0.9), expect `B`.

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/tracker.test.ts`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `Tracker`**

Match by box centre distance. Greedy: sort all face-to-track pairs by distance, take the closest first, skip pairs over `maxMatchDistance`.

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/tracker.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/tracker.ts tests/tracker.test.ts
git commit -m "feat: track faces with stable letters"
```

---

### Task 3: Signals, baseline and classifier

**Files:**
- Create: `src/core/signals.ts`, `src/core/classifier.ts`
- Test: `tests/signals.test.ts`, `tests/classifier.test.ts`

**Interfaces:**
- Consumes: `RawFace`, `Signals`, `Reading`, `CONFIG`.
- Produces:
  ```ts
  // signals.ts
  export function pitchFromMatrix(m: ArrayLike<number>): number; // degrees, column-major 4x4
  export class Baseline { toSignals(face: RawFace, nowMs: number): Signals }
  // classifier.ts
  export class PersonClassifier { update(s: Signals, nowMs: number, delaySec: number): Reading }
  ```

**Signal rules:**
- Paired blendshapes are averaged left and right: `eyeBlink*` gives raw closure, `eyeLookDown*` gives `lookDown`, `mouthSmile*`, `browDown*`, `mouthFrown*`. `jawOpen` and `browInnerUp` are used directly.
- `pitchFromMatrix`: `CONFIG.pitchSign * atan2(m[6], m[10])` in degrees. Positive means the head tips forward. Task 7 confirms the sign on camera.
- Baseline collects samples for at least `baselineMs` and until it has `baselineMinSamples` samples with raw closure under 0.5. It then freezes the median closure and median pitch.
- Before it freezes: closure base 0, `pitchDelta` 0.
- After: `eyeClosure = clamp01((raw - base) / (1 - base))`, `pitchDelta = pitchDeg - basePitch`.

**Classifier rules, in order:**
- `eyesClosed`: `eyeClosure >= eyesClosedAt`, or `eyesClosedWhenLookingDownAt` instead if `lookDown >= lookDownAt`.
- The closed count starts when eyes close and resets once eyes have been open for `openGraceMs`.
- If updates are more than `frameGapResetMs` apart, reset all timers and history. This must stay above `letterReleaseMs`, so a face lost briefly keeps its count.
- A strong expression means `smile >= strongSmileAt`, or the angry or crying rule below is true.
- **Sleeping:** closed count reaches `delaySec * 1000`. If a strong expression is present, it must also reach `expressionLimitMs`. Confidence is `eyeClosure * 100`.
- **Drowsy:** any of these, and it stays until none has been true for `drowsyClearMs`:
  - `slowBlinksForDrowsy` closures of `slowBlinkMs` or longer within `slowBlinkWindowMs`
  - `eyeClosure >= droopyEyesAt` held for `droopyHoldMs`, while `lookDown < lookDownAt`
  - `pitchDelta >= headDroopDeg` and `eyeClosure >= headDroopEyesAt` held for `headDroopHoldMs`, while `lookDown < lookDownAt`

  Confidence is `max(eyeClosure, 0.5) * 100`. A strong expression suppresses drowsy.
- **Laughing:** `smile >= strongSmileAt` and `jawOpen >= laughJawAt`. Confidence is `smile * 100`.
- **Smiling:** `smile >= smileAt`. Confidence is `smile * 100`.
- **Angry:** `browDown >= angryBrowAt` and `smile < 0.2`. Confidence is `browDown * 100`.
- **Crying:** `browInnerUp >= cryBrowAt` and `frown >= cryFrownAt`. Confidence is the average of the two times 100.
- **Neutral:** otherwise. Confidence is `(1 - strongest of smile, browDown, browInnerUp) * 100`.
- **Display hold:** a new candidate label shows only after `labelHoldMs`, except `sleeping` and `drowsy`, which show at once.
- All confidences are rounded integers from 0 to 100.

- [ ] **Step 1: Write failing tests**

`tests/signals.test.ts`:
- `identity matrix gives 0 pitch`
- `20 degree X rotation matrix gives 20 pitch`
- `averages left and right scores`
- `baseline raises closure scale for naturally narrow eyes`: open-eye samples at raw 0.3 for 2.5s, then raw 0.3 gives `eyeClosure` 0.
- `arriving with closed eyes does not freeze a bad baseline`: raw 0.9 for 3s, then `eyeClosure` still 0.9.
- `pitchDelta is measured from the person's normal pitch`: baseline at 10, then 28 gives 18.

`tests/classifier.test.ts`, with a helper `run(classifier, signals, fromMs, toMs, delaySec = 3)` stepping 66ms and returning the last reading:
- `normal blinks for one minute never give sleeping or drowsy`: 200ms closed every 4s for 60s.
- `eyes closed 3 seconds gives sleeping`, and at 2.5 seconds it does not.
- `delay setting of 1 gives sleeping after 1 second`
- `a brief open-eye glitch under 200ms does not reset the count`
- `opening eyes for longer resets the count`
- `two slow blinks within 10 seconds give drowsy`
- `drooping eyes for 3 seconds give drowsy`
- `head droop with half-closed eyes for 1.5 seconds gives drowsy`
- `looking down with eyes open never triggers`: `pitchDelta` 25, `lookDown` 0.7, `eyeClosure` 0.5, for 30s.
- `smile with closed eyes stays laughing at 4 seconds and becomes sleeping at 5`
- `jaw open with closed eyes still becomes sleeping at 3 seconds`
- `angry with closed eyes becomes sleeping at 5 seconds`
- `each of laughing, smiling, angry, crying and neutral is reached from its scores`
- `a label flicker under 500ms is not displayed`
- `a 5 second frame gap does not count as closed time`: closed for 2s, jump 5s, closed one frame, not sleeping.
- `a face lost for 1.5 seconds keeps its closed count`: closed for 2s, no updates for 1.5s, closed one frame, sleeping.
- `confidence stays an integer from 0 to 100`

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/signals.test.ts tests/classifier.test.ts`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement `signals.ts` and `classifier.ts`** to the rules above.

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/signals.test.ts tests/classifier.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/signals.ts src/core/classifier.ts tests/signals.test.ts tests/classifier.test.ts
git commit -m "feat: classify expressions and sleep from face signals"
```

---

### Task 4: Wording, activity log and alarm policy

**Files:**
- Create: `src/core/text.ts`, `src/core/activityLog.ts`, `src/core/alarmPolicy.ts`
- Test: `tests/text.test.ts`, `tests/activityLog.test.ts`, `tests/alarmPolicy.test.ts`

**Interfaces:**
- Consumes: `Label`, `Letter`, `Person`, `LogEntry`, `CONFIG`.
- Produces:
  ```ts
  // text.ts
  export function faceCountText(n: number): string;
  export function personLine(p: Person): string;
  // activityLog.ts
  export function diffEvents(prev: Map<Letter, Label>, next: Map<Letter, Label>, at: Date): LogEntry[];
  export function appendLog(log: LogEntry[], entries: LogEntry[]): LogEntry[]; // newest first, capped
  export function formatEntry(e: LogEntry): string;                            // "12:04:31 B fell asleep"
  // alarmPolicy.ts
  export function shouldSound(labels: Label[], muted: boolean): boolean;
  ```

**Exact copy:**
- Count: `No faces found`, `1 face found`, `2 faces found`.
- Line: `Person A is sleeping 92%`, `is about to sleep`, `is angry`, `is laughing`, `is smiling`, `is crying`, `looks neutral`.
- Log: `fell asleep`, `looks drowsy`, `looks angry`, `started laughing`, `started smiling`, `looks like crying`, `looks neutral`, `joined`, `left`.

- [ ] **Step 1: Write failing tests**

- `faceCountText` for 0, 1 and 2, and `personLine` for all seven labels, using the copy above.
- `no change gives no log entries`
- `a label change gives one entry`: B from neutral to sleeping gives `B fell asleep`.
- `a new letter gives joined, then a missing letter gives left`
- `log is newest first and capped at 100`
- `formatEntry uses 24 hour HH:MM:SS`
- `shouldSound is true for sleeping or drowsy, false otherwise, and false when muted`
- `shouldSound is false for an empty list`: the alarming person left, so no labels remain.

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/text.test.ts tests/activityLog.test.ts tests/alarmPolicy.test.ts`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement the three modules.**

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core tests
git commit -m "feat: add panel wording, activity log and alarm policy"
```

---

### Task 5: Camera, detector and sound

**Files:**
- Create: `src/core/errors.ts`, `src/browser/camera.ts`, `src/browser/detector.ts`, `src/browser/beeper.ts`
- Test: `tests/errors.test.ts`

**Interfaces:**
- Consumes: `RawFace`, `CONFIG`, `pitchFromMatrix`.
- Produces:
  ```ts
  // errors.ts
  export function cameraErrorMessage(err: unknown): string;
  export function modelErrorMessage(err: unknown): string;
  // camera.ts
  export function startCamera(video: HTMLVideoElement, onEnded: () => void): Promise<void>;
  // detector.ts
  export class Detector {
    static create(): Promise<Detector>;
    detect(video: HTMLVideoElement, nowMs: number): RawFace[];
  }
  // beeper.ts
  export class Beeper {
    constructor();                 // call inside the Start click, so audio is unlocked
    setVolume(v: number): void;    // 0..1
    setSounding(on: boolean): void; // safe to call every frame
  }
  ```

- [ ] **Step 1: Write failing tests in `tests/errors.test.ts`**

Build errors with `new DOMException('x', name)`:
- `NotAllowedError` message contains `allow camera access`.
- `NotFoundError` message contains `No camera found`.
- `NotReadableError` message contains `another app`.
- Unknown errors give a general message and never an empty string.
- `modelErrorMessage` contains `npm run setup:assets`.

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/errors.test.ts`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `errors.ts`.**

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/errors.test.ts`
Expected: PASS.

- [ ] **Step 5: Implement `camera.ts`**

Uses `getUserMedia({ video: { width: 1280, height: 720 }, audio: false })`. Calls `onEnded` when the video track fires `ended`.

- [ ] **Step 6: Implement `detector.ts`**

`FilesetResolver.forVisionTasks('/mediapipe/wasm')`, then `FaceLandmarker.createFromOptions` with `modelAssetPath: '/mediapipe/face_landmarker.task'`, `runningMode: 'VIDEO'`, `numFaces: CONFIG.maxFaces`, `outputFaceBlendshapes: true`, `outputFacialTransformationMatrixes: true`. Use `delegate: 'GPU'` and retry with `'CPU'` if creation throws.

`detect` calls `detectForVideo(video, nowMs)`. For each face it builds the box from the min and max of `faceLandmarks[i]`, `scores` from `faceBlendshapes[i].categories`, and `pitchDeg` from `facialTransformationMatrixes[i].data`.

- [ ] **Step 7: Implement `beeper.ts`**

One `AudioContext`, one `GainNode`. While sounding, a repeating two-tone beep from an `OscillatorNode` (880 Hz, 300ms on, 200ms off).

- [ ] **Step 8: Verify types**

Run: `npm run build`
Expected: exit 0.

- [ ] **Step 9: Commit**

```bash
git add src tests
git commit -m "feat: add camera, face detector and alarm sound"
```

---

### Task 6: Screen, snapshot and main loop

**Files:**
- Create: `src/ui/view.ts`, `src/ui/snapshot.ts`, `src/styles.css`
- Modify: `index.html`, `src/main.ts`

**Interfaces:**
- Consumes: everything above.
- Produces:
  ```ts
  // view.ts
  export function drawOverlay(canvas: HTMLCanvasElement, people: Person[]): void;
  export function renderStrip(el: HTMLElement, people: Person[]): void;
  export function renderLog(el: HTMLElement, log: LogEntry[]): void;
  export function showMessage(el: HTMLElement, text: string | null): void;
  // snapshot.ts
  export function saveSnapshot(video: HTMLVideoElement, overlay: HTMLCanvasElement): void;
  ```

**Layout:** matches the approved mockup. The camera is top left, the bottom strip is under it, and the right panel (about 260px wide) holds Activity log, Alarm (mute button, volume slider, delay slider from 1 to 10, default 3) and the Snapshot button. A "Start camera" button sits over the camera area until clicked.

**Mirroring trap:** `video` gets `transform: scaleX(-1)` in CSS. The overlay canvas does NOT, or its letters would read backwards. Instead `drawOverlay` flips box positions with `x' = 1 - x - width`.

**Colours:** box and letter are green, and red for `sleeping` or `drowsy`. In the strip, those people's lines are also red.

- [ ] **Step 1: Build the page, styles and `view.ts`.**

- [ ] **Step 2: Implement `saveSnapshot`**

Draws the video flipped to match the screen, then the overlay, to an off-screen canvas. Downloads `snapshot-YYYYMMDD-HHmmss.png` through a temporary link.

- [ ] **Step 3: Implement `main.ts`**

On Start click: create `Beeper`, show `Loading...`, then `Detector.create()` and `startCamera()`, each in `try/catch` that calls `showMessage` with the matching error text.

Loop on `requestAnimationFrame`, skipping frames under `minFrameGapMs` apart:
1. `detector.detect`, wrapped in `try/catch`.
2. `tracker.update`.
3. For each tracked face, its own `Baseline` and `PersonClassifier`, stored in a `Map` by letter.
4. Build `Person[]`.
5. Delete state for letters no longer in `tracker.activeLetters()`.
6. Render overlay, strip and log.
7. `beeper.setSounding(shouldSound(labels, muted))`.

Labels include faces in their 2-second grace, using their last reading, and the log's `left` entry fires on release. If `onEnded` fires, stop the loop, call `setSounding(false)` and show the camera message.

Add a debug line of raw signals, shown only with `?debug=1`, for Task 7.

- [ ] **Step 4: Verify in a real browser**

Run: `npm run dev`, open `http://localhost:5173`, click Start.
Expected: video shows, a box and letter sit correctly on your face, letter reads the right way round, the strip updates, no console errors, and the Network tab shows only `localhost` requests.

- [ ] **Step 5: Commit**

```bash
git add index.html src
git commit -m "feat: add live screen, panels, snapshot and main loop"
```

---

### Task 7: Tuning, README and final checks

The owner does this with their face on camera, since an agent cannot act out expressions. Testing is by the owner alone. The second face is a photo or video on a phone.

**Files:**
- Modify: `src/core/config.ts`
- Create: `README.md`

- [ ] **Step 1: Confirm pitch direction**

With `?debug=1`, tip your head forward. `pitchDelta` must rise. If it falls, set `pitchSign` to `-1`.

- [ ] **Step 2: Tune `config.ts` from the debug line**

Act out each expression and adjust thresholds. After each change, run `npm test`. If a test's scenario still holds but its values no longer match, update the signals in that test. Never delete a test.

- [ ] **Step 3: If angry can't be made usable, stop and ask the owner**

The spec's fallback is adding `@vladmandic/face-api` for angry and sad only. Do not add it without the owner's agreement.

- [ ] **Step 4: Write `README.md`**

Must include:
- What the project is, and setup: `npm install`, `npm run setup:assets`, `npm run dev`.
- Offline use, and that the setup steps need internet only once.
- Not for driver or medical use.
- The confidence score is rule strength, not a true probability.
- All known limits from spec section 8.
- Troubleshooting for camera blocked and model files missing.

- [ ] **Step 5: Run the spec's success checks**

Each of these must pass:
- Sleeping, about to sleep, laughing and smiling are correct nearly every try.
- Angry is correct most of the time when clearly acted out. Crying is best effort.
- One minute of normal blinking gives no alarm.
- Nodding forward with eyes closed sets off sleep. Looking down at the controls with eyes open does not.
- A phone photo as a second face gets its own letter and label, and both letters stay put.
- Unplugging or blocking the camera shows the message and stops the sound.
- With Wi-Fi off, the app still starts and works.
- Network tab shows no image or video going anywhere.

- [ ] **Step 6: Final checks**

Run: `npm test && npm run build`
Expected: all pass, exit 0.

- [ ] **Step 7: Commit**

```bash
git add README.md src tests
git commit -m "docs: add README and tune detection thresholds"
```
