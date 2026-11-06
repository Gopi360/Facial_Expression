# Python Rebuild, Stage 1: Design

Date: 2026-11-06
Owner: Supriya Gope
Status: Draft, waiting for owner approval
Replaces: `2026-11-06-facial-expression-demo-design.md` (the TypeScript version)

## 1. Intent

A demo that watches the laptop camera and says, live, what each person is doing, and keeps proof of anything flagged. It may be used for exams later, but today it is a demo.

The owner tested the TypeScript version on a real camera and found:

- Letters changed when a person left and came back. The owner wants people recognised.
- Neutral and smiling worked. Angry, sleeping and crying never showed. Yawning did not exist.

The owner then decided to move fully to Python. The main technical gain is using trained models in place of hand-tuned rules for expressions.

## 2. Stages

- **Stage 1 (this document):** Python rebuild with trained expressions, recognition, yawning, sleep alarm, red marks, session logs and snaps.
- **Stage 2 (later, separate design):** ear visibility and device detection. See section 12.

## 3. Goals

- Anyone can sit at the camera with no sign-up or calibration.
- Several people at once. Each gets a letter: Person A, Person B and so on.
- A person keeps their letter when they leave and return during a session.
- Text results update live in a panel, never on the video.
- Red box and red text for sleeping or about to sleep.
- Sound plays for sleeping or about to sleep.
- Every session is logged to its own file. Every red mark takes a timestamped snap.

## 4. Non-goals

- Recognising a person across sessions, or knowing anyone's name.
- Accounts, a database, or a page for reviewing logs.
- Cloud hosting. It runs locally on the laptop.
- Mobile layout.
- Ear and device detection (Stage 2).

## 5. Anti-goals

These would make it a failure even if it technically works:

- Text drawn on the camera view.
- Video, images, logs or snaps leaving the laptop.
- The alarm firing on ordinary blinks, or on a laughing person with eyes squeezed shut.
- The alarm staying on for someone who is awake.
- Labels flickering too fast to read.
- Disk filling with near-identical snaps.
- Presenting it as a safety device, or a flag as a proven fact. The README must say it is not for driver or medical use, and that a flag needs human review.

## 6. Decided by the owner

**Screen (carried over from the approved layout)**

- Camera view on the left, with only a thin box and small letter on each face.
- Bottom strip under the camera: face count and one line per person, with the confidence score beside the expression.
- Right panel: activity log, alarm controls (mute, volume, seconds before sound), snapshot button.

**Behaviour**

- Expressions: sleeping, about to sleep, yawning, angry, laughing, smiling, crying, neutral.
- "Sleeping" means eyes closed for about 3 seconds. "About to sleep" means drooping eyes, slow blinks or a head drooping forward.
- A head drooping forward counts as a sleep sign. Looking down with eyes open must not.
- Crying is best effort.
- People are recognised during a session. A returning person keeps their letter.

**Records**

- Every session is one log file in a `logs` folder under the project.
- Snaps go in a `snaps` folder under the project.
- Any red mark takes a timestamped snap as proof.
- Someone who is flagged stays flagged. The app records, and a person judges from the proof later.

**Stack**

- Python for all detection and logic.

## 7. Speed test results

Measured on the owner's laptop (Intel i3-1005G1, 4 threads, 12 GB RAM, no AI graphics card, Python 3.12.5). One sample photo, one face, no camera or Chrome running, so real numbers will be lower.

| Check | Model | Time | Per second |
|---|---|---|---|
| Eyes, mouth, head tilt | MediaPipe Face Landmarker | 12 ms | 83 |
| Expressions | HSEmotion `enet_b0_8_best_vgaf` (ONNX) | 21 ms per face | 47 |
| Finding faces | YuNet | 20 ms | 50 |
| Recognition | SFace | 20 ms per face | 50 |

All together the process used under 500 MB of memory. Free memory was 4.6 GB.

## 8. Technical approach (author's call, not chosen by the owner)

The owner chose Python. The split below is the author's recommendation and can be overruled when reviewing this document.

**Recommended: Python does all the thinking, and a thin browser page does only the camera, screen and sound.**

| Option | Pros | Cons |
|---|---|---|
| **Browser captures, sends frames to a local Python server (recommended)** | Reuses the approved screen. Browser handles sound and controls well. Can later run on someone else's computer for exams. | Small amount of JavaScript remains. Frames pass through a local socket. |
| Python reads the camera with OpenCV, streams video to the page | Almost no JavaScript. | Exam use would need rebuilding. Video stream adds delay. |
| Python desktop window, no browser | Pure Python. | A new screen to build, and it loses the approved layout. |

**Parts**

- **Server:** FastAPI with one WebSocket, bound to `127.0.0.1` only. It also serves the page.
- **Page:** plain HTML, CSS and a small JavaScript file. No Node, no build step. Reuse `index.html`, `src/styles.css` and the drawing logic in `src/ui/view.ts` from the TypeScript version. The debug view stays, under `?debug=1`.
- **Frames:** 640x480 JPEG. The page sends the next frame only after the previous reply arrives, so frames never pile up. Target about 15 per second.
- **Reply:** JSON with each person's letter, box, label, confidence and red flag, plus log lines and whether the alarm should sound.
- **Models:** MediaPipe for eyes, mouth and head tilt on every frame. HSEmotion for expressions about 3 times a second per face. SFace for recognition, see section 9.
- **Finding faces (author's call):** MediaPipe drives detection and frame-to-frame tracking. SFace crops are aligned from MediaPipe landmarks. Use YuNet only if that alignment proves poor in testing.
- **Speed floor (author's call):** 15 frames a second is the target and 8 is acceptable. When busy, slow the expression and recognition checks first. MediaPipe stays on every frame.
- **Tests:** pytest. Detection logic must be testable with no camera and no models.
- **Setup:** Python 3.12 virtual environment, exact versions pinned. A setup script downloads models into a git-ignored `models` folder, so the app works offline afterwards.
- **No PyTorch.** Use ONNX Runtime. PyTorch failed to install on this laptop and is not needed.
- **Docker:** not needed, nothing is deployed to production.

**Carry-over from the TypeScript version**

The source is `src/core/` and `tests/` on the `feat/expression-demo` branch, with all thresholds in `src/core/config.ts`. Python uses the same MediaPipe model file and the same blendshape names, so signals map directly. Port the logic and the tests as the starting point. Do not start the sleep rules again from nothing. If the debug capture in section 11 shows a rule can't work on real numbers, change the rule to fit the data and keep its test's intent. These hard-won behaviours must survive, and each needs a test:

- Ordinary blinks and double blinks never count as slow blinks.
- A brief open-eye glitch doesn't reset the closed count.
- A frame gap, such as a paused tab, doesn't count as closed time.
- The label hold times how long the candidate has differed from the shown label, so alternating labels can't stay stuck.
- One noisy frame can't set off or cancel the alarm.
- Droopy eyes keep their count through normal blinks.
- Someone already drowsy stays drowsy until sleeping takes over.
- Each face gets an automatic baseline for eye openness and head position.
- Two slow blinks are one event, not a long hold.
- The debug view and downloadable debug log.

The TypeScript code stays on the `feat/expression-demo` branch for reference.

## 9. Behaviour details

**Expressions from the trained model (author's call)**

| Model output | Shown as |
|---|---|
| Anger | angry |
| Sadness | crying |
| Happiness with mouth open | laughing |
| Happiness | smiling |
| Neutral | neutral |
| Contempt, Disgust, Fear, Surprise | neutral |

Confidence is the model's score for that class. For sleep labels it remains a rule strength, not a probability. The README must say so.

**Yawning (author's call)**

- Mouth wide open held for about 1.5 seconds, without a smile. The exact rule is tuned on camera.
- Shows as "yawning" and is logged.
- Yawning is not red and plays no sound on its own.

**Priority between labels:** sleep rules first, as in the carry-over rules, then yawning, then the trained model's expression.

**Recognition**

- Each person seen in a session has a stored face signature, in memory only, never on disk.
- A face being followed keeps its letter by position from frame to frame.
- A new face is compared against all stored signatures. If it matches, it takes that letter back. If not, it takes the next free letter.
- SFace's usual cosine match threshold of 0.363 is the starting value, tuned on camera.
- Author's call: a new face shows no letter in the panel until its first comparison is done, a fraction of a second, to avoid a wrong letter.
- Signatures are refreshed from good frontal views only, and clear when the app stops.
- Baseline and sleep history follow the person, so a returning person is not relearned.
- Letters run A to Z per session. At most 4 faces are processed at once.
- Author's call: if more than 4 faces are in view, the 4 largest are processed and the strip says so. After Z, new people show as "Person ?" and are still logged.
- Known risks: relatives or similar faces may share a letter, and strong side angles may get a new letter.

**Red marks**

- Red box and red text apply to sleeping and about to sleep. Stage 2 adds its own red reasons.
- Each red mark also shows the reason in the panel.

**Alarm**

- Delay setting defaults to 3 seconds, range 1 to 10. It sets when "sleeping" starts and when the sound starts.
- "About to sleep" and head droop can sound before that delay.
- Plays while any person is sleeping or about to sleep. Stops when nobody is, or when muted.
- Browsers block sound until a click, so the Start button unlocks it.
- Sound plays in the page. The server only says when.

**Snaps**

- One snap when a person turns red, not on every frame. If the person stays red, no more snaps. Returning to normal and turning red again is a new event and a new snap.
- A red mark that lasts under a second takes no snap, so flicker can't create photos.
- The snap is the whole frame with boxes and letters, flipped to match what the screen showed.
- Saved by the server to `snaps/`, for example `2026-11-06_16-30-00_16-42-11_A_sleeping.jpg` (session start, event time, person, reason).
- The manual Snapshot button saves to the same folder.
- If a snap can't be saved, the page shows a message and the log says so. Detection keeps running.

**Session logs**

- A session starts at Start camera and ends when the camera stops or the page closes.
- One file per session: `logs/session-2026-11-06_16-30-00.log`.
- Plain readable lines, written as they happen, so a crash loses nothing. For example:
  `16:42:11 | Person A | sleeping | 92% | snaps/2026-11-06_16-30-00_16-42-11_A_sleeping.jpg`
- Records label changes, joined, left, red marks, snaps and alarm start and stop.
- The right panel shows the same events live.

**One client at a time (author's call):** the server runs one session. A second tab is told the app is already open. A page reload ends the session and starts a new one. Log lines are written straight to disk, so a crash leaves a usable file.

**Recording notice:** the page always shows that it is recording logs and snaps.

**Screens with nothing to report:** "No faces found", a loading message while models load, and clear messages for a blocked camera, a stopped server, or missing models saying what to do. Never a blank page.

**Hidden tab:** the page stops sending frames, so detection pauses. The page silences the alarm and shows a note on return. The README states this limit.

## 10. Known limits

State these in the README:

- A head flat on a desk has no visible face, so the app sees the person leave.
- Looking down, glasses, dim light and turned heads reduce accuracy.
- Expression reading is a rough guess at a face, not a measure of feeling. Crying is the weakest.
- Very narrow eyes or a naturally low brow can cause wrong flags.
- Recognition can confuse similar faces and can miss a strong side angle.
- Tuned on the owner's face only.

## 11. How we'll know it worked

The owner tests on camera. Second faces use a photo or video on a phone.

- Runs from the README in a few commands.
- Leave for 10 seconds and return: still Person A, every try.
- Sleeping, about to sleep, yawning, laughing and smiling correct nearly every try.
- Angry correct most of the time when clearly acted out. Crying is best effort.
- Normal blinking for one minute gives no alarm.
- Nodding forward with eyes closed sets off sleep. Looking down at the controls with eyes open does not.
- A person turning red makes exactly one snap, and it appears in that session's log.
- Two sessions make two log files.
- With Wi-Fi off, the app still starts and works.
- Nothing is sent anywhere but `127.0.0.1`.

**How firm these are (author's call):** these are demo targets, not hard gates, except recognition on the owner's face and the alarm rules. If angry falls short with HSEmotion, try a hybrid of the model and the brow scores, then report to the owner rather than holding the build. A model that fails badly is reported early, not hidden.

The first build step after the camera works is capturing the owner's real signal numbers with the debug log, before tuning anything. The TypeScript version never reached sleep on a real face, and the cause is still unknown. The main suspect is the eye and gaze scores on closed eyes, so this must be checked, not assumed.

## 12. Stage 2 notes (not in this build)

Recorded so Stage 1 leaves room for it:

- Check order per person: are ears visible? If not, say "ears are not visible". If so, look for headset, headphones, earbuds or earpods, and say "device found in the ears".
- Both are red with a snap, silent unless decided otherwise. Red reasons must allow more than one at a time.
- A person can have an expression and a warning together, for example "Person A is smiling, device found in the ears".
- A person with covered ears stays red, and the snap lets a human judge.
- No ready-made earbud model was found. Training is expected, on Google Colab. Check licenses first: Ultralytics YOLO is AGPL-3.0.
- Before any real exam use: consent, a deletion rule for photos, and a review of fairness for people with long hair or head coverings.

## 13. Left to the builder

- Exact thresholds and rules, tuned on a real camera.
- Project folder layout.
- Alarm sound, visual styling within the approved layout.
- How heavy checks are scheduled, within the rates in section 8.
- Confirming each model's license before relying on it. A non-commercial license is acceptable for this demo, but must be recorded in the README as a blocker for commercial or exam use.
