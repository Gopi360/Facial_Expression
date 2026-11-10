# Live Facial Expression Demo: Design

Date: 2026-11-06
Owner: Supriya Gope
Status: Replaced by `2026-11-06-python-rebuild-design.md`. Code stays on `feat/expression-demo`.

## 1. Intent

A demo web page that watches the laptop camera and says, live, what each person in view is doing: sleeping, about to sleep, angry, laughing, smiling, crying or neutral. If someone is sleeping or about to sleep, it plays a sound.

This is a personal demo project, not client work. It needs to impress when run live on a laptop. The sleep alarm is the feature the owner cares about most ("this one sounds more better").

## 2. Goals

- Anyone can sit at the camera. No sign-up, setup or per-person calibration.
- Works with several people at once. Each face gets a letter: Person A, Person B and so on.
- Text results update live in a panel, never on top of the video.
- Sound plays when anyone is sleeping or about to sleep.
- Right panel offers: activity log, alarm controls, confidence score, snapshot button.

## 3. Non-goals

- Recognising who a person is. Letters are only for the current session.
- Accounts, a database, or saved history between sessions.
- Cloud hosting. It runs locally on the laptop.
- Mobile layout.
- Sleepiness meter and session summary. Both were offered and not chosen.

## 4. Anti-goals

These would make it a failure even if it technically works:

- Text drawn on the camera view.
- Video or images leaving the laptop.
- The alarm firing on ordinary blinks, or on a laughing person whose eyes are squeezed shut.
- Labels flickering so quickly they can't be read.
- Presenting it as a safety device. It is a demo, and the README must say it is not for driver or medical use.

## 5. Decided by the owner

**Screen layout**

- Camera view is the main area on the left.
- Camera view shows only a thin box and a small letter on each face.
- A bottom strip under the camera shows the face count and one line per person, for example "2 faces found · Person A is laughing · Person B is sleeping".
- A right panel holds the extras.

**Right panel contents**

- Activity log: timestamped changes, for example "12:04:31 B fell asleep".
- Alarm controls: mute, volume, and seconds before the sound plays.
- Confidence score: shown next to each person's expression, for example "angry 82%". The owner approved placing it beside the expression, so it appears in the bottom strip line.
- Snapshot button.

**Behaviour the owner approved in the playback**

- Expressions: sleeping, about to sleep, angry, laughing, smiling, crying, neutral.
- "Sleeping" means eyes closed for about 3 seconds. "About to sleep" means drooping eyes or slow blinks.
- Snapshots save to the laptop only. Nothing is uploaded.
- Letters can change if someone leaves the view and comes back.
- Crying is best effort. It will mostly read a sad, scrunched face and will sometimes be wrong.

## 6. Technical approach (builder's call, not chosen by the owner)

The owner said to go ahead without specifying a stack, so these are the author's recommendations. The owner can overrule them when reviewing this document.

**Recommended: everything runs in the browser.** TypeScript and Vite, using MediaPipe Face Landmarker (`@mediapipe/tasks-vision`), with no backend.

| Option | Pros | Cons |
|---|---|---|
| **Browser only, MediaPipe (recommended)** | Video never leaves the page, so the privacy goal holds by design. No server to run. Fast on a laptop. Eye and mouth measurements suit sleep detection well. | Expressions come from hand-tuned rules, not a trained emotion model. Angry and crying need tuning. |
| Browser, MediaPipe plus `@vladmandic/face-api` | Trained model for angry and sad. | Two models per frame is slower. Adds a dependency. |
| Python backend (FastAPI, OpenCV, DeepFace) | Python is a preferred language and has many emotion models. | Must stream video to a server. More to install. Slower and more fragile for a live demo. |

**Fallback:** if rules can't produce a usable "angry" in testing, add the second option's expression model for angry and sad only.

**Face Landmarker facts that shape the design** (from the current MediaPipe docs):

- `numFaces` defaults to 1 and must be raised. Set it to 4.
- Built-in smoothing applies only when `numFaces` is 1, so this app must do its own smoothing.
- `outputFacialTransformationMatrixes: true` is required, to read head tilt.
- `outputFaceBlendshapes: true` is required. It gives scores such as `eyeBlinkLeft`, `mouthSmileLeft`, `jawOpen`, `browDownLeft`, `browInnerUp` and `mouthFrownLeft`.
- `detectForVideo()` runs synchronously and blocks the page, so detection should be capped at about 15 per second. Move it to a web worker only if the page stutters.
- `Left` and `Right` blendshapes refer to the person's own face. Passing a mirrored image swaps them, so mirror the video with CSS only.

**Parts**

1. Camera: starts the webcam, handles errors.
2. Detector: wraps Face Landmarker and returns faces with boxes and blendshapes.
3. Tracker: keeps a stable letter on each face between frames.
4. Classifier: pure functions that turn blendshapes and recent history into an expression and a score.
5. Alarm: plays the sound using the Web Audio API, so no sound file is needed.
6. UI: video with overlay canvas, bottom strip, right panel.

Keep the classifier and tracker free of browser code so they can be unit tested with Vitest.

## 7. Behaviour details

**Start.** The page shows a "Start camera" button. This click also unlocks audio, because browsers block sound until the user interacts with the page.

**Classification**

- Sleep state is judged over time from eye closure, not from a single frame. Ordinary blinks must not count.
- Author's call: while eyes are closed, a clear smile counts as laughing, so a laughing person doesn't set off the alarm. An open mouth alone does not count, so a yawn or a slack sleeping mouth stays on the sleep path.
- Author's call: that exception lasts at most 5 seconds of continuous eye closure. After that the person counts as sleeping even if smiling, because real laughter doesn't keep eyes shut that long. Angry and crying faces with eyes shut follow the same limit.
- A label must hold for about 0.5 seconds before it is shown, so it doesn't flicker. Author's call: sleeping and about to sleep skip this hold, because they are already judged over time.
- Owner's decision: a head drooping forward counts as a sleep sign, not just eyes closed while upright. Use head tilt together with eye closure.
- Looking down with eyes open must not count as sleep. Use head tilt plus the eye and gaze scores to tell the two apart, so glancing at the alarm controls does not set off the alarm.
- Author's call: a drowsy person's sleep count survives a brief loss of the face (up to the 2-second letter release), since a drooping head can drop out of detection for a moment.
- Author's call: each face gets an automatic baseline. The app learns that person's normal eye openness and head position during their first few seconds in view, and judges changes against it. No calibration step is shown to the user.
- If nothing is strong enough, the label is neutral.
- The confidence score is a rule strength from 0 to 100%, not a true statistical probability. The README must say so.

**Alarm**

- Author's call: the delay setting is one number that means "seconds of closed eyes". It defines both when the label becomes "sleeping" and when the sound starts, so the text and the sound always arrive together.
- Delay defaults to 3 seconds and can be set from 1 to 10.
- Author's call: the count runs separately for each person, starts when that person's eyes close, and resets when they open.
- Author's call: "about to sleep" has no closed-eyes count. It sounds as soon as that label appears, because its rule already looks at the last several seconds.
- Plays while any person is sleeping or about to sleep.
- Stops when nobody is in those states, or when muted.
- Mute and volume last for the current session only.

**Tracking and letters**

- Match faces between frames by nearest position.
- A new face takes the lowest free letter.
- A letter is released after the face has been gone for 2 seconds.
- Up to 4 faces are tracked. If more are in view, the extras are ignored.

**Activity log**

- Adds a line only when a person's displayed label changes, or when a person joins or leaves.
- Newest first, kept in memory, cleared on reload, capped at 100 lines.

**Snapshot**

- Downloads a PNG of the current frame with the boxes and letters, using the browser's normal download.

**Screens with nothing to report**

- No faces: strip reads "No faces found".
- Model loading: show a loading message.
- Camera blocked or missing: show a clear message saying what to do. Never a blank page.

## 8. Known limits

State these in the README:

- Someone who puts their head down on a desk has no visible face, so the app sees them leave rather than fall asleep.
- Looking down at a keyboard can still occasionally look like closed eyes, despite the checks above.
- Settings are tuned on the owner's face only, so other faces may be somewhat less accurate. Everyone still gets an expression.
- A person who arrives with eyes already closed gives a poor baseline until they open them.
- Glasses, dim light and turned heads reduce accuracy.
- Expression reading is a rough guess at a face, not a measure of what someone feels.

## 9. Constraints

- Must work offline during a demo. Serve the WASM files and the model from the project, not from a CDN. Pin the package version.
- Runs from `localhost`, which browsers treat as secure for camera access.
- Target browser: current Chrome on the owner's Windows laptop.
- Follow the owner's standards: README with setup steps, error handling on every camera and model call, no secrets, Git with feature branches and conventional commits.
- Docker is not needed, because nothing is deployed to production.

## 10. How we'll know it worked

- The owner can run it from the README in a few commands.
- On camera, sleeping, about to sleep, laughing and smiling are correct nearly every try.
- Angry is correct most of the time when clearly acted out. Crying is best effort.
- Normal blinking for one minute never triggers the alarm.
- Nodding forward with eyes closed triggers sleep, and looking down at the controls with eyes open does not.
- Two faces get separate letters and labels that stay put. Owner's decision: testing is by the owner alone, as no one else is available, so the second face can be a photo or video held up on a phone.
- Browser network tools show no image or video being sent anywhere.

## 11. Left to the builder

- Exact thresholds and blendshape combinations, tuned by testing on a real camera.
- The alarm's sound.
- Visual styling, within the approved layout.
- Whether a hidden debug view of raw scores is useful during tuning.
