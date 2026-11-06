# Live Facial Expression Demo

A browser-only demo that watches a webcam, finds up to four faces, gives each face a letter (A, B, C, D) and guesses an expression for each one: sleeping, about to sleep, laughing, smiling, angry, crying or neutral. It sounds an alarm when someone looks asleep or about to sleep. It is built with TypeScript, Vite and MediaPipe Face Landmarker.

Everything runs in the browser. No video or image is sent anywhere.

## Setup

You need Node.js and current Chrome (the target browser, on Windows).

```bash
npm install
npm run setup:assets
npm run dev
```

Then open http://localhost:5173/ and click Start camera. Allow camera access when Chrome asks.

`npm run setup:assets` copies the MediaPipe WASM files from `node_modules` into `public/mediapipe/wasm` and downloads the face model to `public/mediapipe/face_landmarker.task`. Both are served from the project, with no CDN at runtime. The generated files are git-ignored, so run this once on each fresh checkout.

## Offline use

The app works with no internet. Only `npm install` and `npm run setup:assets` need internet, and only once. After that you can turn Wi-Fi off and run `npm run dev` as normal. Camera access works on `localhost` because browsers treat it as secure.

## Other commands

```bash
npm test          # unit tests (vitest)
npm run build     # type-check and production build into dist/
```

## Important notes

- Not for driver monitoring, safety or medical use. It is a demo, not a safety system and not a diagnostic tool.
- The confidence number is rule strength, not a true probability. It shows how strongly the rules for a label were met. It does not mean "70% chance this is right".

## Known limits

- Someone who puts their head down on a desk has no visible face, so the app sees them leave rather than fall asleep.
- Looking down at a keyboard can still occasionally look like closed eyes, despite the checks in the code.
- Settings were chosen for one face only, so other faces may be somewhat less accurate. Everyone still gets an expression.
- A person who arrives with eyes already closed gives a poor baseline until they open them.
- Glasses, dim light and turned heads reduce accuracy.
- Expression reading is a rough guess at a face, not a measure of what someone feels.
- The detection numbers have not yet been tuned on a real camera. See "Tuning on camera" below. Expect some wrong labels, especially for angry and crying.

## Tuning on camera

The thresholds in `src/core/config.ts` are starting estimates. They have not been tuned, so treat the current accuracy as unknown until you have done the steps below.

1. Run `npm run dev` and open http://localhost:5173/?debug=1, then click Start camera.
2. With `?debug=1` a debug line appears for each face, in the form `A: eyeClosure 0.05  lookDown 0.02  pitchDelta 0.00  smile 0.01  jawOpen 0.00  browDown 0.00  browInnerUp 0.00  frown 0.00`. Each line starts with the face's letter, followed by its current signal values (two decimals). These are the numbers the rules compare against `CONFIG`. Values are relative to a baseline taken from your face in the first couple of seconds, so start with a relaxed, open-eyed face.
3. Check pitch direction first. Tip your head forward: `pitchDelta` must rise. If it falls, set `pitchSign` to `-1` in `src/core/config.ts`.
4. Act out each expression and watch which numbers move. Adjust the matching thresholds in `src/core/config.ts` (for example `smileAt`, `laughJawAt`, `angryBrowAt`, `cryBrowAt`, `eyesClosedAt`, `headDroopDeg`).
5. Run `npm test` after each change. If a test's scenario still holds but its values no longer match the new numbers, update the signals in that test. Do not delete tests.

If angry cannot be made reliable by tuning, the design's fallback is adding `@vladmandic/face-api` for angry and sad only. Decide that deliberately; it is not part of the current build.

### On-camera checklist

Tick these off once tuned. The second face can be a photo or video on a phone.

- [ ] Sleeping, about to sleep, laughing and smiling are correct nearly every try.
- [ ] Angry is correct most of the time when clearly acted out. Crying is best effort.
- [ ] One minute of normal blinking gives no alarm.
- [ ] Nodding forward with eyes closed sets off sleep. Looking down at the controls with eyes open does not.
- [ ] A phone photo as a second face gets its own letter and label, and both letters stay put.
- [ ] Unplugging or blocking the camera shows the message and stops the sound.
- [ ] With Wi-Fi off, the app still starts and works.
- [ ] The browser's Network tab shows no image or video going anywhere.

## Troubleshooting

- "Camera access was blocked." Click the camera icon in Chrome's address bar, allow the camera for localhost, then reload.
- "No camera found." or "The camera is in use by another app." Connect a camera or close the other app (Teams, Zoom and similar), then reload.
- "The face model could not be loaded." The model or WASM files are missing. Run `npm run setup:assets` (needs internet the first time), then reload. Check that `public/mediapipe/face_landmarker.task` and `public/mediapipe/wasm/` exist.
- `setup:assets` says the WASM folder is not found. Run `npm install` first.
- No sound. Browsers only allow audio after a click, so click Start camera. Also check the Mute button and the volume slider.
