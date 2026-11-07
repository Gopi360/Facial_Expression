# Live Facial Expression Demo

A demo that watches the laptop camera and says, live, what each person in view seems to be doing. It runs on your laptop only. Python does the detection and logic. The browser page only handles the camera, the screen and the alarm sound.

**Status: demo. The thresholds are untuned estimates carried over from an earlier version, and accuracy on real faces has not been checked yet.** See "Tuning on camera" below.

## Not for safety, driver or medical use

This is not a safety device, and not a medical tool. Do not use it to watch drivers or patients. A red flag is a guess from a rough rule, not a proven fact. **Every flag needs a human to look at the snap and decide.**

## What it does

- Reads each face for: sleeping, about to sleep, yawning, angry, laughing, smiling, crying, neutral.
- Gives each person a letter (Person A, Person B...). A person who leaves and comes back during one session keeps their letter.
- Shows a thin box and a small letter on each face in the camera view. All text goes in the strip under the camera and in the right panel, never on the video.
- Sleeping and about to sleep turn the box red, show the reason in the log, play an alarm sound and save a snap.
- Writes one log file per session.

### How to read the confidence number

- For angry, laughing, smiling, crying and neutral, the number is the trained model's score for that label.
- For sleeping, about to sleep and yawning, the number is the strength of a hand-written rule. It is **not a probability**.

## Setup (Windows)

You need Python 3.12 and a webcam. Run these in PowerShell from the project folder.

```powershell
python -m venv .venv
.venv\Scripts\activate
.venv\Scripts\python -m pip install -r requirements.txt
python scripts/setup_models.py
.venv\Scripts\python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000> in Chrome or Edge, click **Start camera**, and allow the camera.

The setup script downloads about four files into `models/` and skips any already there. It needs the internet once. A line "Model setup failed" means a download failed: check the connection and run it again.

To run the tests: `.venv\Scripts\python -m pytest -q`. Tests marked `models` need the downloaded model files.

### Offline use

After `setup_models.py` has run once, the app works with Wi-Fi off. The server listens on `127.0.0.1` only, the page loads no outside files, and nothing is sent anywhere else. The server also refuses WebSocket connections from any web page that is not its own.

## Using the page

- **Start camera:** begins a session. The sound is unlocked by this click, because browsers block sound until you click.
- Bottom strip: how many faces are found, and one line per person, such as `Person A is smiling 87%`. A new face shows `Identifying...` until a view facing the camera decides who it is, or for up to 2 seconds if the person stays turned away.
- **Activity log** (right): label changes, people joining and leaving, red marks, snaps, alarm start and stop.
- **Alarm:** Mute, Volume, and "Sound after" (1 to 10 seconds, default 3). The delay sets when "sleeping" starts and when the sound starts, but "about to sleep" and head droop can sound before the slider's time.
- **Snapshot:** saves the current frame by hand.
- The page always shows "Recording logs and snaps".
- If more than 4 faces are in view, only the 4 largest are read and the page says "Showing the 4 largest faces". After Person Z, new people show as "Person ?" and are still logged.
- Only one session runs at a time. A second tab is told "The app is already open in another tab." Reloading the page ends the session and starts a new one.

### Alarm timing

The alarm plays while anyone is sleeping or about to sleep, and stops when nobody is, or when muted. "About to sleep" and head droop can set the alarm off **before** the delay setting has passed. Yawning is not red and makes no sound.

## Logs and snaps hold photos of people

- `logs/session-YYYY-MM-DD_HH-MM-SS.log`: one plain-text file per session, written line by line as things happen.
- `snaps/`: a whole-frame photo, with boxes and letters, each time a person turns red (and when you press Snapshot). A person who stays red gets one snap, not many. A red mark seen for under one second takes no snap: someone red for 0.3 seconds and then lost has the log line but no photo, on purpose, so flicker can't create photos. Someone lost just as they reach one second gets their last visible frame, saved when their letter is released, so that photo can be up to 2 seconds older than the log line.
- Both folders are git-ignored. They stay on this laptop and nothing deletes them: **delete them by hand.** Face signatures used for recognition are kept in memory only and are gone when the app stops.
- **Before any real exam or other real-world use, you need consent from the people filmed and a written rule for how long photos are kept and who deletes them.**

## Known limits

- A head flat on a desk has no visible face, so the app sees the person leave.
- Looking down, glasses, dim light and turned heads reduce accuracy.
- Expression reading is a rough guess from a face, not a measure of feeling. Crying is the weakest.
- Very narrow eyes or a naturally low brow can cause wrong or missed flags. In particular:
  - A person whose resting eye closure is above 0.7 can stay in a false alarm.
  - A naturally low brow can silently suppress the about-to-sleep alarm.
- A face read as happy with an open mouth delays the alarm.
- Recognition can confuse similar faces (relatives, for example). People are only recognised within one session.
- Someone who stays turned away for over 2 seconds gets a new letter. If they then face the camera and are recognised, they take their old letter back, and the log shows the new letter leaving.
- A letter's first three good views are kept for the whole session, so a poor first view stays a weak reference, and a big change in appearance (glasses off, very different light) can cost a person their letter.
- A stranger who sits in the seat of someone who just left can show under that person's letter for about 2 seconds, until two checks facing the camera fail. Then the stranger gets a new letter, and the first person is logged as left.
- Someone returning with their head down and eyes open, before anything is known about their normal head position, can get a new letter.
- Time with no face does not count as closed-eye time, and neither does any frame gap over 0.2 seconds. **If the app runs below about 5 frames a second, the sleep count stops growing and "sleeping" will not show.** Target is 15, 8 is acceptable.
- When the browser tab is hidden, the page stops sending frames, so **detection and the alarm pause**. The alarm is silenced, and a note shows on return.
- If the app takes more than 5 seconds to answer one frame, the session ends with a message.
- The thresholds were not tuned on camera yet, and were only ever aimed at one face (the owner's).

## Models and licenses

Models are downloaded by `scripts/setup_models.py`. Licenses below were looked up from the sources linked on 2026-11-07.

| Use | Model | License found | Source |
|---|---|---|---|
| Eyes, mouth, head tilt | MediaPipe Face Landmarker (`face_landmarker.task`) | **Not confirmed.** Google's page and model cards I could open state no license. Third-party mirrors say Apache 2.0, which I did not accept as proof. | [Google docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) |
| Expressions | HSEmotion `enet_b0_8_best_vgaf` | **Code: Apache 2.0** (repo LICENSE file). **The weights file itself: not confirmed**, the repo does not name a separate license for it. Its training data terms (VGGFace2, AffectNet) were not checked. | [Repo](https://github.com/HSE-asavchenko/face-emotion-recognition), [LICENSE](https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/LICENSE) |
| Recognition | OpenCV SFace (`face_recognition_sface_2021dec`) | **Apache 2.0.** "All files in this directory are licensed under Apache 2.0 License." | [Hugging Face](https://huggingface.co/opencv/face_recognition_sface) |

**Blocker for commercial or exam use:** two of the three models have an unconfirmed license (MediaPipe model file, HSEmotion weights and their training data). Confirm both before any commercial or exam use. None of the three was found to be non-commercial, but "not found" is not "confirmed".

`models/sample_face.png` (a scikit-image sample photo) is downloaded only for the model tests.

## Troubleshooting

| What you see | What to do |
|---|---|
| `Missing model file ... Run: python scripts/setup_models.py` | Run the setup script, then reload. |
| "Model setup failed" | A download failed. Check the internet and run the script again. |
| Camera blocked or no picture | Allow the camera in the browser's address bar, close other apps using it, reload. |
| "The app is already open in another tab." | Close the other tab. If none is open, reload once. |
| "Connection to the app was lost..." | The server stopped. Start it again and reload. |
| "The app stopped responding..." | A frame took over 5 seconds. Reload, and restart the app if it repeats. |
| "The app hit a problem and stopped this session." | Reload. The cause is printed in the terminal running the server. |
| "The alarm sound could not be started." | Reload and click Start camera again. |
| Slow or choppy | Close other apps and tabs. Target is 15 frames a second, 8 is acceptable on this laptop. |
| `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)` | Port 8000 is in use. Add `--port 8001` to the run command and open that port instead. |

## Tuning on camera

All tunable numbers are in `app/core/config.py`. **Capture real numbers before changing any of them.**

Open <http://127.0.0.1:8000/?debug=1>. This shows a raw-numbers panel under the camera. For each person:

- their live signals, and the lowest and highest `eye_closure`, `look_down` and `pitch_delta` over the last 10 seconds;
- which drowsy sign is active, or whether a strong expression is blocking it;
- the closed count in milliseconds, and the closed threshold in force;
- raw `eyeBlink`, pitch and yaw, and their frozen baseline, which reads `learning` until it is set;
- the last recognition check: best score, runner-up and what was decided.

The **Download debug log** button saves the last 90 seconds as a JSON file to your computer. Nothing is sent anywhere. Each frame sample also has the emotion scores. Recognition checks are separate entries with `"kind": "identity"`, holding scores and letters only, never a face signature. Their `outcome` is one of `new letter`, `matched`, `waiting`, `kept`, `refused`, `mismatch`, or `stripped` followed by what happened next.

1. **Check `look_down` on closed eyes first.** Close your eyes, face the camera, and read `look_down` and `eye_closure`. The old TypeScript version never reached "sleeping" on a real face, and the main suspect is that these scores misbehave on closed eyes. If `look_down` goes above `look_down_at` (0.4) while eyes are closed, the rule uses the stricter `eyes_closed_when_looking_down_at` (0.8) and sleep may never fire.
2. Record neutral, blinking for a minute, eyes closed for 5 seconds, a nod forward with eyes closed, looking down at the controls with eyes open, a yawn, a laugh, and a smile.
3. Download the debug log for each, and compare against the values in `app/core/config.py` (for example `eyes_closed_at`, `droopy_eyes_at`, `head_droop_deg`, `yawn_jaw_at`, `smile_at`, `identity_match_at`).
4. **Recognition:** leave and return, doze and wake, and turn away, then read `best_score`, `runner_up_score` and `anchor_score` in the identity entries. The settings are:
   - `identity_match_at` (0.363): the score that counts as the same person.
   - `identity_anchor_signatures` (3): first good views kept for the session, on top of the `identity_max_signatures` (5) rolling views.
   - `identity_mismatch_checks` (2): frontal checks in a row that must fail before a track loses its letter.
   - `frontal_yaw_deg` (20) and `frontal_pitch_delta_deg` (12): how far from facing the camera and from the person's normal head pitch a view can be and still count as good. It must also have eyes open, under `eyes_closed_at`.
5. Change one number at a time and retry.

## Checks to run on camera

Second faces can be a photo or video on a phone. These are demo targets, except recognition on your own face and the alarm rules.

- [ ] Runs from this README in a few commands.
- [ ] Leave for 10 seconds and return: still Person A, every try.
- [ ] Sleeping, about to sleep, yawning, laughing and smiling are correct nearly every try.
- [ ] Angry is correct most of the time when clearly acted out. Crying is best effort.
- [ ] Normal blinking for one minute gives no alarm.
- [ ] Nodding forward with eyes closed sets off sleep. Looking down at the controls with eyes open does not.
- [ ] A person turning red makes exactly one snap, and it appears in that session's log.
- [ ] Two sessions make two log files.
- [ ] With Wi-Fi off, the app still starts and works.
- [ ] Nothing is sent anywhere but `127.0.0.1`.

## Project layout

- `app/core/`: detection rules and logic, with no camera or model code, fully unit tested.
- `app/models.py`: the three model wrappers.
- `app/pipeline.py`, `app/server.py`, `app/session_files.py`: the frame pipeline, the FastAPI server and WebSocket, and the log and snap files.
- `web/`: the page (`index.html`, `app.js`, `styles.css`). No build step.
- `scripts/setup_models.py`: downloads the models.
- `tests/`: pytest tests.
