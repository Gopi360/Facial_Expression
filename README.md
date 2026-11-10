# Live Facial Expression Demo

A demo that watches the laptop camera and says, live, what each person in view seems to be doing. It runs on your laptop only. Python does the detection and logic. The browser page only handles the camera, the screen and the alarm sound.

**Status: demo. Sleep thresholds are tuned from real readings of one person with glasses. Sleep, the alarm, angry, smiling, recognition and far faces have worked on real people, but other faces and rooms may need tuning.** See "Tuning on camera" below.

## Not for safety, driver or medical use

This is not a safety device, and not a medical tool. Do not use it to watch drivers or patients. A red flag is a guess from a rough rule, not a proven fact. **Every flag needs a human to look at the snap and decide.**

## What it does

- Reads each face for: sleeping, about to sleep, yawning, angry, laughing, smiling, crying, neutral.
- Gives each person a letter (Person A, Person B...). A person who leaves and comes back during one session keeps their letter.
- Shows a thin box and a small letter on each face in the camera view. All text goes in the strip under the camera and in the right panel, never on the video.
- Sleeping and about to sleep turn the box red, show the reason in the log, play an alarm sound and save a snap.
- Checks whether each person's ears can be seen. If an ear is hidden for about a second (a hand, hair, a hat), the person's box turns red, a snap is saved, and a second line shows under their expression line: `↳ ears are not visible`. This needs the ear model (see "Ear check" below). Without it the app still runs and shows "Ear check is off: models/ears.onnx was not found." once.
- Finds faces too far away to read, down to about 20 pixels wide at best. They get a box and a letter only: no expression, no red mark, no alarm and no snap.
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

The setup script downloads five files into `models/` and skips any already there. It needs the internet once. A line "Model setup failed" means a download failed: check the connection and run it again.

To run the tests: `.venv\Scripts\python -m pytest -q`. Tests marked `models` need the downloaded model files.

### Offline use

After `setup_models.py` has run once, the app works with Wi-Fi off. The server listens on `127.0.0.1` only, the page loads no outside files, and nothing is sent anywhere else. The server also refuses WebSocket connections from any web page that is not its own.

## Ear check

For each close face, the app cuts out a square around the ear on the side the head is turned away from, or both sides when the face is nearly front-on. A small model (`models/ears.onnx` and `models/ears.json`) says how likely each ear is covered. When that stays at or above `ear_covered_at` for `ear_hidden_hold_ms`, the person is flagged. It clears after `ear_clear_hold_ms` of clear checks. Someone whose ears were not checked for `ear_unchecked_grace_ms` drops the state quietly. Far faces are never checked.

- **The model is not in Git and is not downloaded by `setup_models.py`.** You make it yourself: take photos in photo mode, then train. The steps are in `training/README.md`.
- **Photo mode:** open <http://127.0.0.1:8000/?collect=1>. Buttons (Both clear, Left covered, Right covered, Both covered) choose what the photos show, and the app saves a photo of each ear square, at most once per `collect_interval_ms`, into `dataset/ears/`. These are photos of people: keep `dataset/` private and delete it by hand when done. Be alone in the frame, because every face seen is saved under the chosen label.
- A hidden ear is red like sleeping, but it is a guess from a small model trained on a few photos. A human must look at the snap.

## Using the page

- **Start camera:** begins a session. The sound is unlocked by this click, because browsers block sound until you click.
- Bottom strip: how many faces are found, and one line per person, such as `Person A is smiling 87%`. A new face shows `Identifying...` until a view facing the camera decides who it is, or for up to 2 seconds if the person stays turned away. Someone too far to read shows `Person B is too far to read`, with no percent.
- **Activity log** (right): label changes, people joining and leaving, red marks, snaps, alarm start and stop.
- A hidden ear adds an indented red line under that person's expression line, such as `↳ ears are not visible`. The expression line is red only for sleeping or about to sleep, so the two reasons stay apart. The sound is for sleep only.
- **Alarm:** Mute, Volume, and "Sound after" (1 to 10 seconds, default 3). The delay sets when "sleeping" starts and when the sound starts, but "about to sleep" and head droop can sound before the slider's time.
- **Snapshot:** saves the current frame by hand.
- The page always shows "Recording logs and snaps".
- If more than 4 faces are in view, only the 4 largest are read and the page says "Showing the 4 largest faces". Far faces count towards the 4, and being smallest they are the first left out. After Person Z, new people show as "Person ?" and are still logged.
- `Running slowly, so sleep detection may be late.` shows if the page measures under 6 frames a second for 4 seconds.
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
- A stranger who sits in the seat of someone who just left shows under that person's letter until two checks in a row fail. Checks run about once a second, so that is about 2 seconds **only if the stranger faces the camera the whole time**. A check needs a face turned under 20 degrees, open eyes, and a head tilt under 30 degrees from that person's normal, and it is skipped when heavy eyes come with a tilt past 12 degrees. A view good enough to store fails below a score of 0.363. A tilted view only fails below 0.2, so a stranger who looks a little like that person and stays tilted is not caught. Closing the eyes clears any failed check. A stranger who stays turned, sits with eyes closed or stays far away keeps the letter for as long as that lasts. Then the stranger gets a new letter, and the first person is logged as left.
- Someone returning with their head down and eyes open, before anything is known about their normal head position, can get a new letter. They take their old letter back after they look up, once the checks above have run, which can take about 2 seconds. A head more than 30 degrees from where that new letter learned its normal is not checked until it comes closer.
- Time with no face does not count as closed-eye time, and neither does any gap over 0.5 seconds between frames. Gaps up to that count as watched, so from 4 to 9 frames a second "sleeping" arrives about 3.2 to 3.4 seconds after eyes shut, with the 3 second setting. **If the app runs below about 2 frames a second, every frame is a gap, the sleep count stops growing and "sleeping" will not show.** Target is 15, 8 is acceptable, and the page warns under 6.
- When the browser tab is hidden, the page stops sending frames, so **detection and the alarm pause**. The alarm is silenced, and a note shows on return.
- **The frame rate on the i3 laptop depends on what else is running, by as much as double.** In a real browser, with a canvas standing in for the camera, 4 people close to the camera measured 4.5 frames a second in one session and 9.5 in a quieter one (5.1 and 11.2 with far detection off), and 1 person measured 9.1 and 13.5 (11.9 and 14.4 off). Sent straight to the server with no browser, both ran at about 13. These are test scenes, not a real camera, so close other apps and watch for the slow-running warning.
- If the app takes more than 5 seconds to answer one frame, the session ends with a message.
- The thresholds were not tuned on camera yet, and were only ever aimed at one face (the owner's).

### Far faces

- **Nothing can be read at this size.** No expressions, no sleep, no alarm and no snap, so a far person who falls asleep is not flagged. Someone red who moves far away stops the alarm after about 2 seconds of far views in a row, the same as someone who leaves. A sleeper whose tipped head loses the face detector looks the same at first, which is why it is not sooner.
- **Recognition at a distance is weak.** A far face can match a letter already known, wrongly as well as rightly, but it never stores a view or strips a letter. A new far person gets a new letter after 2 seconds, and takes their old one back if they come close and are recognised.
- **A real room finds less than the test did.** The floor is about 20 pixels wide: faces of 20 were found every time with a sharp, well-lit, front-on photo, and 16 never counted. With noise added, faces of 20 to 34 pixels were found only some of the time. Expect a shorter range with dim light, a turned head, motion or a soft webcam picture.
- **A printed picture is a real face to YuNet.** A far face must be detected 5 times in a row before it shows. YuNet runs on every second frame, so that takes about 9 frames, around a second at 10 frames a second. This stops flicker. It does **not** stop a calendar or photo on the wall: that can show as a far person, get a letter and count as a face.
- A picture uncovered just as someone leaves can inherit their track, and for a while their letter, because a far view can't fail a check. When they come back facing the camera they take the letter back, and the picture gets a new one. If they don't come back it keeps showing under their letter.
- A face over 100 pixels wide is never far. If MediaPipe loses it, it is lost, as it always was.
- Someone who moves between near and far is hidden for a moment while the far view proves itself, then logged as `is too far to read`. A person first seen far is logged as `joined` only.

## Models and licenses

Models are downloaded by `scripts/setup_models.py`. Licenses below were looked up from the sources linked on 2026-11-07.

| Use | Model | License found | Source |
|---|---|---|---|
| Eyes, mouth, head tilt | MediaPipe Face Landmarker (`face_landmarker.task`) | **Not confirmed.** Google's page and model cards I could open state no license. Third-party mirrors say Apache 2.0, which I did not accept as proof. | [Google docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) |
| Expressions | HSEmotion `enet_b0_8_best_vgaf` | **Code: Apache 2.0** (repo LICENSE file). **The weights file itself: not confirmed**, the repo does not name a separate license for it. Its training data terms (VGGFace2, AffectNet) were not checked. | [Repo](https://github.com/HSE-asavchenko/face-emotion-recognition), [LICENSE](https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/LICENSE) |
| Recognition | OpenCV SFace (`face_recognition_sface_2021dec`) | **Apache 2.0.** "All files in this directory are licensed under Apache 2.0 License." | [Hugging Face](https://huggingface.co/opencv/face_recognition_sface) |
| Ears visible | Own MobileNetV3-Small, trained by you from your photos (`models/ears.onnx`) | **Code: torchvision, BSD-3-Clause. The ImageNet starting weights: no license stated** (see `training/README.md`). Your own photos are your own data. | [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE) |
| Far faces | OpenCV YuNet (`face_detection_yunet_2023mar`) | **MIT.** "All files in this directory are licensed under MIT License." The LICENSE file reads MIT, copyright 2020 Shiqi Yu. Its training data terms were not checked. | [Hugging Face](https://huggingface.co/opencv/face_detection_yunet), [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [LICENSE](https://huggingface.co/opencv/face_detection_yunet/blob/main/LICENSE) |

**Blocker for commercial or exam use:** three of the five models have an unconfirmed license (MediaPipe model file, HSEmotion weights and their training data, and the ImageNet weights the ear model starts from). Confirm all three before any commercial or exam use. None was found to be non-commercial, but "not found" is not "confirmed".

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
| Slow or choppy, or "Running slowly, so sleep detection may be late." | Close other apps and tabs. Target is 15 frames a second, 8 is acceptable on this laptop. |
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

1. **Start from these real numbers.** Tuned on the owner's face with glasses: open eyes read `eye_closure` about 0, closed eyes about 0.48, a nod 0.13 with `pitch_delta` about 22, and looking at the keyboard 0.135 with `pitch_delta` about 11. `look_down` sat at about 0.6 even at rest, so it is shown for information only and no rule uses it. A deep look down, past `head_droop_deg` (16), can read as a nod. Half-closed eyes can't be told apart from looking down, so they don't count as about to sleep.
2. Record neutral, blinking for a minute, eyes closed for 5 seconds, a nod forward with eyes closed, looking down at the controls with eyes open, a yawn, a laugh, and a smile.
3. Download the debug log for each, and compare against the values in `app/core/config.py` (for example `eyes_closed_at`, `droopy_eyes_at`, `head_droop_deg`, `yawn_jaw_at`, `smile_at`, `identity_match_at`).
4. **Recognition:** leave and return, doze and wake, and turn away, then read `best_score`, `runner_up_score` and `anchor_score` in the identity entries. The settings are:
   - `identity_match_at` (0.363): the score that counts as the same person.
   - `identity_anchor_signatures` (3): first good views kept for the session, on top of the `identity_max_signatures` (5) rolling views.
   - `identity_mismatch_checks` (2): checks in a row that must fail before a track loses its letter.
   - `frontal_yaw_deg` (20) and `frontal_pitch_delta_deg` (12): how far from facing the camera and from the person's normal head pitch a view can be and still count as good enough to store. It must also have eyes open, under `eyes_closed_at`.
   - `identity_check_pitch_delta_deg` (30): a looser limit for checking a letter, never for storing. The pitch must be under it. The yaw and open-eye limits stay, and heavy eyes (`head_droop_eyes_at` or more) past `frontal_pitch_delta_deg` are not a check, so a person dozing keeps their letter.
   - `identity_check_fail_at` (0.2): a check on a view too tilted to store fails only under this score. Frontal checks use `identity_match_at`.
5. **Far faces:** the settings are:
   - `far_score_at` (0.7): how sure YuNet must be. Raise it if pictures on a wall show as people.
   - `far_min_width_px` (16) and `far_max_width_px` (100): the widths that count as far, in pixels of the frame sent to the server. YuNet's own floor sits above the minimum, at about 20. Keep the top well above where MediaPipe stops finding faces, which was between 60 and 70.
   - `far_confirm_frames` (5): real YuNet detections in a row before a far face shows at all. Frames that reuse its last result don't count.
   - `far_every_frames` (2): how often YuNet runs. 1 is every frame, at a real cost in frame rate.
   - `far_same_face_overlap` (0.5): how much of the smaller box must overlap for a YuNet face and a MediaPipe face to be one face, if YuNet's centre is not already inside the MediaPipe box.
6. **Slow machines:** `frame_gap_limit_ms` (500) is the longest time between frames that still counts as watched. `low_fps_warning` (6) and `low_fps_hold_ms` (4000) set the page's slow-running warning.
7. **Ears:** the settings are:
   - `ear_both_yaw_deg` (20): under this head turn both ears are checked, over it only the ear turned away.
   - `ear_yaw_sign` (-1): which ear is "turned away" for a given turn. Measured on camera: +1 checked the wrong ear. Change only if you flip the camera.
   - `ear_offset_frac` (0.10) and `ear_square_frac` (0.45): where the ear square sits and how big it is, as a fraction of the face width. `ear_square_frac` was set from photos on camera. Training photos and the app must use the same values, so collect again after changing them.
   - `ear_input_px` (96): the model's input size. Training and the app must agree.
   - `ear_interval_ms` (333): how often the ear model runs for each person.
   - `ear_covered_at` (0.8): the chance at or above which a side counts as covered. A cautious start, still to be tuned on camera.
   - `ear_hidden_hold_ms` (1000), `ear_clear_hold_ms` (500) and `ear_unchecked_grace_ms` (2000): how long covered checks must last before red, how long clear checks must last before it clears, and how long without a check before the state is dropped.
   - `collect_interval_ms` (300): photo mode saves at most once per this.
8. Change one number at a time and retry.

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
- [ ] Walk away from the camera and back: the letter stays, the line changes to `is too far to read` and back.
- [ ] The wall behind you shows no far person. If the calendar does, raise `far_score_at`.
- [ ] Hand over one ear turns the box red with the ear line within about 2 seconds, and taking it away clears it.
- [ ] Sitting normally and turning the head give no ear flag.
- [ ] With a person far away, the page still runs at 8 frames a second or more.

## Project layout

- `app/core/`: detection rules and logic, with no camera or model code, fully unit tested.
- `app/models.py`: the model wrappers.
- `app/pipeline.py`, `app/server.py`, `app/session_files.py`: the frame pipeline, the FastAPI server and WebSocket, and the log and snap files.
- `web/`: the page (`index.html`, `app.js`, `styles.css`). No build step.
- `scripts/setup_models.py`: downloads the models.
- `training/`: the ear model's training script and its own README. `dataset/` (your photos) is git-ignored.
- `tests/`: pytest tests.
