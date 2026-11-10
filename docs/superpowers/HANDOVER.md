# Project Handover

Last updated: 2026-11-10 (Stage 2a built). Read this first when starting a new chat.

## Where things stand

A local Python demo that reads the laptop camera and shows each person's expression, sounds an alarm for sleep, and keeps a log and snaps of red marks. Stage 1 is built and working on real people, and is merged into `main` (merge commit `be7b9fb`, 2026-11-10). Stage 2a (ear visibility) is built on `feat/stage-2a-ears`, awaiting the owner's photos, training and on-camera checks.

- **`main`:** Stage 1, merged locally and pushed.
- **Work branch:** `feat/stage-2a-ears`, for Stage 2a.
- **`feat/python-rebuild`:** the Stage 1 work branch, now merged.
- **GitHub:** https://github.com/Gopi360/Facial_Expression (public). All three branches are pushed.
- **`feat/expression-demo`:** the old TypeScript version, kept for reference.
- **Tests:** 511 passing, 1 skipped. Run `.venv\Scripts\python -m pytest -q`.
- **Start the app:** `.venv\Scripts\python -m uvicorn app.server:app --host 127.0.0.1 --port 8000`, then open `http://127.0.0.1:8000/`. Add `?debug=1` for raw numbers.
- **Fresh machine:** follow the setup steps in `README.md`. `models/`, `logs/`, `snaps/` and `.venv/` are not in Git.

## Documents

| File | What it is |
|---|---|
| `docs/superpowers/specs/2026-11-06-python-rebuild-design.md` | Approved design for Stage 1, plus Stage 2 notes in section 12 |
| `docs/superpowers/plans/2026-11-06-python-rebuild.md` | Stage 1 build plan, all 8 tasks done |
| `docs/superpowers/python-rebuild-ledger.md` | Full record of task reviews, rulings and deferred minor issues |
| `docs/superpowers/specs/2026-11-10-stage-2a-ear-visibility-design.md` | Design for Stage 2a (ear visibility), section 9 lists the on-camera checks |
| `docs/superpowers/plans/2026-11-10-stage-2a-ear-visibility.md` | Stage 2a build plan, Tasks 1 to 6 built |
| `training/README.md` | How to set up `.venv-train`, collect photos, train and read the report |
| `docs/superpowers/python-rebuild-ledger.md` | Also holds the Stage 2a deferred-items pointer, at the end |
| `docs/superpowers/specs/2026-11-06-facial-expression-demo-design.md` | Old TypeScript design, replaced |
| `docs/superpowers/plans/2026-11-06-facial-expression-demo.md` | Old TypeScript plan, replaced |

## What the owner tested on a real camera

| Check | Result |
|---|---|
| Angry and smiling | Working |
| Recognition of the owner | Working |
| Sleep and alarm sound, on the owner (with glasses) and a colleague | Working |
| Nod, keyboard look and normal blinking, after the far-face change | Working |
| Far face gets a box and a letter | Working |
| Walk away and back keeps the same letter | **Not tested yet** |
| Wall calendar picked up as a far person | **Not tested yet** |
| Two people close: "running slowly" warning | **Not tested yet** |

## Stage 2a status

Built: photo mode (`?collect=1`), the training script, the ear model wrapper, the ear state with hold times, the red flag with snap and log line, and the page's second line. **Not done: the owner's photos, training, and the 8 on-camera checks (spec section 9).** Until `models/ears.onnx` and `models/ears.json` exist, the app runs with the ear check off and says so once.

On-camera results: **not recorded yet.** Fill these in when the checks are run: sitting normally, head turns and keyboard, hand over one ear, hand removed, hair over an ear, sleep still sounds, frame rate (time per ear crop and one-face frame rate), and the training report. Then tune `ear_covered_at`, the hold times, `ear_offset_frac`, `ear_square_frac` and `ear_yaw_sign` one at a time, and pin the numbers with `test_owner_ears_*` tests.

## Pending work, in order

0. **Stage 2a: photos, training, on-camera checks.** Follow `training/README.md`, then the checks above.
1. **Finish the three untested checks above** when a second person and a wall picture are available. Deferred on 2026-11-10, not a blocker.
2. **Stage 2: ear and device detection.** Split by the owner: 2a ears visible or not (design: `docs/superpowers/specs/2026-11-10-stage-2a-ear-visibility-design.md`), then 2b EarPods. See below.
3. **Small known issues**, listed below.
4. **Names (undecided).** The app shows Person A and Person B and learns faces live. The owner raised few-shot registration from photos, as in their earlier project `E:\MCA_MAJOR_PROJECT\Smart-Exam-Hall-Attendance-System`, but did not say whether they want real names. Ask.

## Stage 2: ears and devices

The owner's rule, for each person: are the ears visible? If not, say "ears are not visible". If they are, look for a headset, headphones, earbuds or earpods, and say "device found in the ears". Both are red, with a snap. Red is shared with sleeping.

- It starts as a demo feature. Exam use may come later.
- A person whose ears are covered stays red, and a human judges from the snap.
- Near-term target: it only has to work on the owner.
- The owner has **EarPods** to test with. Training will be on the laptop, with Colab as the fallback.

Planned approach, not yet designed in detail. Run brainstorming first:
1. Add a photo-capture mode to the app.
2. The owner takes about 50 photos per class: no device, and each device, with varied head turns and light.
3. Train a detector on free Google Colab, because the laptop is too slow. The owner needs a Google account. A notebook would be prepared for them.

Facts already found:
- No ready-made earbud model was found. Public headphone datasets exist on Roboflow, mostly CC BY 4.0, quality unchecked.
- Ear visibility has no known ready-made model. A dedicated search has not been done.
- Ultralytics YOLO is AGPL-3.0, which matters for any commercial use. YOLOX and NanoDet timed at 220 ms and 22 ms per run on this laptop. Confirm their licenses before use.
- Earbuds will be the least reliable, and ears are often barely visible from the front.
- Before real exam use: consent, a photo deletion rule, and fairness for long hair and head coverings. HSEmotion weights and the MediaPipe model license are also unconfirmed, which blocks commercial or exam use until checked.

## How it works

- `web/`: thin browser page for camera, screen and sound. Sends JPEG frames over one WebSocket.
- `app/server.py`: FastAPI on `127.0.0.1`, one session at a time, refuses other websites.
- `app/pipeline.py`: one frame in, one result out.
- `app/models.py`: MediaPipe (eyes, mouth, head tilt), HSEmotion (expressions), SFace (recognition), YuNet (far faces).
- `app/core/`: pure rules with no camera or model imports. All tunable numbers are in `app/core/config.py`.

## Lessons that matter

- **Sleep thresholds came from real readings** of the owner with glasses: open 0, closed 0.48, nod 0.13 at +22 degrees, keyboard 0.135 at +11 degrees. Tests named `test_owner_*` pin these. `look_down` sits near 0.6 at rest, so no rule uses it.
- **Half-closed eyes don't trigger "about to sleep"**, since behind glasses they look like looking at the keyboard. A nod shows "about to sleep", not "sleeping", because a tipped head reads as barely closed.
- **Rejected:** MTCNN and FaceNet. In testing, MTCNN took 77 ms against YuNet's 19 ms, and FaceNet 197 ms against SFace's 20 ms, and `facenet-pytorch` needs an old PyTorch. PyTorch is deliberately not used.
- **Speed varies:** one close face about 13.5 frames a second. Four close faces 4.5 to 9.5, cause unknown.
- Reviews caught real bugs in nearly every task, so keep the review step on anything touching the alarm or letters.
- The laptop has shut down mid-work before. Commit often.

## Known small issues

- If a red person walks away while far detection flickers, the alarm can run well past 2 seconds. Mute works. A fix option is in the ledger.
- Eyes closed about 2.5 seconds plus a very brief face loss can flash "sleeping" for about half a second.
- A letter can still stick after a head-down return if the upright match score is middling (0.2 to 0.363).
- A wall picture can appear as a far person, and keeps a letter if the real person never returns.
- A person drifting asleep inside the frontal zone with a low score can lose a letter. Real scores are unmeasured.
- Real `anchor_score` at head pitch 15 to 30 hasn't been measured on camera.
- More deferred items are in the ledger, under lines marked `minor (deferred)`. Stage 2a's are in `.superpowers/sdd/2026-11-10-stage-2a-ear-visibility/progress.md` (a git-ignored working file), under the same marker, and the ledger points there.
