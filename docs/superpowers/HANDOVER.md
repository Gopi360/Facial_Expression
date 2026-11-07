# Project Handover

Last updated: 2026-11-07. Read this first when starting a new chat.

## Where things stand

A local Python demo that reads the laptop camera and shows each person's expression, sounds an alarm for sleep, and keeps a log and snaps of red marks. Stage 1 is built and working on real people. **Nothing is merged yet.**

- **Work branch:** `feat/python-rebuild` (all current code).
- **`main`:** holds only an empty first commit. There is no GitHub remote yet.
- **`feat/expression-demo`:** the old TypeScript version, kept for reference.
- **Tests:** 451 passing. Run `.venv\Scripts\python -m pytest -q`.
- **Start the app:** `.venv\Scripts\python -m uvicorn app.server:app --host 127.0.0.1 --port 8000`, then open `http://127.0.0.1:8000/`. Add `?debug=1` for raw numbers.
- **Fresh machine:** follow the setup steps in `README.md`. `models/`, `logs/`, `snaps/` and `.venv/` are not in Git.

## Documents

| File | What it is |
|---|---|
| `docs/superpowers/specs/2026-11-06-python-rebuild-design.md` | Approved design for Stage 1, plus Stage 2 notes in section 12 |
| `docs/superpowers/plans/2026-11-06-python-rebuild.md` | Stage 1 build plan, all 8 tasks done |
| `docs/superpowers/python-rebuild-ledger.md` | Full record of task reviews, rulings and deferred minor issues |
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

## Pending work, in order

1. **Finish the three untested checks above**, then decide on the merge. Options: merge to `main` locally, or create a GitHub repo and open a pull request. No remote exists, so a pull request needs one first.
2. **Stage 2: ear and device detection.** Not built. See below.
3. **Small known issues**, listed below.
4. **Names (undecided).** The app shows Person A and Person B and learns faces live. The owner raised few-shot registration from photos, as in their earlier project `E:\MCA_MAJOR_PROJECT\Smart-Exam-Hall-Attendance-System`, but did not say whether they want real names. Ask.

## Stage 2: ears and devices

The owner's rule, for each person: are the ears visible? If not, say "ears are not visible". If they are, look for a headset, headphones, earbuds or earpods, and say "device found in the ears". Both are red, with a snap. Red is shared with sleeping.

- It starts as a demo feature. Exam use may come later.
- A person whose ears are covered stays red, and a human judges from the snap.
- Near-term target: it only has to work on the owner.
- **Open question for the owner:** which devices do they have to test with?

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
- More deferred items are in the ledger, under lines marked `minor (deferred)`.
