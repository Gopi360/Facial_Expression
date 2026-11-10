# Project Handover

Last updated: 2026-11-10 (Stage 2a working on camera, not merged). Read this first when starting a new chat.

## Where things stand

A local Python demo that reads the laptop camera and shows each person's expression, sounds an alarm for sleep, and keeps a log and snaps of red marks. Stage 1 is built and working on real people, and is merged into `main` (merge commit `be7b9fb`, 2026-11-10). Stage 2a (ear visibility) is built, trained and working on the owner on camera, on `feat/stage-2a-ears`. It is **not merged** yet.

- **`main`:** Stage 1, merged locally and pushed.
- **Work branch:** `feat/stage-2a-ears`, for Stage 2a.
- **`feat/python-rebuild`:** the Stage 1 work branch, now merged.
- **GitHub:** https://github.com/Gopi360/Facial_Expression (public). All branches are pushed, including `feat/stage-2a-ears`.
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
| `docs/superpowers/plans/2026-11-10-stage-2a-ear-visibility.md` | Stage 2a build plan, all 6 tasks done |
| `training/README.md` | How to set up `.venv-train`, collect photos, train and read the report |
| `docs/superpowers/stage-2a-ledger.md` | Stage 2a rulings, fix rounds and deferred minor issues, by task |
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

Done and working on the owner (glasses on), checked on camera 2026-11-10:

- Photo mode `?collect=1` saves ear squares to `dataset/ears/clear` and `dataset/ears/covered` (1013 and 459 photos now, not in Git).
- Training on the laptop takes about 25 s: `.venv-train\Scripts\python training\train.py --data dataset\ears --out models\ears.onnx`. Held-out runs: clear 98.3%, covered 100% (small covered test set). Restart the app after training.
- On camera the owner reports: sitting normally and head turns are not flagged; a hand over an ear gives the red `↳ ears are not visible` line, silent, with a snap and a log line, and it clears when the hand moves away; sleep still sounds. Per-check timings and the frame rate were not written down.
- Measured on camera while building: ear square `ear_offset_frac` 0.10 and `ear_square_frac` 0.45 (squares about 64-74 px), and `ear_yaw_sign` -1. Nothing else was tuned.

Known Stage 2a limits: trained on the owner only; some squares land on the eye and glasses when the head is turned far or the side of the face is covered; EarPods are not detected yet. The full record is in `docs/superpowers/stage-2a-ledger.md`.

## Pending work, in order

0. **Stage 2a: finish and merge.** Run the final whole-branch review (subagent-driven process, most capable model, pointed at the deferred items in `docs/superpowers/stage-2a-ledger.md`), then merge `feat/stage-2a-ears` into `main`. Optional before that: write down the frame rate with the ear check on, and add `test_owner_ears_*` tests if any number gets tuned.
1. **Finish the three untested checks above** when a second person and a wall picture are available. Deferred on 2026-11-10, not a blocker.
2. **Stage 2b: EarPods.** Next after 2a merges. Add an `earpod` photo label (photo mode buttons and folder), take EarPods photos with the same photo mode, retrain with the same script, and add the `↳ device in the ears` line (red, snap, log, silent). Start with brainstorming. Notes in the Stage 2a design, section 11.
3. **Small known issues**, listed below.
4. **Names (undecided).** The app shows Person A and Person B and learns faces live. The owner raised few-shot registration from photos, as in their earlier project `E:\MCA_MAJOR_PROJECT\Smart-Exam-Hall-Attendance-System`, but did not say whether they want real names. Ask.

## Stage 2: ears and devices

The owner's rule, for each person: are the ears visible? If not, say "ears are not visible". If they are, look for a headset, headphones, earbuds or earpods, and say "device found in the ears". Both are red, with a snap. Red is shared with sleeping.

- It starts as a demo feature. Exam use may come later.
- A person whose ears are covered stays red, and a human judges from the snap.
- Near-term target: it only has to work on the owner.
- The owner has **EarPods** to test with. Training will be on the laptop, with Colab as the fallback.

How it was built (2a), and how 2b should follow: MediaPipe gives where each ear should be, a square is cut around it, and a small classifier (MobileNetV3-Small, fine-tuned on the owner's photos on the laptop) labels it. A classifier, not a detector, so no boxes need drawing. Colab is the fallback only if laptop training fails.

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
- `app/models.py`: MediaPipe (eyes, mouth, head tilt, ear positions), HSEmotion (expressions), SFace (recognition), YuNet (far faces), and the ear model (`models/ears.onnx`, optional).
- `app/collect.py`: photo mode. `training/`: the ear model training script, run in its own `.venv-train` with PyTorch. The app itself never uses PyTorch.
- `app/core/`: pure rules with no camera or model imports. All tunable numbers are in `app/core/config.py`.

## Lessons that matter

- **Sleep thresholds came from real readings** of the owner with glasses: open 0, closed 0.48, nod 0.13 at +22 degrees, keyboard 0.135 at +11 degrees. Tests named `test_owner_*` pin these. `look_down` sits near 0.6 at rest, so no rule uses it.
- **Half-closed eyes don't trigger "about to sleep"**, since behind glasses they look like looking at the keyboard. A nod shows "about to sleep", not "sleeping", because a tipped head reads as barely closed.
- **Rejected:** MTCNN and FaceNet. In testing, MTCNN took 77 ms against YuNet's 19 ms, and FaceNet 197 ms against SFace's 20 ms, and `facenet-pytorch` needs an old PyTorch. PyTorch is deliberately not used.
- **Speed varies:** one close face about 13.5 frames a second. Four close faces 4.5 to 9.5, cause unknown.
- Reviews caught real bugs in nearly every task, so keep the review step on anything touching the alarm or letters.
- The laptop has shut down mid-work before. Commit often.
- **Look at real photos before training.** The first ear squares cut the ear off, the second overshot, and head turns picked the wrong ear. All three showed up only in saved photos, not in tests.

## Known small issues

- If a red person walks away while far detection flickers, the alarm can run well past 2 seconds. Mute works. A fix option is in the ledger.
- Eyes closed about 2.5 seconds plus a very brief face loss can flash "sleeping" for about half a second.
- A letter can still stick after a head-down return if the upright match score is middling (0.2 to 0.363).
- A wall picture can appear as a far person, and keeps a letter if the real person never returns.
- A person drifting asleep inside the frontal zone with a low score can lose a letter. Real scores are unmeasured.
- Real `anchor_score` at head pitch 15 to 30 hasn't been measured on camera.
- More deferred items are in the ledger, under lines marked `minor (deferred)`. Stage 2a's are in `docs/superpowers/stage-2a-ledger.md`, with its rulings and fix rounds.
