# Stage 2a: Ear Visibility, Design

Date: 2026-11-10. Branch: `feat/stage-2a-ears`, from `main` after the Stage 1 merge (`be7b9fb`).

Marks: **owner** = said by the owner. **Author's call** = decided by the author, open to the owner's correction.

## 1. Intent

Stage 2 adds an ear check to the demo: for each person, are the ears visible, and is a device in them. The owner split it into two steps (owner):

- **Stage 2a (this design):** ears visible or not. Includes the photo mode and the training setup.
- **Stage 2b (later, own design):** EarPods detection, built on 2a's photo mode and training setup.

2b starts only after 2a works on camera (owner). It is a demo feature. Exam use may come later but is not a goal now (owner).

## 2. Goals

1. For each near person with a letter, show a second line under their expression line when their ears are not visible: `↳ ears are not visible` (owner chose layout C: two lines per person).
2. That person turns red, box included, and stays red until the ears are visible again, the same as sleeping (owner).
3. **No sound** for this. The alarm stays for sleep only (owner).
4. A snap and a log line, as for other red marks (owner).
5. Seen within **a second or two** of an ear being covered, and cleared about as fast when it is uncovered (owner: "showing it within a second or two is enough").
6. Works on the owner. Other people are not a target yet (handover, owner).
7. The owner can train the model on this laptop, with steps simple enough to repeat for 2b.

## 3. Non-goals

- EarPods or any device detection (Stage 2b).
- Working on people other than the owner.
- Exam use, consent flows, a photo deletion rule (needed before exam use; see section 11).
- Real names (still an open question from the handover, separate from this).
- Google Colab. It is the fallback only if laptop training fails or is too slow (owner).
- Any change to the sleep rules or their thresholds.

## 4. Anti-goals

The build fails, even if it technically works, if:

- **Sitting normally, facing the camera, gets flagged.** From the front ears are always only partly visible, and that must count as visible (owner).
- **Turning the head gets flagged** because the far ear is hidden (owner agreed to section 6.2).
- The glasses arm passing the ear makes it read as covered. The owner wears glasses.
- It noticeably slows the app, so sleep detection gets later. Sleep stays the main feature.
- Photos of the owner leave the laptop, or are committed to Git.
- PyTorch becomes a dependency of the app. It is allowed only in a separate training setup.

## 5. Decided by the owner

- Two-step split: ears first, EarPods after (2a, 2b).
- Ears not visible: red, stays red until visible, snap, log line, **no sound**.
- Later, for 2b: "device in the ears" also red, snap, log line, no sound. The owner has **EarPods** to test with.
- Layout C: expression line, then a second line for the ear status, red.
- Front-on partial visibility counts as visible.
- When the head is turned, check only the ear facing the camera (section 6.2).
- Use MediaPipe to locate the ears.
- Train on this laptop. Fall back to Colab only if that fails.
- Success: ears covered shows within a second or two.

## 6. Behaviour

### 6.1 States per person

| Ear state | Second line | Red | Sound | Snap and log |
|---|---|---|---|---|
| Visible (fully or partly) | none | no (unless sleeping) | no | log only when it changes back: `ears visible again` (author's call) |
| Not visible | `↳ ears are not visible` | yes | no | log `ears are not visible`, snap |
| Not checked (far face, no letter yet, ear out of frame, turned away) | none | keeps the last ear state for a short grace (start about 2 s, like `letter_release_ms`), then drops it quietly with no `ears visible again` line (author's call) | no | none |

A person can be sleeping and have ears not visible at once. Both lines show. The sound follows sleep only.

### 6.2 Which ears are checked (owner agreed)

- Head roughly straight (turn under a set angle, start from the existing `frontal_yaw_deg`, 20 degrees): check **both** ears.
- Head turned further: check only the **ear facing the camera**. The far ear is ignored.
- If the ear's expected spot is outside the frame, that ear is not checked.
- **Author's call:** if any checked ear is covered, the person is "ears not visible". So one covered ear is enough when facing front. Hair over one side therefore turns a person red. This matches "a person whose ears are covered stays red, and a human judges from the snap" (handover).

### 6.3 Timing (author's call, numbers tuned on camera)

- Ear checks run on a fixed interval, like the emotion check (`emotion_interval_ms` 333), not every frame.
- "Not visible" must hold for a short time (start about 1 s) before the person turns red, so one bad check or a hand passing the face doesn't flag. "Visible" must hold a shorter time (start about 0.5 s) to clear.
- Both hold times plus the check interval must keep the total within the owner's "a second or two".
- The snap follows the existing rule (`min_red_ms`, seen red on camera for 1 s).

### 6.4 Red, snaps and the log (author's call)

- Red today comes only from the expression label (`events.is_red`). It must become: red if sleep-red **or** ear-red. The alarm (`should_sound`) must still use sleep only.
- **Each red reason earns its own snap.** A person already red for sleeping who then covers an ear gets a second snap. Today `SnapPolicy` keys by letter, so this needs a key per letter and reason.
- Log lines: `Person A: ears are not visible` when it starts, `Person A: ears visible again` when it clears.
- The person dict sent to the page must carry the ear line separately, so the page can draw layout C. The overlay box turns red for either reason.

## 7. Technical approach (author's call)

### 7.1 Finding the ears

MediaPipe Face Landmarker already runs on every near face. Its 478 points include the face edge next to each ear (for example points near 234 and 454 on the face outline; the builder confirms which). From those, plus face size and head turn, cut a square around where each ear should be. `RawFace.points` keeps only 5 points today, so the ear anchor points must be added.

MediaPipe tells us **where** the ear should be. It can't tell **what is there**. That needs a small model.

### 7.2 The ear model

- An **image classifier** on each ear square, not a detector. Labels for 2a: `clear` and `covered`. 2b adds `earpod`.
- Built by **fine-tuning** a small ready-made image model (for example MobileNetV3-Small) on the owner's photos.
- Exported to **ONNX** and run with `onnxruntime`, already in `requirements.txt`. No new runtime tool for the app.
- The cut-off for "covered" lives in `app/core/config.py`. Start on the cautious side, so it only says "not visible" when clearly covered. Tune on camera with the owner, the same way the sleep numbers were set.
- If the model file is missing, the app still runs without the ear check and says so once (like other optional pieces). Sleep and expressions are unaffected.

### 7.3 Photo mode

- A hidden page mode, like `?debug=1` (for example `?collect=1`), so it never shows in a normal demo.
- **Labels are per ear, not per run (author's call).** Covering is usually one-sided, so the owner picks one of: `both clear`, `left covered`, `right covered`, `both covered`. Left and right are as seen on the screen, which is mirrored like the video, and the page says so beside the buttons. The app saves each ear square into its own label folder, so in `left covered` the left ear goes to `covered/` and the right ear to `clear/`.
- Only ears the live check would look at are saved (section 6.2). When the head is turned, the far ear is not saved.
- Photos from one Start-to-Stop run share a run id in the file name, so training can split by run (section 7.4).
- The owner presses Start, moves their head and light around, and presses Stop. While running, the app saves the ear squares a few times a second into `dataset/ears/<label>/`.
- The page shows how many photos each label has.
- `dataset/` is git-ignored.
- **Where `clear` ends and `covered` begins.** The owner's hair does not normally reach their ears (owner), so their usual look is always `clear`. An ear only partly seen because of the head angle is `clear`. An ear with most of it hidden by something in front of it (hand, hair pulled over, hood) is `covered` (author's call). Borderline photos, where the ear is about half hidden, are left out of training rather than guessed.
- **What to photograph, for `covered`:** hair over the ear, hand over the ear, hood or scarf if available.
- **What to photograph, for `clear`:** ears visible, with glasses on, front-on and turned, different light. These are the cases that must not flag.

### 7.4 Training

- A `training/` folder with a script and a short how-to, committed.
- A separate training setup (its own virtual environment, for example `.venv-train`, git-ignored) with CPU PyTorch and torchvision. The app's `.venv` stays without PyTorch.
- The script:
  1. Splits the photos, keeping about 1 in 5 aside for testing.
  2. Fine-tunes the model.
  3. Prints accuracy and which mistakes it made on the held-out photos.
  4. Writes `models/ears.onnx`.
- Photos taken in one burst look alike. Split by burst or session, not by single photo, or the test score will look better than it is.
- Expected time on this laptop's CPU: minutes. This is an estimate, not measured. If it is far slower or fails, move to Colab (owner).

### 7.5 First check: are the ears big enough? (author's call)

Frames are 640x480 (`frame_width`, `frame_height`). At normal sitting distance a front-on ear may be only about 20 to 40 pixels wide. **Before any training work**, the first build step saves a few ear squares of the owner, clear and covered, at today's size and looks at them.

- If a person can tell clear from covered by eye, carry on at 640x480.
- If not, stop and bring options back to the owner. One option is a larger frame, but that costs frame rate and sleep detection, so it is not chosen without them.

### 7.6 Speed

- Budget: the ear check must not drop the frame rate noticeably. One face runs about 13.5 frames a second today; sleep detection stops working under about 2.
- Measure the per-crop time on this laptop before wiring it into the app. Target is under about 10 ms per crop.

## 8. Code layout (author's call, the builder may adjust)

- `app/core/ears.py`: pure rules. Which ears to check from the head turn, the hold times, the state per person. No cv2, mediapipe, onnxruntime or fastapi (project rule).
- `app/core/config.py`: all new numbers.
- `app/models.py`: the ear crop and the ONNX classifier.
- `app/pipeline.py`: calls the ear check, joins its result into red, snaps and log.
- `app/core/text.py`, `events.py`, `snap_policy.py`: new wording, red from two sources, a snap per reason.
- `web/`: the second line (layout C) and the photo mode.
- `training/`: script, `requirements-train.txt`, how-to.

Tests follow the project's style: rules tested in `app/core/` without models, and the model wrapper tested with a fake or a sample image.

## 9. How we'll know it worked

On the owner, on the real camera, with glasses on:

1. Sitting normally, facing the camera, for a few minutes: never flagged.
2. Head turned left and right, and looking down at the keyboard: never flagged.
3. Hand over one ear: red `↳ ears are not visible` within a second or two, one snap, one log line, no sound.
4. Hand removed: the line clears within a second or two, log `ears visible again`.
5. Hair pulled over an ear, if possible: same as 3.
6. Sleeping still works and still sounds, and sounds only for sleep.
7. Frame rate with one face stays close to today's.
8. The training script reports its held-out accuracy and mistakes.

The full test suite still passes.

## 10. Known limits (expected)

- Trained on one person, so other people, hair types, skin tones and head coverings are untested and may flag wrongly.
- A low or dark camera may leave ears too small or dim to judge.
- During 2a, EarPods are not detected. An ear with an EarPod may read as covered or as clear.

## 11. Stage 2b notes (not in this build)

- Add an `earpod` label and photos with EarPods in, then retrain with the same script.
- Second line `↳ device in the ears`, red, snap, log, no sound (owner).
- Order per ear: covered → "ears are not visible"; EarPod → "device in the ears".
- Before any real exam use: consent, a deletion rule for photos, a fairness check for long hair and head coverings, and confirmed licenses for every model (MediaPipe and HSEmotion are still unconfirmed, see README).

## 12. Left to the builder

- Exact MediaPipe points and crop size for the ears.
- Which small base model, after checking its license and the license of its ready-made weights.
- Input size, training settings, how many photos are enough. Start with about 100 to 200 per label.
- Exact hold times, check interval and cut-off, tuned on camera.
- Photo mode controls and styling, within the hidden-mode rule above.
