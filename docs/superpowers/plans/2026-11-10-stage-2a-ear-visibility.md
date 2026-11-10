# Stage 2a: Ear Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flag each near person whose ears are not visible: a red second line `↳ ears are not visible`, red box, snap and log line, no sound. The model is trained on the owner's own photos on this laptop.

**Architecture:**
- MediaPipe's face outline points next to each ear give where the ear should be. A pure rule in `app/core/ears.py` picks which ears to check from the head turn and cuts a square around each one.
- A small ONNX classifier (`models/ears.onnx`), fine-tuned from a ready-made image model in a separate PyTorch setup, says `clear` or `covered` for each square.
- `EarWatch` in `app/core/ears.py` turns those checks into a steady per-person state, using hold times and a grace period.
- The pipeline joins that state into red, snaps and the log, alongside the existing sleep rules.

**Tech Stack:**
- App: Python, FastAPI, MediaPipe, onnxruntime, OpenCV, vanilla JS page.
- Training only: PyTorch and torchvision (CPU), in `.venv-train`.

**Spec:** `docs/superpowers/specs/2026-11-10-stage-2a-ear-visibility-design.md`

## Global Constraints

- `app/core/` must not import cv2, mediapipe, onnxruntime or fastapi.
- Every new tunable number goes in `app/core/config.py`.
- Do not change any existing sleep or expression threshold.
- No network calls at runtime. The training script may download base weights once.
- PyTorch never enters `requirements.txt` or the app's `.venv`. It is only in `training/requirements-train.txt`, installed into `.venv-train`.
- `dataset/`, `.venv-train/` and `models/` stay out of Git.
- Ears not visible: red, **no sound**, snap, log line. The alarm (`should_sound`) stays sleep-only.
- Panel copy, exact: second line `↳ ears are not visible`; log `ears are not visible`, then `ears visible again`.
- Head turned under `ear_both_yaw_deg` (20): check both ears. Otherwise check only the ear facing the camera. If any checked ear is covered, the person counts as covered.
- A missing `models/ears.onnx` must not stop the app. The ear check is off, and the app says so once.
- Run tests with `.venv\Scripts\python -m pytest -q`. The whole suite must pass at the end of every task.
- Conventional commits, one per task unless a step says otherwise. End each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A hand passing over an ear for under a second** (scratching, adjusting glasses) must not flag. Pinned in Task 4 (`test_brief_cover_does_not_hide`).
2. **A head turn hovering around 20 degrees**, flipping between both ears and one ear, must not reset an ongoing covered run or a hidden state. Pinned in Task 4 (`test_side_switch_keeps_state`).
3. **An ear square partly outside the frame**, with the person at the edge or very close, is skipped, not cropped short or crashed. Pinned in Task 1 (`test_square_outside_frame_is_none`).
4. **`models/ears.onnx` missing or corrupt:** the app runs, sleep still works, and one warning is shown. Pinned in Task 6 (`test_missing_ear_model_runs_without_ears`).
5. **An ear-red person going far or leaving:** the ear warning drops quietly after the grace period, and their owed ear snap is still saved. Pinned in Task 5 (`test_far_drops_ear_state_quietly`, `test_leaving_saves_owed_ear_snap`).

---

### Task 1: Ear anchors and ear squares

**Files:**
- Modify: `app/core/types.py`
  - Add `Side`.
  - Add `ears` to `RawFace`.
  - Add `ears_hidden` to `Person`.
- Modify: `app/core/config.py`: add the ear geometry numbers.
- Create: `app/core/ears.py`: `sides_to_check`, `ear_square`.
- Modify: `app/models.py` (`FaceFinder.find`): fill `RawFace.ears`.
- Test: `tests/test_ears.py`, `tests/test_models.py`.

**Interfaces:**
- Produces:
  - `Side = Literal["left", "right"]`. This is the person's own ear. On the mirrored page it is the same side of the screen.
  - `RawFace.ears: dict[Side, tuple[float, float]]`. Default empty, so far faces and existing tests are unaffected. Each value is the face-outline point beside that ear, in frame pixels.
  - `Person.ears_hidden: bool = False`.
  - `sides_to_check(yaw_deg: float) -> tuple[Side, ...]`.
  - `ear_square(anchor: tuple[float, float], face_width_px: float, side: Side, frame_w: int, frame_h: int) -> tuple[int, int, int] | None`. Returns `(x0, y0, size)` in pixels, or `None` when the square is not wholly inside the frame.

- [ ] **Step 1: Add config values**

  Add a `# ears` block to `Config`:

  ```python
  ear_both_yaw_deg: float = 20  # under this turn both ears are checked
  ear_yaw_sign: int = 1  # flip if the wrong ear is checked when turned; confirm on camera
  ear_offset_frac: float = 0.08  # square centre, outward from the face edge, x face width
  ear_square_frac: float = 0.3  # square side, x face width
  ```

- [ ] **Step 2: Write the failing tests in `tests/test_ears.py`**

  Tests:
  - `test_straight_checks_both`: `sides_to_check(0) == ("left", "right")` and `sides_to_check(19.9) == ("left", "right")`.
  - `test_turned_checks_facing_ear`: `sides_to_check(25) == ("left",)` and `sides_to_check(-25) == ("right",)`. This uses `ear_yaw_sign` 1. Also patch the sign to -1 and assert the two results swap.
  - `test_left_square_is_right_of_anchor`: on a 640x480 frame, anchor `(400, 240)`, face width 200. `ear_square(..., "left", ...)` has its centre x at `400 + 0.08 * 200` and size `60`.
  - `test_right_square_is_left_of_anchor`: the mirror of the test above, centre x at `240 - 16` for anchor `(240, 240)`.
  - `test_square_outside_frame_is_none`: anchor `(630, 240)`, face width 200, side `"left"` → `None`.

  The left ear sits to the right in the unmirrored camera frame, because the camera sees the person face to face.

- [ ] **Step 3: Run the tests to verify they fail**

  Run: `.venv\Scripts\python -m pytest tests/test_ears.py -q`

  Expected: FAIL with `ModuleNotFoundError: app.core.ears`.

- [ ] **Step 4: Implement the types, `sides_to_check` and `ear_square`**

  - `sides_to_check`: if `abs(yaw_deg) < CONFIG.ear_both_yaw_deg`, check both ears. Otherwise check `("left",)` when `yaw_deg * CONFIG.ear_yaw_sign > 0`, else `("right",)`.
  - `ear_square`: a centred square, with the centre pushed outward from the anchor. That means `+x` for `"left"` and `-x` for `"right"`. Round to ints.

- [ ] **Step 5: Run the tests to verify they pass**

  Run: `.venv\Scripts\python -m pytest tests/test_ears.py -q`

  Expected: PASS.

- [ ] **Step 6: Write the failing model test**

  Add `test_find_gives_ear_anchors` to `tests/test_models.py`, with `@pytest.mark.models` like its neighbours. On `sample_face.png`, the one found face has both `"left"` and `"right"` in `face.ears`, and `face.ears["left"][0] > face.ears["right"][0]`.

- [ ] **Step 7: Fill `RawFace.ears` in `FaceFinder.find`**

  - Use landmark 454 for `"left"` and 234 for `"right"`, in pixels like `points`.
  - Add `_EAR_INDICES = {"left": 454, "right": 234}` beside `_POINT_INDICES`.
  - If the test shows the sides reversed, swap the indices, not the test.

- [ ] **Step 8: Run the full suite**

  Run: `.venv\Scripts\python -m pytest -q`

  Expected: all pass.

- [ ] **Step 9: Commit**

  `feat: locate each ear from MediaPipe face points`

---

### Task 2: Photo mode and the ear-size check

**Files:**
- Create: `app/collect.py`: `EarCollector`. It may use cv2.
- Modify: `app/core/config.py`: `collect_interval_ms`.
- Modify: `app/pipeline.py`: hold an optional collector and feed it each frame.
- Modify: `app/server.py`: `collect` start flag and `collect` message.
- Modify: `web/app.js`, `web/index.html`, `web/styles.css`: the `?collect=1` controls and counts.
- Modify: `.gitignore`: add `dataset/` and `.venv-train/`.
- Test: `tests/test_collect.py`, `tests/test_server.py`.

**Interfaces:**
- Consumes: `sides_to_check`, `ear_square`, `RawFace.ears` (Task 1).
- Produces:
  - `CollectLabel = Literal["both_clear", "left_covered", "right_covered", "both_covered"]` in `app/collect.py`.
  - `EarCollector(root: Path)`. It saves under `root / "dataset" / "ears" / {"clear" | "covered"}`.
    - `start(label: CollectLabel, run_id: str) -> None`
    - `stop() -> None`
    - `offer(frame_bgr: np.ndarray, faces: list[RawFace], now_ms: float) -> None`. It saves at most once per `collect_interval_ms`. It skips far faces, faces with no `ears`, and any ear whose square is `None`.
    - `counts() -> dict[str, int]`, with keys `"clear"` and `"covered"`, counted from the folders.
  - File name: `{run_id}_{n:05d}_{side}.png`. The crop is saved at native pixel size, not resized, so the size check is honest.
  - `Pipeline(..., collector: EarCollector | None = None)`. `process` calls `collector.offer(frame, faces, now_ms)` after `_find`.
  - `FrameResult.collect_counts: dict[str, int] | None`. It is `None` unless collecting is on.
  - WebSocket:
    - `{"type": "start", "debug": bool, "collect": bool}`
    - `{"type": "collect", "label": CollectLabel | null}`. `null` stops the run, and a new label starts a new run with `run_id = datetime.now().strftime("%Y%m%d-%H%M%S")`.
    - A `collect` message on a session started without `collect: true` gets the existing `BAD_MESSAGE` error.

- [ ] **Step 1: Write the failing tests in `tests/test_collect.py`**

  Use `tmp_path` and a synthetic 640x480 frame with a `RawFace` that has `ears` set.

  Tests:
  - `test_left_covered_sorts_each_ear`: label `"left_covered"`, front-on face. One file lands in `covered/` named `*_left.png`, and one in `clear/` named `*_right.png`.
  - `test_turned_saves_only_facing_ear`: `yaw_deg=30`. Only the `left` ear is saved.
  - `test_interval_limits_saves`: offers at 0, 100 and 300 ms with `collect_interval_ms` 300 save 2 frames' worth.
  - `test_stopped_saves_nothing`.
  - `test_far_face_skipped`.
  - `test_counts_reads_folders`.
  - `test_saved_crop_is_native_size`: the file's width equals `ear_square(...)[2]`.

- [ ] **Step 2: Run the tests to verify they fail**

  Run: `.venv\Scripts\python -m pytest tests/test_collect.py -q`

  Expected: FAIL, import error.

- [ ] **Step 3: Implement `EarCollector`**

  - Add `collect_interval_ms: int = 300` to config.
  - Map labels to covered sides: `both_clear` covers none, `left_covered` covers `{"left"}`, `right_covered` covers `{"right"}`, `both_covered` covers both.
  - Write with `cv2.imwrite`. A failed write is logged, not raised.

- [ ] **Step 4: Run the collect tests**

  Expected: PASS.

- [ ] **Step 5: Write the failing server tests**

  Add to `tests/test_server.py`, following its existing fake-pipeline style:
  - `test_collect_message_needs_collect_session` → error `BAD_MESSAGE`.
  - `test_collect_session_passes_label`: the fake pipeline records `set_collect("left_covered")`, and later `set_collect(None)`.

- [ ] **Step 6: Wire the collector into the server and pipeline**

  - `make_pipeline(debug: bool, collect: bool)`. When `collect` is true it passes `EarCollector(ROOT)`.
  - `Pipeline.set_collect(label: CollectLabel | None) -> None` starts or stops the collector.
  - Update the `make` callable type and every call site, including test fakes.

- [ ] **Step 7: Build the page controls**

  Only when `?collect=1`:
  - **Label buttons:** four buttons, "Both clear", "Left covered", "Right covered", "Both covered", plus a Stop button.
  - **Note** beside the buttons: "Left and right as you see yourself on screen."
  - **Counts line:** "clear: N, covered: M".
  - **Active state:** the active label button shows pressed.

  Without `?collect=1` the page is unchanged.

- [ ] **Step 8: Run the full suite**

  Run: `.venv\Scripts\python -m pytest -q`

  Expected: all pass.

- [ ] **Step 9: Commit**

  `feat: add a hidden photo mode that saves ear squares`

- [ ] **Step 10: STOP. The owner does the ear-size check (spec 7.5)**

  1. The owner opens `http://127.0.0.1:8000/?collect=1` and saves about 20 `both_clear` and 20 `left_covered` squares, sitting normally with glasses on.
  2. Look at the saved PNGs together with the owner.
  3. If clear and covered can be told apart by eye, continue.
  4. If not, stop the plan and bring options to the owner. Do not change the frame size without them.

  Record the typical square size in pixels in the ledger.

---

### Task 3: Training setup

**Files:**
- Create: `training/split.py`: pure Python, no torch, so it is testable in the app venv.
- Create: `training/train.py`: torch and torchvision.
- Create: `training/requirements-train.txt`: CPU `torch` and `torchvision` (pinned), `onnx`.
- Create: `training/README.md`: setup and run steps for the owner.
- Modify: `app/core/config.py`: `ear_input_px`.
- Test: `tests/test_training_split.py`.

**Interfaces:**
- Consumes: the folder layout and file names from Task 2.
- Produces:
  - `split_by_run(paths: list[Path], test_share: float = 0.2, seed: int = 0) -> tuple[list[Path], list[Path]]`. The run id is the file-name part before the first `_`. Every file from one run lands on the same side. Run order is shuffled with `random.Random(seed)`. Runs are added to the test side until it holds at least `test_share` of the files, with at least one run of each label on the test side when that label has two or more runs.
  - `models/ears.onnx`: input `float32 [1, 3, ear_input_px, ear_input_px]`, RGB, ImageNet mean and std (same as `EmotionModel`). Output: logits, one per label.
  - `models/ears.json`: `{"labels": ["clear", "covered"], "input_px": 96}`. The labels are the sorted folder names, so 2b's `earpod` slots in without code changes.
  - Command: `.venv-train\Scripts\python training\train.py --data dataset\ears --out models\ears.onnx`

- [ ] **Step 1: Write the failing split tests**

  Tests:
  - `test_runs_never_split`: no run id appears on both sides.
  - `test_test_share_reached`: with 10 runs of 10 files each, the test side holds at least 20 files.
  - `test_each_label_tested`: two runs per label → each label has a run on the test side.
  - `test_same_seed_same_split`.

- [ ] **Step 2: Run them to verify they fail**

  Run: `.venv\Scripts\python -m pytest tests/test_training_split.py -q`

  Expected: FAIL, import error.

- [ ] **Step 3: Implement `split_by_run`**

  Add `ear_input_px: int = 96` to config.

- [ ] **Step 4: Run the split tests**

  Expected: PASS.

- [ ] **Step 5: Write `train.py`**

  - **Base model and license:** `torchvision.models.mobilenet_v3_small` with its ImageNet weights. Before writing code, check the license of the code and of the weights, and record it in `training/README.md`. If the weights' terms are unclear, record that as a blocker for commercial use, like the README does for HSEmotion.
  - **Classifier head:** replace the last layer with one output per label.
  - **Training passes:**
    1. Train only the new head for a few passes.
    2. Unfreeze the last blocks with a lower rate.
  - **Augmentation:** small shifts, brightness and contrast changes, and horizontal flips. A flipped left ear looks like a right ear, which is fine because labels are per ear, not per side.
  - **Report:** after training, print the held-out accuracy per label and a confusion table.
  - **Export:** ONNX with opset 17, plus `ears.json`.
  - **Input size:** read `ear_input_px` from `app.core.config`. That module is pure, so it imports cleanly.

- [ ] **Step 6: Write `training/README.md`**

  1. **Make the training setup:** `python -m venv .venv-train`, then `.venv-train\Scripts\pip install -r training\requirements-train.txt`.
  2. **Photos per label:** about 100 to 200 per label, taken over several Start-to-Stop runs.
  3. **What counts as `covered` and what to leave out:** copy spec 7.3.
  4. **The training command.**
  5. **How to read the report.**
  6. **The Colab fallback:** one line, owner's decision.

- [ ] **Step 7: Run the full suite and commit**

  `feat: add the ear model training script and its setup`

- [ ] **Step 8: STOP. The owner collects photos and trains**

  1. The owner collects photos. Cover clear, hand over the left ear, hand over the right ear, and hair over an ear if possible. Use several runs with head turns and different light, glasses on.
  2. Run the training command.
  3. Record the held-out accuracy, the confusion table and the training time in the ledger.
  4. If training on the laptop fails or takes over about 30 minutes, bring the Colab fallback to the owner.

---

### Task 4: `EarWatch`, the per-person ear state

**Files:**
- Modify: `app/core/ears.py`.
- Modify: `app/core/config.py`.
- Test: `tests/test_ears.py`.

**Interfaces:**
- Produces:
  - `EarStatus = Literal["unknown", "visible", "hidden"]`.
  - `class EarWatch`, with `update(covered: dict[Side, float] | None, now_ms: float) -> EarStatus`.
    - `covered` maps each checked side to its covered probability.
    - `None` or `{}` means no new check this frame.

- [ ] **Step 1: Add config values**

  ```python
  ear_interval_ms: int = 333  # how often the ear model runs per person
  ear_covered_at: float = 0.8  # cautious start; tune on camera
  ear_hidden_hold_ms: int = 1000  # covered checks for this long before red
  ear_clear_hold_ms: int = 500  # clear checks for this long before red clears
  ear_unchecked_grace_ms: int = 2000  # no check for this long drops the state quietly
  ```

- [ ] **Step 2: Write the failing tests**

  These use the defaults above and checks every 333 ms.

  Tests:
  - `test_starts_unknown`: `update(None, 0) == "unknown"`.
  - `test_clear_check_is_visible`: `update({"left": 0.1, "right": 0.1}, 0) == "visible"`.
  - `test_any_side_covered_counts`: `{"left": 0.9, "right": 0.1}` held from 0 to 1000 ms → `"hidden"` at 1000. A covered value is one at or above `ear_covered_at`.
  - `test_hidden_needs_full_hold`: covered checks at 0, 333 and 666 → still not `"hidden"` at 666.
  - `test_brief_cover_does_not_hide`: covered at 0 and 333, clear at 666, covered at 1000 and 1333 → never `"hidden"`.
  - `test_clear_needs_hold`: from `"hidden"`, a clear check at t → still `"hidden"`. A clear check at t+500 → `"visible"`.
  - `test_frames_without_checks_keep_state`: `update(None, ...)` between checks returns the last status.
  - `test_grace_drops_to_unknown`: `"hidden"`, then no check for 2000 ms → `"unknown"`.
  - `test_side_switch_keeps_state`: covered checks alternating `{"left": 0.9}` and `{"left": 0.9, "right": 0.2}` for 1000 ms → `"hidden"`.

- [ ] **Step 3: Run them to verify they fail**

  Run: `.venv\Scripts\python -m pytest tests/test_ears.py -q`

  Expected: the new tests FAIL.

- [ ] **Step 4: Implement `EarWatch`**

  Keep:
  - the current status;
  - the time of the first check in the current covered run, or of the first clear check while hidden;
  - the time of the last check.

  When no check is given, only the grace rule applies.

- [ ] **Step 5: Run the ear tests, then the full suite**

  Expected: all pass.

- [ ] **Step 6: Commit**

  `feat: hold the ear state steady across checks`

---

### Task 5: Ear model wrapper and pipeline join

**Files:**
- Modify: `app/models.py`: `EarModel`.
- Modify: `app/core/events.py`: `person_red`, ear log words.
- Modify: `app/core/text.py`: `ear_line`.
- Modify: `app/session_files.py`: `render_snap` box colour.
- Modify: `app/pipeline.py`: ear checks, red, snaps, log, `_person_dict`.
- Test: `tests/test_models.py`, `tests/test_events.py`, `tests/test_text.py`, `tests/test_session_files.py`, `tests/test_pipeline.py`.

**Interfaces:**
- Consumes: `sides_to_check`, `ear_square` (Task 1); `EarWatch`, `EarStatus` (Task 4); `models/ears.onnx` and `models/ears.json` (Task 3).
- Produces:
  - `EarModel(models_dir: Path)` raises `ModelsMissing` if either file is absent. `EarModel.covered(frame_bgr: np.ndarray, face: RawFace, sides: tuple[Side, ...]) -> dict[Side, float]` returns the softmax probability of `"covered"` per side whose square is not `None`.
  - `person_red(label: Label, ears_hidden: bool) -> bool` in `events.py`. `is_red` and `should_sound` are unchanged.
  - `EAR_HIDDEN_LOG = "ears are not visible"` and `EAR_VISIBLE_LOG = "ears visible again"` in `events.py`.
  - `ear_line(p: Person) -> str | None` in `text.py`: `"ears are not visible"` when `p.ears_hidden`, else `None`. The page adds the `↳ `.
  - `_person_dict` gains `"ear_line": ear_line(p)`. `"red"` becomes `person_red(p.reading.label, p.ears_hidden)`. `"reason"` stays `person_line(p) if is_red(p.reading.label) else None`, so it is set only for sleep-red, which the page relies on in Task 6.
  - `Pipeline(finder, emotion, signer, files, debug=False, collector=None, ears=None)`. `ears` is any object with `covered(frame, face, sides)`.

- [ ] **Step 1: Write the failing unit tests**

  - `test_events.py`, `test_person_red`:
    - `person_red("neutral", True)` is `True`.
    - `person_red("sleeping", False)` is `True`.
    - `person_red("neutral", False)` is `False`.
    - `should_sound(["neutral"])` is still `False`.
  - `test_text.py`, `test_ear_line`: `"ears are not visible"` for a hidden person, and `None` otherwise.
  - `test_session_files.py`, `test_snap_box_red_for_ears`: `render_snap` draws `RED_BGR` for `Person(..., ears_hidden=True)` with label `"neutral"`.
  - `test_models.py`:
    - `test_ear_model_missing_raises`.
    - `test_ear_model_shape`, marked `models` and skipped if `models/ears.onnx` is absent. On `sample_face.png` it returns keys for the checked sides, with values in 0..1.

- [ ] **Step 2: Run them to verify they fail, then implement**

  - **`EarModel`:** reads `ears.json` for the labels and input size. It crops each square, resizes it to `input_px`, and converts BGR to RGB. It normalises like `EmotionModel`, runs one batch for all sides, and applies softmax.
  - **`render_snap`:** uses `person_red`.

- [ ] **Step 3: Write the failing pipeline tests**

  These go in `tests/test_pipeline.py`. Use a `FakeEars` whose `covered()` returns a set dict, and that can raise.

  Tests:
  - `test_hidden_ears_make_person_red_silently`: covered for 1.5 s.
    - The person dict has `"red": True` and `"ear_line": "ears are not visible"`.
    - `alarm` is `False`.
    - The log has one `Person A | ears are not visible` line.
  - `test_ears_visible_again_logs`: after hidden, clear for 1 s → a log line `ears visible again`, and `ear_line` is `None`.
  - `test_ear_red_takes_its_own_snap`: hidden for over `min_red_ms` → one snap file whose name ends `_A_ears.jpg`.
  - `test_sleeping_and_ears_take_two_snaps`: the person is sleeping, then their ears go hidden → two snaps, and the alarm is on for sleep only.
  - `test_ear_check_rate`: over 1 s at `STEP_MS` 100, `FakeEars.covered` is called 3 or 4 times per person.
  - `test_far_drops_ear_state_quietly`: hidden, then the face turns far for 2.5 s.
    - `ear_line` becomes `None`.
    - There is no `ears visible again` line.
    - The far face never calls `covered`.
  - `test_leaving_saves_owed_ear_snap`: hidden for 1.2 s, then the face is gone → the ear snap is still saved when the letter is released.
  - `test_ear_model_error_warns_and_continues`: `covered` raises → `FACE_WARNING` and no crash, and the next frame still works.
  - `test_no_ear_model_no_ear_lines`: `ears=None` → `ear_line` is always `None`, and the existing behaviour is unchanged.
  - `test_no_letter_no_ear_check`: a face still being identified is never checked.

- [ ] **Step 4: Implement the pipeline join**

  - **State:** `_PersonState` gains `ears: EarWatch`, `ear_tried_ms: float = -math.inf` and `ear_status: EarStatus = "unknown"`.
  - **Running checks:** in `_read_people`, for a near face with a letter, when `ears` is set and `now_ms - ear_tried_ms >= ear_interval_ms`, call `covered(frame, face, sides_to_check(face.yaw_deg))`, wrapped like `_read_emotion`. Otherwise pass `None`.
  - **Far faces:** pass `None`, so the grace rule ends the state.
  - **Person:** `Person.ears_hidden = status == "hidden"`. `_read_people` needs `frame` and `warnings`, so add those parameters.
  - **Snaps:** in `_events`, run the snap policy and red proof a second time per person under the key `f"{key}#ears"`, with `red = person.ears_hidden`. The far branch and the leaving branch also drop that key with `_owed_snap`.
  - **`_snap_event`:** gains `reason: str` and `what: str`. The sleep path passes the label for both. The ear path passes `"ears"` and `EAR_HIDDEN_LOG`, with no confidence.
  - **Log lines:** track the last `EarStatus` per key and log only `visible`/`unknown` → `hidden` (`EAR_HIDDEN_LOG`) and `hidden` → `visible` (`EAR_VISIBLE_LOG`). The change `hidden` → `unknown` logs nothing.

- [ ] **Step 5: Run the full suite**

  Expected: all pass, including all 451 earlier tests.

- [ ] **Step 6: Commit**

  `feat: flag hidden ears as red, with a snap and log line`

---

### Task 6: Load the model, show layout C, update docs

**Files:**
- Modify: `app/server.py`: optional `EarModel` load.
- Modify: `web/app.js`, `web/styles.css`: the second line.
- Modify: `README.md`, `docs/superpowers/HANDOVER.md`.
- Test: `tests/test_server.py`.

**Interfaces:**
- Consumes: `EarModel`, `ModelsMissing` (Task 5); `make_pipeline(debug, collect)` (Task 2).
- Produces:
  - `EAR_OFF_WARNING = "Ear check is off: models/ears.onnx was not found."` It is shown once, as the first frame's warning, via the pipeline's pending warnings.
  - `Pipeline` accepts `ears_off: bool = False` and adds `EAR_OFF_WARNING` to `_pending_warnings` when true.

- [ ] **Step 1: Write the failing server test**

  `test_missing_ear_model_runs_without_ears`:
  - `models/` has no `ears.onnx`.
  - `make_pipeline` still returns a pipeline with `ears=None`.
  - The first frame's `warning` contains `EAR_OFF_WARNING`, and later frames don't.

  Also make a corrupt `ears.onnx` in `tmp_path`. The load error is caught, and the result is the same as when the file is missing.

- [ ] **Step 2: Implement the optional load**

  In `_loaded_models`, load `EarModel` in its own `try`, catching `ModelsMissing` and onnxruntime load errors. Log them, and keep `None`.

- [ ] **Step 3: Render layout C in `renderStrip`**

  - After each person line, when `reply.people[i]?.ear_line` is set, add a `div` with class `strip-person strip-ear alert` and text `↳ ${ear_line}`.
  - The expression line keeps `alert` only when the expression itself is red. Use `reply.people[i].reason !== null` for that: today `reason` is set only for sleep-red.
  - The box colour already follows `red`.

- [ ] **Step 4: Update the README and handover**

  - **README:**
    - The ear check and what it shows.
    - Photo mode (`?collect=1`).
    - A pointer to `training/README.md`.
    - The new config numbers, in the tuning guide.
    - The ear model license line.
  - **Handover:** the Stage 2a status and on-camera results.

- [ ] **Step 5: Run the full suite and commit**

  `feat: show the ear line on the page and load the ear model if present`

- [ ] **Step 6: STOP. The owner's on-camera checks (spec section 9)**

  Run all 8 checks with the owner:
  1. Sitting normally.
  2. Head turns and looking at the keyboard.
  3. Hand over one ear.
  4. Hand removed.
  5. Hair over an ear.
  6. Sleep still sounds.
  7. Frame rate.
  8. Training report.

  - **Tuning:** tune `ear_covered_at`, the hold times, `ear_offset_frac`, `ear_square_frac` and `ear_yaw_sign` one at a time from real readings.
  - **Records:** put the numbers in the ledger, and pin them with `test_owner_ears_*` tests as was done for sleep.
  - **Frame rate:** measure the ear check's time per crop and the one-face frame rate, and record both.
  - **Commit:** `fix: tune the ear check from real readings`.
