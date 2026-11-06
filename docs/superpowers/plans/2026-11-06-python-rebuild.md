# Python Rebuild, Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Python app that reads frames from a thin browser page and shows each recognised person's expression, sounds an alarm for sleep, and records every session with snaps of each red mark.

**Architecture:** A browser page captures the camera and sends JPEG frames over one WebSocket to a FastAPI server on `127.0.0.1`. Pure Python modules in `app/core/` hold all rules and are unit tested with fake inputs. `app/models.py` wraps the real models, and `app/pipeline.py` joins them with dependency injection, so tests need no camera and no model files.

**Tech Stack:** Python 3.12, FastAPI 0.162.0, uvicorn 0.54.0, mediapipe 1.2.2, onnxruntime 1.31.1, opencv-contrib-python 5.0.0.93, numpy 2.5.4, pytest 9.2.0, httpx 0.28.1.

**Spec:** `docs/superpowers/specs/2026-11-06-python-rebuild-design.md`

**Reference code:** the TypeScript version on branch `feat/expression-demo`. Read with `git show feat/expression-demo:<path>`.

## Global Constraints

- No text on the camera view: only a thin box and a small letter per face.
- Nothing leaves the laptop. The server binds to `127.0.0.1`. No CDN, no analytics, no network calls at runtime.
- Models are downloaded once by the setup script to `models/`. The app works offline afterwards.
- No PyTorch. No Node or build step. Exact versions pinned in `requirements.txt`.
- At most 4 faces processed at once. Target 15 frames per second, 8 acceptable.
- Video is mirrored by CSS only. Frames sent to the server are never mirrored.
- `app/core/` must not import cv2, mediapipe, onnxruntime or fastapi.
- All tunable numbers live in `app/core/config.py`.
- Face signatures stay in memory, never on disk.
- Every camera, model, file and socket operation has error handling that ends in a readable message on the page.
- `logs/`, `snaps/` and `models/` are git-ignored.
- Git: never commit to `main`. All work is on `feat/python-rebuild`, with conventional commit messages.
- Every commit message ends with the attribution trailer from the session (Co-Authored-By naming the authoring model, plus `Claude-Session: https://claude.ai/code/session_0175kJheuKPY7sg8wM26zVt6`).

## Review Focus

1. A corrupt or empty frame arrives: it is skipped, and the session carries on. Test in Task 6.
2. `logs/` or `snaps/` can't be written: a warning reaches the page and detection keeps running. Tests in Tasks 4 and 6.
3. The browser disconnects mid-session: the log gets an end line and the next client is accepted. Test in Task 7.
4. Two similar faces in view at once: they never share a letter. Test in Task 3.
5. A red person leaves the view: the sound stops once their letter's grace ends, and returning still red within the same event takes no second snap. Test in Task 6.

---

## File Map

| File | Responsibility |
|---|---|
| `app/core/config.py` | All thresholds and timings |
| `app/core/types.py` | Shared dataclasses |
| `app/core/signals.py` | Raw scores to signals, per-person baseline, head angles |
| `app/core/classifier.py` | Label and confidence over time |
| `app/core/expressions.py` | Model output to label |
| `app/core/tracker.py` | Faces between frames, by position |
| `app/core/identity.py` | Session letters from face signatures |
| `app/core/text.py` | Panel and log wording |
| `app/core/events.py` | Label changes to log events, alarm decision |
| `app/core/snap_policy.py` | When a snap is due |
| `app/core/debug_log.py` | Rolling tuning buffer |
| `app/session_files.py` | Writes log files and snap images |
| `app/models.py` | Real MediaPipe, HSEmotion and SFace wrappers |
| `app/pipeline.py` | One frame in, one result out |
| `app/server.py` | FastAPI app and WebSocket |
| `web/index.html`, `web/styles.css`, `web/app.js` | Thin page |
| `scripts/setup_models.py` | Downloads models |

---

### Task 1: Project reset, tooling and shared types

**Files:**
- Delete: `src/`, `tests/`, `scripts/fetch-assets.mjs`, `public/`, `dist/`, `node_modules/`, `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `README.md`
- Create: `.gitignore`, `requirements.txt`, `pytest.ini`, `app/__init__.py`, `app/core/__init__.py`, `app/core/config.py`, `app/core/types.py`, `scripts/setup_models.py`, `tests/__init__.py`

**Interfaces:**
- Produces, in `types.py`:
  ```python
  Label = Literal['sleeping','drowsy','yawning','angry','laughing','smiling','crying','neutral']
  @dataclass(frozen=True) class Box: x: float; y: float; width: float; height: float  # 0..1 of the frame
  @dataclass class RawFace: box: Box; scores: dict[str, float]; pitch_deg: float; yaw_deg: float; points: list[tuple[float, float]]  # 5 points, pixels
  @dataclass(frozen=True) class Signals: eye_closure, look_down, pitch_delta, smile, jaw_open, brow_down, brow_inner_up, frown: float
  @dataclass(frozen=True) class Reading: label: Label; confidence: int  # 0..100
  @dataclass class Person: letter: str | None; box: Box; reading: Reading; visible: bool
  @dataclass(frozen=True) class LogEvent: at: datetime; who: str; what: str; confidence: int | None = None; snap: str | None = None
  ```
- Produces, `config.py`: `CONFIG`, a frozen dataclass instance. Copy every sleep, label and baseline value from `git show feat/expression-demo:src/core/config.ts`, in snake_case. Drop the overlay, beep, smile-based angry and crying rule keys only if Task 2 no longer uses them. Add: `max_faces 4, target_fps 15, yawn_jaw_at 0.5, yawn_hold_ms 1500, emotion_interval_ms 333, identity_interval_ms 1000, identity_match_at 0.363, identity_pending_ms 2000, identity_max_signatures 5, frontal_yaw_deg 20, min_red_ms 1000, jpeg_max_bytes 2000000`.

- [ ] **Step 1: Remove the TypeScript files** listed above with `git rm -r` for tracked ones. Replace `.gitignore` with: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.superpowers/`, `models/`, `logs/`, `snaps/`, `.env`.

- [ ] **Step 2: Add `requirements.txt`** with the exact Tech Stack versions, and create `.venv`: `python -m venv .venv`, then `.venv/Scripts/python -m pip install -r requirements.txt`.

- [ ] **Step 3: Write `scripts/setup_models.py`**

Downloads to `models/`, skipping files that exist, writing to a temp name first and renaming on success, and exiting non-zero with a readable message on failure:
- `face_landmarker.task` from `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- `emotion.onnx` from `https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx`
- `sface.onnx` from `https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx`
- `sample_face.png` from `https://raw.githubusercontent.com/scikit-image/scikit-image/v0.22.0/skimage/data/astronaut.png`

- [ ] **Step 4: Add `config.py`, `types.py`, `pytest.ini`** (`testpaths = tests`, a `models` marker for tests needing real model files).

- [ ] **Step 5: Verify**

Run: `.venv/Scripts/python scripts/setup_models.py && .venv/Scripts/python -c "import cv2, mediapipe, onnxruntime, fastapi; print(hasattr(cv2,'FaceRecognizerSF'))"`
Expected: four files in `models/`, prints `True`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: replace TypeScript project with Python tooling"
```

---

### Task 2: Signals, classifier and expressions

**Files:**
- Create: `app/core/signals.py`, `app/core/classifier.py`, `app/core/expressions.py`
- Test: `tests/test_signals.py`, `tests/test_classifier.py`, `tests/test_expressions.py`

**Interfaces:**
- Consumes: `RawFace`, `Signals`, `Reading`, `CONFIG`.
- Produces:
  ```python
  # signals.py
  def angles_from_matrix(m: Sequence[float]) -> tuple[float, float]  # (pitch_deg, yaw_deg), column-major 4x4
  class Baseline: def to_signals(self, face: RawFace, now_ms: float) -> Signals
  # expressions.py
  EMOTIONS = ('Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise')  # model output order
  def expression_reading(scores: dict[str, float] | None, s: Signals) -> Reading
  # classifier.py
  class PersonClassifier:
      def update(self, s: Signals, now_ms: float, delay_sec: float, emotion: dict[str, float] | None) -> Reading
      def debug_info(self) -> dict
  ```

**Port rules:** port `src/core/signals.ts`, `src/core/classifier.ts` and their tests from `feat/expression-demo` faithfully, keeping every scenario. Test names become snake_case. Only these changes apply:
- `pitchFromMatrix` becomes `angles_from_matrix`. Pitch is unchanged. Yaw is `degrees(atan2(-m[2], hypot(m[0], m[1])))`.
- The blendshape expression rules stay for "strong expression" protection on the sleep path. The DISPLAYED non-sleep label comes from `expression_reading`.
- New priority in the candidate step: sleeping, then drowsy, then yawning, then `expression_reading`.
- **Yawning:** `jaw_open >= yawn_jaw_at` and `smile < smile_at`, held for `yawn_hold_ms`. Confidence is `jaw_open * 100`. It skips nothing: the normal label hold applies once the yawn hold has passed.

**`expression_reading` rules:** with `scores` None, neutral at 100. Otherwise take the highest of the eight. Anger gives angry, Sadness crying, Neutral neutral. Happiness gives laughing if `jaw_open >= laugh_jaw_at`, else smiling. Contempt, Disgust, Fear and Surprise give neutral. Confidence is the top score times 100, rounded.

- [ ] **Step 1: Port the tests.** Scenarios that asserted laughing, smiling, angry, crying or neutral from blendshapes now pass an `emotion` dict that gives the expected label, plus the blendshape values the sleep path needs. Add:
  - `test_yawn_held_1_5_seconds_shows_yawning`, and at 1.0s it does not.
  - `test_wide_mouth_with_smile_is_not_a_yawn`
  - `test_yawning_with_closed_eyes_still_reaches_sleeping_at_delay`
  - `test_yawning_alone_never_gives_drowsy`
  - `test_happiness_with_open_jaw_is_laughing`, `test_happiness_closed_mouth_is_smiling`
  - `test_anger_sadness_and_neutral_map`, `test_contempt_disgust_fear_surprise_map_to_neutral`
  - `test_no_emotion_yet_is_neutral`
  - `test_yaw_from_matrix`: identity gives `(0, 0)`, a 30 degree Y rotation gives yaw of magnitude 30.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_signals.py tests/test_classifier.py tests/test_expressions.py -q`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement the three modules.**

- [ ] **Step 4: Run to verify pass**

Same command. Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core tests
git commit -m "feat: port sleep rules and add trained expression mapping and yawning"
```

---

### Task 3: Tracking and recognition

**Files:**
- Create: `app/core/tracker.py`, `app/core/identity.py`
- Test: `tests/test_tracker.py`, `tests/test_identity.py`

**Interfaces:**
- Produces:
  ```python
  # tracker.py: port of tracker.ts, keyed by integer track id, not letter
  class Tracker:
      def update(self, faces: list[RawFace], now_ms: float) -> list[tuple[int, RawFace]]
      def active_ids(self) -> set[int]   # includes faces in their grace period
  # identity.py
  class Identity:
      def letter_for(self, track_id: int) -> str | None
      def needs_signature(self, track_id: int, now_ms: float) -> bool
      def add_signature(self, track_id: int, signature: np.ndarray, frontal: bool, now_ms: float) -> None
      def sync(self, visible_ids: set[int], active_ids: set[int], now_ms: float) -> None
      def people_count(self) -> int
  ```
  `identity.py` may import numpy.

**Identity rules:**
- A new track starts with no letter.
- `add_signature` on a track with no letter: compare by cosine similarity with stored signatures of every letter NOT currently held by a VISIBLE track. Best score at or above `identity_match_at` takes that letter, moving it from any hidden track. Otherwise the track takes the next free letter, A to Z.
- A track still without a letter after `identity_pending_ms` takes the next free letter, so a person is never stuck.
- After Z, `letter_for` returns `'?'`.
- `needs_signature` is True at once for a track with no letter. For a lettered track it is True every `identity_interval_ms`.
- Frontal signatures only are stored, up to `identity_max_signatures` per letter, oldest dropped. A letterless track's first signature is compared even if not frontal, and is stored only if frontal.
- A lettered track's new signature that clearly matches a DIFFERENT non-visible letter better is ignored. Letters don't flip mid-track.
- Signatures are held in memory only.

- [ ] **Step 1: Port tracker tests** from `tests/tracker.test.ts` with integer ids. The fifth-face test becomes: `max_faces` smallest-box faces beyond 4 are dropped, keeping the 4 largest. Add `test_fifth_face_drops_the_smallest`.

- [ ] **Step 2: Write identity tests**, using unit vectors as signatures:
  - `test_first_person_gets_a`
  - `test_returning_person_gets_old_letter_back`: A's track gone and released, new track with A's signature gets `A`.
  - `test_return_within_grace_at_new_position_keeps_letter`: A's track hidden but active, new track with A's signature gets `A`, and the old track has no letter. This is the bug the owner found.
  - `test_different_person_gets_next_letter`
  - `test_two_similar_faces_in_view_never_share_a_letter`: two visible tracks with the same signature get different letters.
  - `test_no_signature_for_2_seconds_gets_next_letter`
  - `test_after_z_letter_is_question_mark`
  - `test_side_view_signature_is_not_stored`
  - `test_stored_signatures_are_capped`
  - `test_lettered_track_never_flips_to_another_letter`

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_tracker.py tests/test_identity.py -q`
Expected: FAIL, modules not found.

- [ ] **Step 4: Implement `tracker.py` and `identity.py`.**

- [ ] **Step 5: Run to verify pass**

Same command. Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app/core tests
git commit -m "feat: track faces and keep letters with session face recognition"
```

---

### Task 4: Wording, events, snaps policy and session files

**Files:**
- Create: `app/core/text.py`, `app/core/events.py`, `app/core/snap_policy.py`, `app/core/debug_log.py`, `app/session_files.py`
- Test: `tests/test_text.py`, `tests/test_events.py`, `tests/test_snap_policy.py`, `tests/test_debug_log.py`, `tests/test_session_files.py`

**Interfaces:**
- Produces:
  ```python
  # text.py
  def face_count_text(n: int) -> str
  def person_line(p: Person) -> str
  # events.py
  RED_LABELS = frozenset({'sleeping', 'drowsy'})
  def is_red(label: Label) -> bool
  def should_sound(labels: Iterable[Label], muted: bool = False) -> bool
  def diff_events(prev: dict[str, Label], nxt: dict[str, Reading], at: datetime) -> list[LogEvent]
  # snap_policy.py
  class SnapPolicy: def update(self, letter: str, red: bool, now_ms: float) -> bool  # True once per red event
  # debug_log.py: port of debugLog.ts
  class DebugBuffer: add(sample: dict); samples() -> list[dict]; recent_ranges(letter: str, now_ms: float) -> dict | None
  # session_files.py
  class SessionFiles:
      def __init__(self, root: Path, started: datetime)
      def write(self, event: LogEvent) -> str | None        # returns a warning, or None
      def save_snap(self, image: np.ndarray, at: datetime, letter: str, reason: str) -> tuple[str | None, str | None]  # (relative path, warning)
      def close(self, at: datetime) -> None
  def render_snap(frame_bgr: np.ndarray, people: list[Person]) -> np.ndarray
  ```

**Exact copy:**
- Count: `No faces found`, `1 face found`, `2 faces found`. Line: `Person A is sleeping 92%`, `is about to sleep`, `is yawning`, `is angry`, `is laughing`, `is smiling`, `is crying`, `looks neutral`. A person with letter None shows `Identifying...`.
- Log wording from `src/core/activityLog.ts`, plus `started yawning`. Also `alarm started`, `alarm stopped`, `session started`, `session ended`, `snap failed`.
- Log line: `16:42:11 | Person A | sleeping | 92% | snaps/2026-11-06_16-30-00_16-42-11_A_sleeping.jpg`. Empty fields are omitted with their separators. System events use `System` as the who.
- Log path `logs/session-2026-11-06_16-30-00.log`. Snap path as above. Manual snaps use `manual` as the reason and `all` as the letter.

**Rules:**
- `SnapPolicy` returns True once when a letter has been red continuously for `min_red_ms`. It re-arms when that letter goes non-red.
- `render_snap` flips the frame horizontally, draws boxes at `x' = 1 - x - width`, with the same colours as the page, and a letter. Hidden people get no box.
- `SessionFiles` creates folders on demand. Every line is flushed straight away. Any `OSError` is caught and returned as a warning, never raised. The session continues.

- [ ] **Step 1: Write tests**, porting `tests/text.test.ts`, `tests/activityLog.test.ts`, `tests/alarmPolicy.test.ts` and `tests/debugLog.test.ts` scenarios, plus:
  - `test_flicker_under_one_second_takes_no_snap`
  - `test_red_for_one_second_takes_exactly_one_snap`: then 60s more red gives no second True.
  - `test_going_normal_then_red_is_a_new_event`
  - `test_two_people_have_separate_snap_events`
  - `test_log_line_format_with_snap`, `test_log_line_format_without_confidence`
  - `test_each_session_has_its_own_file`
  - `test_line_is_on_disk_before_close`: read the file before `close`.
  - `test_snap_saved_with_expected_name`, using `tmp_path`.
  - `test_unwritable_log_returns_warning_and_does_not_raise`
  - `test_unwritable_snap_returns_warning_and_does_not_raise`
  - `test_render_snap_mirrors_box_position`: a box on the left of the input has coloured pixels on the right of the output.
  - `test_render_snap_skips_hidden_people`

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests -q -k "text or events or snap or debug or session"`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement the five modules.**

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app tests
git commit -m "feat: add panel wording, snap policy and session log files"
```

---

### Task 5: Real model wrappers

**Files:**
- Create: `app/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `RawFace`, `Box`, `angles_from_matrix`, `CONFIG`, `EMOTIONS`.
- Produces:
  ```python
  class ModelsMissing(Exception): ...          # message names scripts/setup_models.py
  class FaceFinder:
      def __init__(self, models_dir: Path); def find(self, frame_bgr: np.ndarray) -> list[RawFace]
  class EmotionModel:
      def __init__(self, models_dir: Path); def scores(self, face_bgr: np.ndarray) -> dict[str, float]
  class FaceSigner:
      def __init__(self, models_dir: Path); def signature(self, frame_bgr: np.ndarray, face: RawFace) -> np.ndarray
  ```

**Rules:**
- `FaceFinder`: mediapipe tasks `FaceLandmarker`, `RunningMode.IMAGE`, `num_faces` 8 so that Tracker can choose the 4 largest, blendshapes and matrices on. Box from landmark min and max, clamped to 0..1. `points` in pixels, in SFace order, from landmark indices: 468 and 473 eyes, 1 nose, 61 and 291 mouth corners.
- `EmotionModel`: onnxruntime CPU. Pad the crop by 10% and clamp, convert to RGB, resize to 224x224, scale to 0..1, normalise with mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`, NCHW, softmax. Keys come from `EMOTIONS`.
- `FaceSigner`: `cv2.FaceRecognizerSF.create`. Build a 15-value YuNet-style row from the box and `points` for `alignCrop`, then `feature`, returned flattened and L2-normalised, so `Identity` cosine is a dot product.
- A missing file raises `ModelsMissing`.

- [ ] **Step 1: Write tests, marked `models`:**
  - `test_finds_one_face_in_sample`, with 5 points and scores containing `eyeBlinkLeft`.
  - `test_blank_frame_finds_no_faces`
  - `test_sample_smiles`: `EmotionModel` top class for the sample crop is `Happiness`, and the scores sum to about 1.
  - `test_same_face_matches_itself`: cosine above 0.9.
  - `test_flipped_noise_does_not_match`: cosine with a random-noise crop's signature below `identity_match_at`.
  - `test_missing_models_raise_with_setup_hint`: unmarked, using an empty `tmp_path`.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `app/models.py`.**

- [ ] **Step 4: Run to verify pass**

Same command. Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app tests
git commit -m "feat: wrap MediaPipe, HSEmotion and SFace models"
```

---

### Task 6: Frame pipeline

**Files:**
- Create: `app/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  ```python
  @dataclass class FrameResult:
      count_text: str; lines: list[str]; people: list[dict]; log: list[str]
      alarm: bool; warning: str | None; overflow: bool; debug: list[str] | None
  class Pipeline:
      def __init__(self, finder, emotion, signer, files: SessionFiles, debug: bool = False)
      def process(self, jpeg: bytes, now_ms: float, at: datetime, delay_sec: float) -> FrameResult
      def snapshot(self, at: datetime) -> str | None   # warning
      def debug_samples(self) -> list[dict]
      def close(self, at: datetime) -> None
  ```
  `people` dicts: `{letter, box: {x,y,width,height}, label, confidence, red, reason, visible}`. `reason` is the person's line wording for red people, else None.

**Per-frame order:**
1. Decode. Reject empty, over `jpeg_max_bytes` or undecodable frames: return the last result with `warning` set, and no state change.
2. `finder.find`, then `tracker.update`.
3. One heavy check per frame at most, in priority order: a track needing a signature, then the person whose emotion is most overdue by `emotion_interval_ms`. This keeps the frame cost bounded on the i3.
4. `identity.sync`.
5. Per lettered person, state stays in a dict by letter for the whole session: `Baseline`, `PersonClassifier`, last emotion scores. Never delete it. Run `to_signals` and `update`.
6. Build `Person` list, including hidden people still in grace with their last reading.
7. `diff_events`, alarm start and stop events, `SnapPolicy`, `render_snap`, `save_snap`, and write all events. Any warning goes on the result.

`overflow` is True if `finder` returned more than `max_faces`.

- [ ] **Step 1: Write tests with fakes** for finder, emotion and signer. Fake finder returns scripted `RawFace` lists. Fake signer returns a vector chosen by a marker in the face. Use real `SessionFiles` on `tmp_path` and a real tiny JPEG:
  - `test_person_leaves_and_returns_keeps_letter`: gone 10s, same signature, still `A`.
  - `test_leave_and_return_keeps_baseline`: the same `Baseline` object after return.
  - `test_closed_eyes_goes_red_and_sounds`, with alarm True and red True.
  - `test_red_event_writes_one_snap_and_links_it_in_the_log`
  - `test_alarm_stops_after_red_person_leaves`: True during grace, False after `letter_release_ms`.
  - `test_red_person_returning_within_grace_takes_no_second_snap`
  - `test_corrupt_frame_is_skipped_and_session_continues`
  - `test_empty_frame_is_skipped`
  - `test_unwritable_snap_folder_warns_and_keeps_detecting`
  - `test_one_heavy_check_per_frame`: with 4 new faces, at most 1 signer or emotion call per `process`.
  - `test_more_than_four_faces_sets_overflow`
  - `test_alarm_start_and_stop_are_logged`
  - `test_unknown_person_is_not_classified_until_lettered`: `Identifying...` line, no red.
  - `test_manual_snapshot_saves_file`

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -q`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `app/pipeline.py`.**

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app tests
git commit -m "feat: join models and rules into the frame pipeline"
```

---

### Task 7: Server and page

**Files:**
- Create: `app/server.py`, `web/index.html`, `web/styles.css`, `web/app.js`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `Pipeline`, `SessionFiles`, `FaceFinder`, `EmotionModel`, `FaceSigner`, `ModelsMissing`.
- Produces: `create_app(make_pipeline: Callable[[bool], Pipeline]) -> FastAPI`, and module-level `app` for real use. Run with `uvicorn app.server:app --host 127.0.0.1 --port 8000`.

**WebSocket `/ws` protocol:**
- Client text JSON: `{"type":"start","debug":bool}`, `{"type":"settings","delaySec":n}`, `{"type":"snapshot"}`, `{"type":"debug_dump"}`.
- Client binary: one JPEG frame. Server replies to each with one JSON `{"type":"frame", ...FrameResult fields}`.
- Server: `{"type":"ready"}`, `{"type":"busy"}` then close when a session is already open, `{"type":"error","message":...}`, `{"type":"debug_dump","samples":[...]}`.
- A blocking `process` call runs through `asyncio.to_thread`, so the server stays responsive.
- `WebSocketDisconnect` or any exit calls `pipeline.close`, in `finally`.
- Models load once, on first start, and are reused by later sessions. A new `Pipeline` per session gets fresh tracker, identity and files. A `ModelsMissing` becomes an `error` message.
- Static files from `web/` are mounted at `/` AFTER the WebSocket route.

**Page:** port from `feat/expression-demo`: `index.html`, `src/styles.css`, `src/browser/beeper.ts`, `src/ui/view.ts`, and the `src/main.ts` controls, as plain JavaScript with no imports from a CDN.
- Beep values stay: 880 Hz, 300ms on, 200ms off.
- Capture a 640x480 JPEG at quality 0.8 from an off-screen canvas of the unflipped video. Send the next frame only after the reply, and no faster than `1000 / 15` ms.
- Boxes use `x' = 1 - x - width` on an unflipped overlay canvas. Green, and red for `red` people. Hidden people get no box. Only the letter is drawn.
- Strip shows `count_text` and `lines`, with red people's lines in red. Show `overflow` as `Showing the 4 largest faces`.
- Mute and volume stay in the page. `alarm` from the server drives the beeper unless muted.
- A permanent `Recording logs and snaps` notice.
- Hidden tab: stop sending, silence the alarm, and on return show `Detection paused while this tab was hidden`.
- Socket closed: alarm off, message `Connection to the app was lost. Check it is running, then reload.`
- `busy`: message `The app is already open in another tab.`
- `?debug=1` shows the debug lines and a `Download debug log` button that saves `debug_dump` samples as JSON.

- [ ] **Step 1: Write tests** with `create_app` and a fake pipeline, using `TestClient`:
  - `test_page_is_served`: `GET /` returns 200 and contains `Start camera`.
  - `test_start_then_frame_replies_with_frame_json`
  - `test_second_client_is_told_busy`
  - `test_disconnect_closes_session_and_next_client_is_accepted`
  - `test_delay_setting_reaches_the_pipeline`
  - `test_snapshot_message_reaches_the_pipeline`
  - `test_missing_models_sends_error_message`
  - `test_pipeline_exception_sends_error_and_closes_session`

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_server.py -q`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `app/server.py`.**

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Build the page.**

- [ ] **Step 6: Verify in real Chrome with a fake camera**

Run the server. Open Chrome with `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream`, from a throwaway script outside the repo.
Expected: Start works, `No faces found` shows, no page errors, a `logs/session-*.log` file appears with `session started`, and only `127.0.0.1` requests occur. Report measured frames per second.

- [ ] **Step 7: Commit**

```bash
git add app web tests
git commit -m "feat: add local server and browser page"
```

---

### Task 8: README and owner checks

The owner does tuning on camera, since an agent can't act out expressions.

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

Must include: what it is; setup (`python -m venv .venv`, install, `python scripts/setup_models.py`, run, open `http://127.0.0.1:8000`); offline use; where logs and snaps go and that they hold photos of people; not for driver or medical use, and flags need human review; confidence wording from spec section 9; all known limits from spec section 10; alarm can sound before the delay for about-to-sleep and head droop; hidden tab limit; a table of each model's license, checked against its source, with any non-commercial license stated as a blocker for commercial or exam use; troubleshooting; a "Tuning on camera" section using `?debug=1`, starting with the check of `look_down` on closed eyes; and spec section 11's checks as a checklist.

- [ ] **Step 2: Verify**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all PASS. Then follow the README's setup commands exactly in a fresh virtual environment.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, limits and tuning guide"
```

- [ ] **Step 4: Owner on-camera pass**

The owner runs the spec section 11 checks. First capture real signals with `?debug=1` and send them for analysis, before any threshold changes.
