from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import CONFIG
from app.core.ears import sides_to_check
from app.core.types import Box, FarDetectFailed, RawFace
from app.models import (
    EarModel,
    EmotionModel,
    FaceFinder,
    FaceSigner,
    ModelsMissing,
    _far_faces,
    _padded_crop,
)
from app.pipeline import Pipeline
from app.session_files import SessionFiles

MODELS = Path(__file__).resolve().parent.parent / "models"


def _sample() -> np.ndarray:
    img = cv2.imread(str(MODELS / "sample_face.png"))
    assert img is not None
    return img


@pytest.mark.models
def test_finds_one_face_in_sample():
    faces = FaceFinder(MODELS).find(_sample())
    assert len(faces) == 1
    f = faces[0]
    assert len(f.points) == 5
    assert "eyeBlinkLeft" in f.scores
    assert 0 <= f.box.x <= 1 and 0 < f.box.width <= 1
    assert abs(f.yaw_deg) < 10 and abs(f.pitch_deg) < 25


@pytest.mark.models
def test_find_gives_ear_anchors():
    faces = FaceFinder(MODELS).find(_sample())
    assert len(faces) == 1
    face = faces[0]
    assert "left" in face.ears and "right" in face.ears
    assert face.ears["left"][0] > face.ears["right"][0]


@pytest.mark.models
def test_blank_frame_finds_no_faces():
    blank = np.zeros((480, 640, 3), np.uint8)
    assert FaceFinder(MODELS).find(blank) == []


@pytest.mark.models
def test_sample_smiles():
    frame = _sample()
    face = FaceFinder(MODELS).find(frame)[0]
    scores = EmotionModel(MODELS).scores(frame, face.box)
    assert max(scores, key=scores.get) == "Happiness"
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.models
def test_same_face_matches_itself():
    frame = _sample()
    face = FaceFinder(MODELS).find(frame)[0]
    signer = FaceSigner(MODELS)
    a, b = signer.signature(frame, face), signer.signature(frame, face)
    assert float(a @ b) > 0.9
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.models
def test_noise_does_not_match():
    frame = _sample()
    face = FaceFinder(MODELS).find(frame)[0]
    signer = FaceSigner(MODELS)
    noise = np.random.default_rng(0).integers(0, 256, frame.shape, dtype=np.uint8)
    assert float(signer.signature(frame, face) @ signer.signature(noise, face)) < CONFIG.identity_match_at


def test_missing_models_raise_with_setup_hint(tmp_path):
    for cls in (FaceFinder, EmotionModel, FaceSigner):
        with pytest.raises(ModelsMissing, match="setup_models.py"):
            cls(tmp_path)


def test_ear_model_missing_raises(tmp_path):
    with pytest.raises(ModelsMissing):
        EarModel(tmp_path)
    (tmp_path / "ears.onnx").write_bytes(b"")
    with pytest.raises(ModelsMissing, match="ears.json"):
        EarModel(tmp_path)


class _StubSession:
    """Stands in for onnxruntime: logits favour covered for crops brighter than mid grey."""

    def __init__(self):
        self.inputs = []

    def run(self, _outputs, feeds):
        x = feeds["x"]
        self.inputs.append(x)
        bright = x.mean(axis=(1, 2, 3))
        return [np.stack([-bright, bright], axis=1).astype(np.float32)]


def _stub_ear_model():
    model = EarModel.__new__(EarModel)
    model._session = _StubSession()
    model._input = "x"
    model._covered = 1  # labels ["clear", "covered"]
    model._input_px = CONFIG.ear_input_px
    return model


def test_ear_model_runs_one_batch_for_the_sides_with_a_square():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, 320:] = 255  # the left ear's square (right of the frame) is bright
    ears = {"left": (400.0, 240.0), "right": (240.0, 240.0)}
    face = RawFace(Box(0.375, 0.3, 0.25, 0.4), {}, 0.0, 0.0, [(0.0, 0.0)] * 5, ears=ears)
    model = _stub_ear_model()
    covered = model.covered(frame, face, ("left", "right"))
    assert set(covered) == {"left", "right"}
    assert covered["left"] > 0.5 > covered["right"]
    assert all(0 <= p <= 1 for p in covered.values())
    (batch,) = model._session.inputs
    assert batch.shape == (2, 3, CONFIG.ear_input_px, CONFIG.ear_input_px)
    assert batch.dtype == np.float32


def test_ear_model_skips_sides_off_the_frame_and_missing_anchors():
    frame = np.zeros((480, 640, 3), np.uint8)
    face = RawFace(Box(0.0, 0.3, 0.25, 0.4), {}, 0.0, 0.0, [(0.0, 0.0)] * 5,
                   ears={"right": (2.0, 240.0)})
    model = _stub_ear_model()
    assert model.covered(frame, face, ("left", "right")) == {}
    assert model._session.inputs == []


@pytest.mark.models
@pytest.mark.skipif(not (MODELS / "ears.onnx").is_file(), reason="no trained ear model yet")
def test_ear_model_shape():
    frame = _sample()
    face = FaceFinder(MODELS).find(frame)[0]
    sides = sides_to_check(face.yaw_deg)
    covered = EarModel(MODELS).covered(frame, face, sides)
    assert covered and set(covered) <= set(sides)
    assert all(0 <= p <= 1 for p in covered.values())


def test_finder_says_when_only_the_far_model_is_missing(tmp_path):
    (tmp_path / "face_landmarker.task").write_bytes(b"")
    with pytest.raises(ModelsMissing, match="yunet.onnx"):
        FaceFinder(tmp_path)


# -- far faces -----------------------------------------------------------

W, H = 640, 480


def row(x, y, w, h=None):
    """One YuNet result: box, then eyes, nose and mouth corners, then score."""
    h = w if h is None else h
    marks = [x + 0.3 * w, y + 0.4 * h, x + 0.7 * w, y + 0.4 * h, x + 0.5 * w, y + 0.6 * h]
    marks += [x + 0.35 * w, y + 0.8 * h, x + 0.65 * w, y + 0.8 * h]
    return [x, y, w, h, *marks, 0.9]


def near_face(x, y, w, h):
    return RawFace(Box(x / W, y / H, w / W, h / H), {"jawOpen": 0.1}, 3.0, 4.0, [(0.0, 0.0)] * 5)


def far_from(*rows, near=()):
    return _far_faces(np.array(rows, np.float32), list(near), W, H)


def test_no_yunet_result_gives_no_far_faces():
    assert _far_faces(None, [], W, H) == []


def test_far_face_has_a_box_and_points_and_nothing_else():
    found = far_from(row(320, 240, 32))
    assert len(found) == 1
    face = found[0]
    assert face.far is True
    assert face.scores == {}
    assert (face.pitch_deg, face.yaw_deg) == (0, 0)
    assert face.box == pytest.approx(Box(0.5, 0.5, 0.05, 32 / H))
    assert face.points[0] == pytest.approx((320 + 0.3 * 32, 240 + 0.4 * 32))
    assert face.points[4] == pytest.approx((320 + 0.65 * 32, 240 + 0.8 * 32))
    assert len(face.points) == 5


def test_far_box_is_kept_inside_the_frame():
    box = far_from(row(-8, -8, 32))[0].box
    assert (box.x, box.y) == (0, 0)
    assert box.width == pytest.approx(24 / W)
    assert box.height == pytest.approx(24 / H)


def test_a_centre_inside_a_near_box_is_the_same_face():
    assert far_from(row(120, 120, 40), near=[near_face(100, 100, 90, 110)]) == []


def test_a_clear_overlap_with_a_near_box_is_the_same_face():
    # Centre (100, 130) is left of the near box, but 60% of this box is inside it.
    assert far_from(row(80, 110, 40), near=[near_face(96, 100, 90, 110)]) == []


def test_a_slight_overlap_with_a_near_box_is_someone_else():
    assert len(far_from(row(60, 110, 40), near=[near_face(96, 100, 90, 110)])) == 1


def test_a_face_apart_from_the_near_box_is_kept():
    assert len(far_from(row(400, 300, 32), near=[near_face(100, 100, 90, 110)])) == 1


def test_far_face_under_the_minimum_width_is_dropped():
    assert far_from(row(100, 100, CONFIG.far_min_width_px - 1)) == []
    assert len(far_from(row(100, 100, CONFIG.far_min_width_px))) == 1


def test_a_face_too_big_to_be_far_is_left_to_mediapipe():
    """MediaPipe losing a close face must stay a lost face, never someone too far to read."""
    assert far_from(row(100, 100, CONFIG.far_max_width_px + 1)) == []
    assert len(far_from(row(100, 100, CONFIG.far_max_width_px))) == 1


def _frame_with_face(width_px: int) -> np.ndarray:
    """The sample scaled so its face is about width_px wide, on a plain 640x480 frame."""
    image = _sample()
    scale = width_px / 90  # the face in the sample is about 90 px wide
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame = np.full((H, W, 3), (120, 130, 140), np.uint8)
    y, x = 200, 280
    crop = small[: H - y, : W - x]
    frame[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
    return frame


@pytest.mark.models
def test_finder_uses_the_far_score_from_config():
    assert FaceFinder(MODELS)._yunet.getScoreThreshold() == pytest.approx(CONFIG.far_score_at)


@pytest.mark.models
def test_a_30_px_face_is_found_as_far_with_eyes_in_sface_order():
    faces = FaceFinder(MODELS).find(_frame_with_face(30))
    assert len(faces) == 1
    face = faces[0]
    assert face.far is True
    assert 24 <= face.box.width * W <= 36
    right_eye, left_eye = face.points[0], face.points[1]
    assert right_eye[0] < left_eye[0]  # MediaPipe's points for SFace run the same way


@pytest.mark.models
def test_a_face_both_models_find_is_one_near_face():
    faces = FaceFinder(MODELS).find(_frame_with_face(90))
    assert [face.far for face in faces] == [False]


class FakeYunet:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def setInputSize(self, size):
        pass

    def detect(self, frame):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return 1, result


def finder_on(yunet) -> FaceFinder:
    """Far detection alone. A real landmarker held by a traceback hangs pytest at the end."""
    finder = FaceFinder.__new__(FaceFinder)
    finder._yunet = yunet
    finder._far_rows = None
    finder._far_frames = 0
    return finder


BLANK = np.zeros((H, W, 3), np.uint8)
ONE_FAR = np.array([row(320, 240, 32)], np.float32)


def test_far_model_failure_keeps_the_near_faces():
    near = [near_face(100, 100, 90, 110)]
    with pytest.raises(FarDetectFailed) as failure:
        finder_on(FakeYunet(cv2.error("broke")))._find_far(BLANK, near)
    assert failure.value.near == near


def test_far_model_runs_on_schedule_and_its_faces_are_reused_between():
    yunet = FakeYunet(ONE_FAR, None)
    finder = finder_on(yunet)
    found = [finder._find_far(BLANK, []) for _ in range(CONFIG.far_every_frames)]
    assert yunet.calls == 1
    assert [len(faces) for faces in found] == [1] * CONFIG.far_every_frames
    assert [faces[0].reused for faces in found] == [False] + [True] * (CONFIG.far_every_frames - 1)
    assert finder._find_far(BLANK, []) == []
    assert yunet.calls == 2


def test_a_reused_far_face_gives_way_to_a_near_face_found_since():
    finder = finder_on(FakeYunet(ONE_FAR, ONE_FAR))
    assert len(finder._find_far(BLANK, [])) == 1
    assert finder._find_far(BLANK, [near_face(300, 220, 90, 110)]) == []


def test_faces_from_before_a_far_model_failure_are_not_reused():
    finder = finder_on(FakeYunet(ONE_FAR, cv2.error("broke"), None))
    for _ in range(CONFIG.far_every_frames):
        finder._find_far(BLANK, [])
    with pytest.raises(FarDetectFailed):
        finder._find_far(BLANK, [])
    assert finder._find_far(BLANK, []) == []


@pytest.mark.models
def test_a_30_px_face_gets_a_box_and_letter_through_the_real_pipeline(tmp_path):
    start = datetime(2026, 11, 7, 12, 0, 0)
    pipeline = Pipeline(
        FaceFinder(MODELS),
        EmotionModel(MODELS),
        FaceSigner(MODELS),
        SessionFiles(tmp_path, start),
    )
    jpeg = cv2.imencode(".jpg", _frame_with_face(30))[1].tobytes()
    confirming = CONFIG.far_confirm_frames * CONFIG.far_every_frames
    frames = confirming + CONFIG.identity_pending_ms // 100 + 5
    for i in range(frames):
        result = pipeline.process(jpeg, i * 100, start + timedelta(milliseconds=i * 100), 3)
    assert result.lines == ["Person A is too far to read"]
    assert result.count_text == "1 face found"
    assert 24 <= result.people[0]["box"]["width"] * W <= 36
    assert result.people[0]["label"] == "far"
    assert result.alarm is False


def test_padding_uses_real_frame_pixels():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, :] = (10, 20, 30)
    box = Box(0.25, 0.25, 0.5, 0.5)  # x 50..150, y 25..75
    # 10% of 100px wide = 10px, 10% of 50px tall = 5px
    frame[20:80, 40:160] = (200, 100, 50)
    crop = _padded_crop(frame, box)
    assert crop.shape[:2] == (60, 120)
    assert (crop == (200, 100, 50)).all()
    assert (frame[19, 100] == (10, 20, 30)).all()


def test_padding_clamps_at_frame_edges():
    frame = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3).astype(np.uint8)
    crop = _padded_crop(frame, Box(0.0, 0.0, 0.5, 0.5))
    assert crop.shape[:2] == (55, 110)
    assert (crop == frame[0:55, 0:110]).all()
    crop = _padded_crop(frame, Box(0.5, 0.5, 0.5, 0.5))
    assert (crop == frame[45:100, 90:200]).all()


def test_empty_crop_raises_value_error():
    with pytest.raises(ValueError, match="empty crop"):
        _padded_crop(np.zeros((10, 10, 3), np.uint8), Box(0.5, 0.5, 0.0, 0.0))
