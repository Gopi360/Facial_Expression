from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import CONFIG
from app.core.types import Box
from app.models import EmotionModel, FaceFinder, FaceSigner, ModelsMissing, _padded_crop

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
