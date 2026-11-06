from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import CONFIG
from app.models import EmotionModel, FaceFinder, FaceSigner, ModelsMissing

MODELS = Path(__file__).resolve().parent.parent / "models"


def _sample() -> np.ndarray:
    img = cv2.imread(str(MODELS / "sample_face.png"))
    assert img is not None
    return img


def _crop(frame: np.ndarray, face) -> np.ndarray:
    h, w = frame.shape[:2]
    b = face.box
    x0, y0 = int(b.x * w), int(b.y * h)
    return frame[y0 : y0 + int(b.height * h), x0 : x0 + int(b.width * w)]


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
    scores = EmotionModel(MODELS).scores(_crop(frame, face))
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
def test_flipped_noise_does_not_match():
    frame = _sample()
    face = FaceFinder(MODELS).find(frame)[0]
    signer = FaceSigner(MODELS)
    noise = np.random.default_rng(0).integers(0, 256, frame.shape, dtype=np.uint8)
    assert float(signer.signature(frame, face) @ signer.signature(noise, face)) < CONFIG.identity_match_at


def test_missing_models_raise_with_setup_hint(tmp_path):
    for cls in (FaceFinder, EmotionModel, FaceSigner):
        with pytest.raises(ModelsMissing, match="setup_models.py"):
            cls(tmp_path)
