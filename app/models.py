"""Real model wrappers: MediaPipe face landmarks, HSEmotion ONNX, OpenCV SFace."""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.core.config import CONFIG
from app.core.expressions import EMOTIONS
from app.core.signals import angles_from_matrix
from app.core.types import Box, RawFace

# SFace point order: right eye, left eye, nose, right mouth corner, left mouth corner
_POINT_INDICES = (468, 473, 1, 61, 291)
_MEAN = np.array((0.485, 0.456, 0.406), np.float32)
_STD = np.array((0.229, 0.224, 0.225), np.float32)


class ModelsMissing(Exception):
    """A model file is not in models/."""


def _require(models_dir: Path, name: str) -> str:
    path = Path(models_dir) / name
    if not path.is_file():
        raise ModelsMissing(f"Missing model file {path}. Run: python scripts/setup_models.py")
    return str(path)


class FaceFinder:
    def __init__(self, models_dir: Path):
        path = _require(models_dir, "face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=8,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def find(self, frame_bgr: np.ndarray) -> list[RawFace]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        faces: list[RawFace] = []
        for i, marks in enumerate(result.face_landmarks):
            xs = [m.x for m in marks]
            ys = [m.y for m in marks]
            x0, y0 = min(max(min(xs), 0.0), 1.0), min(max(min(ys), 0.0), 1.0)
            x1, y1 = min(max(max(xs), 0.0), 1.0), min(max(max(ys), 0.0), 1.0)
            scores = {}
            if i < len(result.face_blendshapes):
                scores = {c.category_name: float(c.score) for c in result.face_blendshapes[i]}
            pitch, yaw = 0.0, 0.0
            if i < len(result.facial_transformation_matrixes):
                try:
                    flat = np.asarray(result.facial_transformation_matrixes[i], np.float64)
                    # MediaPipe gives a row-major numpy 4x4; signals expects column-major.
                    pitch, yaw = angles_from_matrix(flat.T.flatten().tolist())
                except (IndexError, ValueError):
                    pitch, yaw = 0.0, 0.0
            points = [(marks[k].x * w, marks[k].y * h) for k in _POINT_INDICES]
            faces.append(RawFace(Box(x0, y0, x1 - x0, y1 - y0), scores, pitch, yaw, points))
        return faces


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class EmotionModel:
    def __init__(self, models_dir: Path):
        path = _require(models_dir, "emotion.onnx")
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def scores(self, face_bgr: np.ndarray) -> dict[str, float]:
        h, w = face_bgr.shape[:2]
        px, py = int(round(w * 0.1)), int(round(h * 0.1))
        # Padding beyond the supplied crop is clamped away: the crop is all we have.
        crop = face_bgr
        crop = cv2.copyMakeBorder(crop, py, py, px, px, cv2.BORDER_REPLICATE)
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        logits = self._session.run(None, {self._input: x})[0][0]
        probs = _softmax(logits.astype(np.float64))
        return {name: float(p) for name, p in zip(EMOTIONS, probs)}


class FaceSigner:
    def __init__(self, models_dir: Path):
        path = _require(models_dir, "sface.onnx")
        self._rec = cv2.FaceRecognizerSF.create(path, "")

    def signature(self, frame_bgr: np.ndarray, face: RawFace) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        b = face.box
        row = [b.x * w, b.y * h, b.width * w, b.height * h]
        for x, y in face.points:
            row += [x, y]
        row.append(1.0)
        aligned = self._rec.alignCrop(frame_bgr, np.array(row, np.float32))
        vec = self._rec.feature(aligned).flatten().astype(np.float64)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
