"""Real model wrappers: MediaPipe face landmarks, YuNet far faces, HSEmotion ONNX, OpenCV SFace."""
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
from app.core.types import Box, FarDetectFailed, RawFace

# SFace point order: right eye, left eye, nose, right mouth corner, left mouth corner
_POINT_INDICES = (468, 473, 1, 61, 291)
_EAR_INDICES = {"left": 454, "right": 234}
_MEAN = np.array((0.485, 0.456, 0.406), np.float32)
_STD = np.array((0.229, 0.224, 0.225), np.float32)


class ModelsMissing(Exception):
    """A model file is not in models/."""


def _require(models_dir: Path, name: str) -> str:
    path = Path(models_dir) / name
    if not path.is_file():
        raise ModelsMissing(f"Missing model file {path}. Run: python scripts/setup_models.py")
    return str(path)


def _same_face(far: Box, near: Box) -> bool:
    """YuNet and MediaPipe boxes for one face differ a little, so an exact match never comes."""
    centre_x, centre_y = far.x + far.width / 2, far.y + far.height / 2
    if near.x <= centre_x <= near.x + near.width and near.y <= centre_y <= near.y + near.height:
        return True
    overlap_w = min(far.x + far.width, near.x + near.width) - max(far.x, near.x)
    overlap_h = min(far.y + far.height, near.y + near.height) - max(far.y, near.y)
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    smaller = min(far.width * far.height, near.width * near.height)
    return overlap_w * overlap_h >= CONFIG.far_same_face_overlap * smaller


def _far_faces(
    rows: np.ndarray | None, near: list[RawFace], w: int, h: int, reused: bool = False
) -> list[RawFace]:
    """YuNet results that are not a face MediaPipe already has. Each row is a box, five points
    already in SFace order, and a score. A wide face MediaPipe missed stays missed, as a person
    close to the camera is never too far to read.
    """
    if rows is None:
        return []
    faces: list[RawFace] = []
    for row in rows:
        x, y, width, height = (float(value) for value in row[:4])
        if not CONFIG.far_min_width_px <= width <= CONFIG.far_max_width_px:
            continue
        x0, y0 = max(x, 0.0), max(y, 0.0)
        x1, y1 = min(x + width, float(w)), min(y + height, float(h))
        box = Box(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
        if any(_same_face(box, face.box) for face in near):
            continue
        points = [(float(row[i]), float(row[i + 1])) for i in range(4, 14, 2)]
        faces.append(RawFace(box, {}, 0.0, 0.0, points, far=True, reused=reused))
    return faces


class FaceFinder:
    def __init__(self, models_dir: Path):
        path = _require(models_dir, "face_landmarker.task")
        far_path = _require(models_dir, "yunet.onnx")
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=8,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._yunet = cv2.FaceDetectorYN.create(
            far_path, "", (CONFIG.frame_width, CONFIG.frame_height), CONFIG.far_score_at
        )
        self._far_rows: np.ndarray | None = None  # YuNet's last results, for the frames between
        self._far_frames = 0

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
            ears = {side: (marks[k].x * w, marks[k].y * h) for side, k in _EAR_INDICES.items()}
            faces.append(
                RawFace(Box(x0, y0, x1 - x0, y1 - y0), scores, pitch, yaw, points, ears=ears)
            )
        return faces + self._find_far(frame_bgr, faces)

    def _find_far(self, frame_bgr: np.ndarray, near: list[RawFace]) -> list[RawFace]:
        """Every frame takes a look at the last results against this frame's near faces, but
        YuNet itself runs less often, because on this laptop its time comes out of everyone's
        frame rate, and the sleep rules need that rate.
        """
        h, w = frame_bgr.shape[:2]
        due = self._far_frames % CONFIG.far_every_frames == 0
        self._far_frames += 1
        try:
            if due:
                self._far_rows = None
                self._yunet.setInputSize((w, h))
                self._far_rows = self._yunet.detect(frame_bgr)[1]
            return _far_faces(self._far_rows, near, w, h, reused=not due)
        except Exception as e:
            raise FarDetectFailed(near) from e


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _padded_crop(frame: np.ndarray, box: Box) -> np.ndarray:
    """Crop the box widened by 10% of its size on each side, clamped to the frame."""
    h, w = frame.shape[:2]
    x0 = max(0, int(round((box.x - 0.1 * box.width) * w)))
    y0 = max(0, int(round((box.y - 0.1 * box.height) * h)))
    x1 = min(w, int(round((box.x + 1.1 * box.width) * w)))
    y1 = min(h, int(round((box.y + 1.1 * box.height) * h)))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"Face box {box} gives an empty crop in a {w}x{h} frame")
    return crop


class EmotionModel:
    def __init__(self, models_dir: Path):
        path = _require(models_dir, "emotion.onnx")
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def scores(self, frame_bgr: np.ndarray, box: Box) -> dict[str, float]:
        crop = _padded_crop(frame_bgr, box)
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
