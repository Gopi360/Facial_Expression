"""Photo mode: saves ear squares to disk so a small classifier can be trained later."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, get_args

import cv2
import numpy as np

from .core.config import CONFIG
from .core.ears import ear_square, sides_to_check
from .core.types import RawFace, Side

logger = logging.getLogger(__name__)

CollectLabel = Literal["both_clear", "left_covered", "right_covered", "both_covered"]
COLLECT_LABELS: tuple[str, ...] = get_args(CollectLabel)

_COVERED_SIDES: dict[str, frozenset[Side]] = {
    "both_clear": frozenset(),
    "left_covered": frozenset({"left"}),
    "right_covered": frozenset({"right"}),
    "both_covered": frozenset({"left", "right"}),
}
FOLDERS = ("clear", "covered")


class EarCollector:
    def __init__(self, root: Path):
        self._dir = root / "dataset" / "ears"
        self._covered: frozenset[Side] | None = None  # None while stopped
        self._run_id = ""
        self._n = 0
        self._last_ms: float | None = None

    def start(self, label: CollectLabel, run_id: str) -> None:
        self._covered = _COVERED_SIDES[label]
        self._run_id = run_id
        self._n = 0
        self._last_ms = None

    def stop(self) -> None:
        self._covered = None

    def offer(self, frame_bgr: np.ndarray, faces: list[RawFace], now_ms: float) -> None:
        if self._covered is None:
            return
        if self._last_ms is not None and now_ms - self._last_ms < CONFIG.collect_interval_ms:
            return
        height, width = frame_bgr.shape[:2]
        saved = False
        for face in faces:
            if face.far or not face.ears:
                continue
            for side in sides_to_check(face.yaw_deg):
                anchor = face.ears.get(side)
                if anchor is None:
                    continue
                square = ear_square(anchor, face.box.width * width, side, width, height)
                if square is None:
                    continue
                x0, y0, size = square
                crop = frame_bgr[y0 : y0 + size, x0 : x0 + size]
                folder = "covered" if side in self._covered else "clear"
                saved = self._save(crop, folder, side) or saved
        if saved:
            self._last_ms = now_ms
            self._n += 1

    def counts(self) -> dict[str, int]:
        return {name: sum(1 for _ in (self._dir / name).glob("*.png")) for name in FOLDERS}

    def _save(self, crop: np.ndarray, folder: str, side: Side) -> bool:
        folder_path = self._dir / folder
        path = folder_path / f"{self._run_id}_{self._n:05d}_{side}.png"
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), crop):
                raise OSError("imwrite returned False")
        except Exception:
            logger.exception("Could not save an ear square to %s", path)
            return False
        return True
