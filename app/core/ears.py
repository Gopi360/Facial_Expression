"""Where each ear is and the square to look at. No cv2, mediapipe, onnxruntime or fastapi imports."""
from __future__ import annotations

from typing import Literal

from app.core.config import CONFIG
from app.core.types import Side


def sides_to_check(yaw_deg: float) -> tuple[Side, ...]:
    """Both ears when facing the camera, only the ear turned towards it otherwise."""
    if abs(yaw_deg) < CONFIG.ear_both_yaw_deg:
        return ("left", "right")
    return ("left",) if yaw_deg * CONFIG.ear_yaw_sign > 0 else ("right",)


def ear_square(
    anchor: tuple[float, float], face_width_px: float, side: Side, frame_w: int, frame_h: int
) -> tuple[int, int, int] | None:
    """(x0, y0, size) of the square around an ear, or None if it is not wholly in the frame.

    The left ear is on the right of the unmirrored camera frame, so its square is pushed +x.
    """
    size = round(CONFIG.ear_square_frac * face_width_px)
    shift = CONFIG.ear_offset_frac * face_width_px * (1 if side == "left" else -1)
    cx = anchor[0] + shift
    x0 = round(cx - size / 2)
    y0 = round(anchor[1] - size / 2)
    if size <= 0 or x0 < 0 or y0 < 0 or x0 + size > frame_w or y0 + size > frame_h:
        return None
    return (x0, y0, size)


EarStatus = Literal["unknown", "visible", "hidden"]


class EarWatch:
    """Steady ear state for one person, built from occasional ear checks."""

    def __init__(self) -> None:
        self._status: EarStatus = "unknown"
        self._run_start: float | None = None  # first check of the current covered run, or of the clear run while hidden
        self._last_check: float | None = None

    def update(self, covered: dict[Side, float] | None, now_ms: float) -> EarStatus:
        """Feed one frame. `covered` maps each checked side to its covered probability; None or {} is no check."""
        if self._last_check is not None and now_ms - self._last_check >= CONFIG.ear_unchecked_grace_ms:
            self._status = "unknown"  # grace applies on every call, so a person who left starts fresh
            self._run_start = None
        if not covered:
            return self._status

        self._last_check = now_ms
        is_covered = any(p >= CONFIG.ear_covered_at for p in covered.values())
        if is_covered:
            if self._status == "hidden":
                self._run_start = None  # a covered check cancels a clear run
            else:
                if self._run_start is None:
                    self._run_start = now_ms
                if now_ms - self._run_start >= CONFIG.ear_hidden_hold_ms:
                    self._status = "hidden"
                    self._run_start = None
        elif self._status == "hidden":
            if self._run_start is None:
                self._run_start = now_ms
            if now_ms - self._run_start >= CONFIG.ear_clear_hold_ms:
                self._status = "visible"
                self._run_start = None
        else:
            self._status = "visible"
            self._run_start = None  # a clear check cancels a covered run
        return self._status
