"""Where each ear is and the square to look at. No cv2, mediapipe, onnxruntime or fastapi imports."""
from __future__ import annotations

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
