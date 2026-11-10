"""Shared plain data types. No cv2, mediapipe, onnxruntime or fastapi imports here."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

DrowsySign = Literal["slowBlinks", "droopy", "headDroop", "stayedClosed"]

Side = Literal["left", "right"]  # the person's own ear; the same side of the screen when mirrored

Label = Literal[
    "sleeping", "drowsy", "yawning", "angry", "laughing", "smiling", "crying", "neutral", "far"
]


@dataclass(frozen=True)
class Box:
    """Face box as fractions (0..1) of the frame."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class RawFace:
    box: Box
    scores: dict[str, float]
    pitch_deg: float
    yaw_deg: float
    points: list[tuple[float, float]]  # 5 points, pixels
    far: bool = False  # too small to read: a box and points only, with empty scores
    reused: bool = False  # a far face from YuNet's last run, not seen again on this frame
    ears: dict[Side, tuple[float, float]] = field(default_factory=dict)  # face-outline point by each ear, pixels


class FarDetectFailed(Exception):
    """Finding far faces failed. The near faces were found and are carried here."""

    def __init__(self, near: list[RawFace]):
        super().__init__("Far face detection failed")
        self.near = near


@dataclass(frozen=True)
class Signals:
    eye_closure: float
    look_down: float
    pitch_delta: float
    smile: float
    jaw_open: float
    brow_down: float
    brow_inner_up: float
    frown: float


@dataclass(frozen=True)
class Reading:
    label: Label
    confidence: int  # 0..100


@dataclass
class Person:
    letter: str | None
    box: Box
    reading: Reading
    visible: bool
    ears_hidden: bool = False


@dataclass(frozen=True)
class LogEvent:
    at: datetime
    who: str
    what: str
    confidence: int | None = None
    snap: str | None = None
