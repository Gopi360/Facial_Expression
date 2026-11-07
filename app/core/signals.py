"""Turns raw face scores into signals, judged against each person's own normal."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from app.core.config import CONFIG
from app.core.types import RawFace, Signals


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def percent(strength: float) -> int:
    """0..1 to a whole percent, halves rounding up. Python's round() rounds them to even."""
    return math.floor(clamp01(strength) * 100 + 0.5)


def angles_from_matrix(m: Sequence[float]) -> tuple[float, float]:
    """(pitch, yaw) in degrees from a column-major 4x4 matrix. Positive pitch is tipped forward."""
    pitch = CONFIG.pitch_sign * math.degrees(math.atan2(m[6], m[10]))
    yaw = math.degrees(math.atan2(-m[2], math.hypot(m[0], m[1])))
    return pitch, yaw


def _pair_average(scores: dict[str, float], name: str) -> float:
    return (scores.get(f"{name}Left", 0.0) + scores.get(f"{name}Right", 0.0)) / 2


def raw_eye_closure(face: RawFace) -> float:
    return _pair_average(face.scores, "eyeBlink")


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.floor(fraction * (len(ordered) - 1))]


@dataclass(frozen=True)
class _Sample:
    ms: float
    closure: float
    pitch: float


@dataclass(frozen=True)
class _Normal:
    closure: float
    pitch: float


class Baseline:
    """Learns one person's normal eye openness and head pitch, then judges changes against them."""

    def __init__(self) -> None:
        self._start_ms: float | None = None
        self._closures: list[float] = []
        self._pitches: list[float] = []
        self._recent: list[_Sample] = []  # every sample in the last baseline_max_wait_ms
        self._frozen: _Normal | None = None

    def normal(self) -> _Normal | None:
        """The frozen normal, or None while it is still being learned."""
        return self._frozen

    def is_frontal(self, face: RawFace) -> bool:
        """A view good enough to recognise this person by: facing the camera, eyes open and
        head near their normal pitch. With no normal yet, pitch cannot be judged. Learns nothing.
        """
        frozen = self._frozen
        pitch_ok = (
            frozen is None
            or abs(face.pitch_deg - frozen.pitch) <= CONFIG.frontal_pitch_delta_deg
        )
        return (
            abs(face.yaw_deg) <= CONFIG.frontal_yaw_deg
            and self._closure(raw_eye_closure(face)) < CONFIG.eyes_closed_at
            and pitch_ok
        )

    def _closure(self, raw_closure: float) -> float:
        base_closure = self._frozen.closure if self._frozen is not None else 0.0
        return clamp01((raw_closure - base_closure) / (1 - base_closure))

    def to_signals(self, face: RawFace, now_ms: float) -> Signals:
        raw_closure = raw_eye_closure(face)
        if self._frozen is None:
            self._collect(raw_closure, face.pitch_deg, now_ms)

        frozen = self._frozen
        return Signals(
            eye_closure=self._closure(raw_closure),
            look_down=_pair_average(face.scores, "eyeLookDown"),
            pitch_delta=face.pitch_deg - frozen.pitch if frozen is not None else 0.0,
            smile=_pair_average(face.scores, "mouthSmile"),
            jaw_open=face.scores.get("jawOpen", 0.0),
            brow_down=_pair_average(face.scores, "browDown"),
            brow_inner_up=face.scores.get("browInnerUp", 0.0),
            frown=_pair_average(face.scores, "mouthFrown"),
        )

    def _collect(self, raw_closure: float, pitch_deg: float, now_ms: float) -> None:
        if self._start_ms is None:
            self._start_ms = now_ms
        if raw_closure < CONFIG.baseline_open_eyes_below:
            self._closures.append(raw_closure)
            self._pitches.append(pitch_deg)
        enough_time = now_ms - self._start_ms >= CONFIG.baseline_ms
        if enough_time and len(self._closures) >= CONFIG.baseline_min_samples:
            self._frozen = _Normal(median(self._closures), median(self._pitches))
            return
        self._try_fallback(raw_closure, pitch_deg, now_ms)

    def _try_fallback(self, raw_closure: float, pitch_deg: float, now_ms: float) -> None:
        """Some eyes never read as open, so after a long wait their low percentile is
        taken as their normal. Eyes truly shut stay above the limit and keep waiting,
        or a sleeper would become the new normal and read as awake forever.
        """
        self._recent.append(_Sample(now_ms, raw_closure, pitch_deg))
        self._recent = [s for s in self._recent if now_ms - s.ms <= CONFIG.baseline_max_wait_ms]
        if self._start_ms is None or now_ms - self._start_ms < CONFIG.baseline_max_wait_ms:
            return

        closure = _percentile(
            [s.closure for s in self._recent], CONFIG.baseline_fallback_percentile
        )
        if closure <= CONFIG.baseline_fallback_max_closure:
            self._frozen = _Normal(closure, median(s.pitch for s in self._recent))
