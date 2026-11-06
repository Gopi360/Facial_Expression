"""Rolling record of signals, for tuning. No cv2, mediapipe, onnxruntime or fastapi imports here."""
from __future__ import annotations

from .config import CONFIG

WATCHED = ("eye_closure", "look_down", "pitch_delta")


class DebugBuffer:
    """Samples are dicts with t_ms, letter, signals (dict by signal name), label, confidence."""

    def __init__(self) -> None:
        self._items: list[dict] = []

    def add(self, sample: dict) -> None:
        self._items.append(sample)
        t = sample["t_ms"]
        first_kept = next(
            (i for i, s in enumerate(self._items) if t - s["t_ms"] <= CONFIG.debug_buffer_ms),
            0,
        )
        if first_kept > 0:
            self._items = self._items[first_kept:]

    def samples(self) -> list[dict]:
        return list(self._items)

    def recent_ranges(self, letter: str, now_ms: float) -> dict | None:
        recent = [
            s for s in self._items
            if s["letter"] == letter and now_ms - s["t_ms"] <= CONFIG.debug_range_ms
        ]
        if not recent:
            return None
        out = {}
        for name in WATCHED:
            values = [s["signals"][name] for s in recent]
            out[name] = {"min": min(values), "max": max(values)}
        return out
