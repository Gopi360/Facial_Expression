"""Decides when a red mark earns a snap. No cv2, mediapipe, onnxruntime or fastapi imports here."""
from __future__ import annotations

from .config import CONFIG


class SnapPolicy:
    """True once per red event: a letter red continuously for min_red_ms. Re-arms when not red."""

    def __init__(self) -> None:
        self._red_since: dict[str, float] = {}
        self._fired: set[str] = set()

    def update(self, letter: str, red: bool, now_ms: float) -> bool:
        if not red:
            self._red_since.pop(letter, None)
            self._fired.discard(letter)
            return False
        since = self._red_since.setdefault(letter, now_ms)
        if letter in self._fired or now_ms - since < CONFIG.min_red_ms:
            return False
        self._fired.add(letter)
        return True
