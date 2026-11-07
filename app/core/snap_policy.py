"""Decides when a red mark earns a snap. No cv2, mediapipe, onnxruntime or fastapi imports here."""
from __future__ import annotations

from dataclasses import dataclass

from .config import CONFIG


@dataclass
class _RedRun:
    visible_ms: float = 0  # red time seen on camera, not counting time hidden
    last_visible_ms: float | None = None  # last visible frame, None while hidden
    lost_ms: float = 0  # frame intervals in which they went out of view, for drop only
    fired: bool = False


class SnapPolicy:
    """True once per red event: a letter seen red on camera for min_red_ms. Re-arms when not red.

    The caller calls update for every present person on every frame, including hidden people
    in their grace period with visible=False, and calls drop when a letter leaves. Hiding keeps
    the event open, so no second snap fires on return, but time hidden does not count towards
    min_red_ms and a snap never fires while hidden, as the frame would not show the person.

    Red time is summed between visible frames, so someone lost around min_red_ms can fall just
    short of it. drop counts the frame interval in which they went out of view, and says
    a snap is owed if that reaches min_red_ms. The caller then saves their last visible frame.
    """

    def __init__(self) -> None:
        self._runs: dict[str, _RedRun] = {}

    def update(self, letter: str, red: bool, now_ms: float, visible: bool = True) -> bool:
        if not red:
            self._runs.pop(letter, None)
            return False
        run = self._runs.setdefault(letter, _RedRun())
        if not visible:
            if run.last_visible_ms is not None:
                interval_ms = now_ms - run.last_visible_ms
                if interval_ms <= CONFIG.open_grace_ms:  # longer is a gap, not a frame interval
                    run.lost_ms += interval_ms
            run.last_visible_ms = None
            return False
        if run.last_visible_ms is not None:
            run.visible_ms += now_ms - run.last_visible_ms
        run.last_visible_ms = now_ms
        if run.fired or run.visible_ms < CONFIG.min_red_ms:
            return False
        run.fired = True
        return True

    def drop(self, letter: str) -> bool:
        """Forget a letter's red run, for when that person leaves. True if a snap is owed."""
        run = self._runs.pop(letter, None)
        if run is None or run.fired:
            return False
        return run.visible_ms + run.lost_ms >= CONFIG.min_red_ms
