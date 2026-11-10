"""Session log and snap files. May use cv2; nothing here raises on disk errors."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .core.config import CONFIG
from .core.events import person_red
from .core.types import LogEvent, Person

GREEN_BGR = (94, 197, 34)  # #22c55e, the page's normal box colour
RED_BGR = (68, 68, 239)  # #ef4444, the page's alert box colour
STAMP = "%Y-%m-%d_%H-%M-%S"


def format_line(event: LogEvent) -> str:
    parts = [event.at.strftime("%H:%M:%S"), event.who, event.what]
    if event.confidence is not None:
        parts.append(f"{event.confidence}%")
    if event.snap:
        parts.append(event.snap)
    return " | ".join(parts)


class SessionFiles:
    def __init__(self, root: Path, started: datetime):
        self.root = Path(root)
        self.started = started
        self.log_path = self.root / "logs" / f"session-{started.strftime(STAMP)}.log"

    def write(self, event: LogEvent) -> str | None:
        """Append one line and flush it. Returns a warning on failure, never raises."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(format_line(event) + "\n")
                f.flush()
        except OSError as e:
            return f"Could not write the session log: {e}"
        return None

    def save_snap(
        self, image: np.ndarray, at: datetime, letter: str, reason: str
    ) -> tuple[str | None, str | None]:
        who = "unknown" if letter == "?" else letter  # Windows forbids ? in file names
        rel = f"snaps/{self.started.strftime(STAMP)}_{at.strftime('%H-%M-%S')}_{who}_{reason}.jpg"
        try:
            ok, buf = cv2.imencode(".jpg", image)
            if not ok:
                return None, "Could not save the snap: the image could not be encoded"
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buf.tobytes())
        except (OSError, cv2.error) as e:
            return None, f"Could not save the snap: {e}"
        return rel, None

    def close(self, at: datetime) -> None:
        self.write(LogEvent(at, "System", "session ended"))


def render_snap(frame_bgr: np.ndarray, people: list[Person]) -> np.ndarray:
    """Mirror the frame as the screen showed it, then draw a box and letter per visible person."""
    out = cv2.flip(frame_bgr, 1)
    h, w = out.shape[:2]
    thickness = max(1, round(h * CONFIG.overlay_line_frac))
    scale = h * CONFIG.overlay_letter_frac / 30
    for p in people:
        if not p.visible:
            continue
        colour = RED_BGR if person_red(p.reading.label, p.ears_hidden) else GREEN_BGR
        x = round((1 - p.box.x - p.box.width) * w)
        y = round(p.box.y * h)
        cv2.rectangle(
            out, (x, y), (x + round(p.box.width * w), y + round(p.box.height * h)), colour, thickness
        )
        letter = p.letter or "?"
        ty = y - thickness - 2 if y >= h * CONFIG.overlay_letter_frac + thickness else y + round(
            h * CONFIG.overlay_letter_frac
        )
        cv2.putText(
            out, letter, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, max(1, thickness), cv2.LINE_AA
        )
    return out
