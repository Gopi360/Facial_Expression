"""Red marks, alarm decision and log events. No cv2, mediapipe, onnxruntime or fastapi imports here."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from .types import Label, LogEvent, Reading

RED_LABELS = frozenset({"sleeping", "drowsy"})

LOG_WORDS: dict[Label, str] = {
    "sleeping": "fell asleep",
    "drowsy": "looks drowsy",
    "yawning": "started yawning",
    "angry": "looks angry",
    "laughing": "started laughing",
    "smiling": "started smiling",
    "crying": "looks like crying",
    "neutral": "looks neutral",
}


def is_red(label: Label) -> bool:
    return label in RED_LABELS


def should_sound(labels: Iterable[Label], muted: bool = False) -> bool:
    if muted:
        return False
    return any(is_red(label) for label in labels)


def diff_events(prev: dict[str, Label], nxt: dict[str, Reading], at: datetime) -> list[LogEvent]:
    out: list[LogEvent] = []
    for letter, reading in nxt.items():
        who = f"Person {letter}"
        before = prev.get(letter)
        if before is None:
            out.append(LogEvent(at, who, "joined"))
        elif before != reading.label:
            out.append(LogEvent(at, who, LOG_WORDS[reading.label], reading.confidence))
    for letter in prev:
        if letter not in nxt:
            out.append(LogEvent(at, f"Person {letter}", "left"))
    return out
