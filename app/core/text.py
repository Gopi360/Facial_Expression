"""Panel wording. No cv2, mediapipe, onnxruntime or fastapi imports here."""
from __future__ import annotations

from .types import Label, Person

LINE_WORDS: dict[Label, str] = {
    "sleeping": "is sleeping",
    "drowsy": "is about to sleep",
    "yawning": "is yawning",
    "angry": "is angry",
    "laughing": "is laughing",
    "smiling": "is smiling",
    "crying": "is crying",
    "neutral": "looks neutral",
}


def face_count_text(n: int) -> str:
    if n <= 0:
        return "No faces found"
    return "1 face found" if n == 1 else f"{n} faces found"


def person_line(p: Person) -> str:
    if p.letter is None:
        return "Identifying..."
    return f"Person {p.letter} {LINE_WORDS[p.reading.label]} {round(p.reading.confidence)}%"
