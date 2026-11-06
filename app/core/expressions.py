"""Maps the trained model's scores to a displayed expression."""
from __future__ import annotations

from app.core.config import CONFIG
from app.core.signals import percent
from app.core.types import Label, Reading, Signals

EMOTIONS = (
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
)  # model output order

_SHOWN_AS: dict[str, Label] = {"Anger": "angry", "Sadness": "crying"}


def expression_reading(scores: dict[str, float] | None, s: Signals) -> Reading:
    """The label for the highest score. With no scores yet, neutral at 100."""
    if not scores:
        return Reading("neutral", 100)

    top = max(EMOTIONS, key=lambda name: scores.get(name, 0.0))
    confidence = percent(scores.get(top, 0.0))
    if top == "Happiness":
        return Reading("laughing" if s.jaw_open >= CONFIG.laugh_jaw_at else "smiling", confidence)
    return Reading(_SHOWN_AS.get(top, "neutral"), confidence)
