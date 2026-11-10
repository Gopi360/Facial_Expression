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


def _top_emotion(scores: dict[str, float]) -> str:
    return max(EMOTIONS, key=lambda name: scores.get(name, 0.0))


def is_model_laugh(scores: dict[str, float] | None, s: Signals) -> bool:
    """Happiness on top with the jaw open. Decides the label and protects the sleep path."""
    if not scores:
        return False
    return _top_emotion(scores) == "Happiness" and s.jaw_open >= CONFIG.laugh_jaw_at


def expression_reading(scores: dict[str, float] | None, s: Signals) -> Reading:
    """The label for the highest score. With no scores yet, neutral at 100."""
    if not scores:
        return Reading("neutral", 100)

    top = _top_emotion(scores)
    confidence = percent(scores.get(top, 0.0))
    if is_model_laugh(scores, s):
        return Reading("laughing", confidence)
    if top == "Happiness":
        return Reading("smiling", confidence)
    return Reading(_SHOWN_AS.get(top, "neutral"), confidence)
