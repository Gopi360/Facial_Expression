"""expression_reading: the trained model's scores to a displayed label."""
from dataclasses import replace

import pytest

from app.core.config import CONFIG
from app.core.expressions import EMOTIONS, expression_reading, is_model_laugh
from app.core.types import Reading, Signals

OPEN = Signals(
    eye_closure=0,
    look_down=0,
    pitch_delta=0,
    smile=0,
    jaw_open=0,
    brow_down=0,
    brow_inner_up=0,
    frown=0,
)
JAW_OPEN = replace(OPEN, jaw_open=CONFIG.laugh_jaw_at)


def scores(**values: float) -> dict[str, float]:
    return {name: values.get(name, 0.0) for name in EMOTIONS}


def test_emotions_are_in_model_output_order():
    assert EMOTIONS == (
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
    )


def test_happiness_with_open_jaw_is_laughing():
    assert expression_reading(scores(Happiness=0.8), JAW_OPEN) == Reading("laughing", 80)


def test_happiness_closed_mouth_is_smiling():
    assert expression_reading(scores(Happiness=0.8), OPEN) == Reading("smiling", 80)


@pytest.mark.parametrize(
    ("emotion", "label"),
    [("Anger", "angry"), ("Sadness", "crying"), ("Neutral", "neutral")],
)
def test_anger_sadness_and_neutral_map(emotion, label):
    assert expression_reading(scores(**{emotion: 0.7}), OPEN) == Reading(label, 70)


@pytest.mark.parametrize("emotion", ["Contempt", "Disgust", "Fear", "Surprise"])
def test_contempt_disgust_fear_surprise_map_to_neutral(emotion):
    assert expression_reading(scores(**{emotion: 0.7}), OPEN) == Reading("neutral", 70)


def test_no_emotion_yet_is_neutral():
    assert expression_reading(None, OPEN) == Reading("neutral", 100)


def test_the_highest_of_the_eight_wins():
    mixed = scores(Anger=0.3, Sadness=0.45, Neutral=0.25)

    assert expression_reading(mixed, OPEN) == Reading("crying", 45)


def test_confidence_rounds_half_up_like_the_typescript_version():
    assert expression_reading(scores(Neutral=0.125), OPEN).confidence == 13


@pytest.mark.parametrize(
    ("emotion", "signals", "expected"),
    [
        (scores(Happiness=0.8), JAW_OPEN, True),
        (scores(Happiness=0.8), OPEN, False),
        (scores(Happiness=0.4, Neutral=0.6), JAW_OPEN, False),
        (None, JAW_OPEN, False),
    ],
)
def test_is_model_laugh_needs_happiness_on_top_and_an_open_jaw(emotion, signals, expected):
    assert is_model_laugh(emotion, signals) is expected
