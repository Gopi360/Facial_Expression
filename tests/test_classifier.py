"""Port of tests/classifier.test.ts, plus yawning.

Scenarios that asserted a label from blendshapes now also pass the model scores
that give that label. The blendshapes stay, because the sleep path reads them.
"""
from collections.abc import Callable
from dataclasses import replace

import pytest

from app.core.classifier import PersonClassifier
from app.core.config import CONFIG
from app.core.expressions import EMOTIONS
from app.core.types import Reading, Signals

STEP_MS = 66

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
CLOSED = replace(OPEN, eye_closure=1)
LAUGH = replace(OPEN, smile=0.8, jaw_open=0.5)
MODEL_LAUGH = replace(OPEN, smile=0.45, jaw_open=0.5)  # under strong_smile_at
ANGRY = replace(OPEN, brow_down=0.7)
CRYING = replace(OPEN, brow_inner_up=0.6, frown=0.5)
DROOPY = replace(OPEN, eye_closure=0.28)
YAWN = replace(OPEN, jaw_open=0.8)

Emotion = dict[str, float] | None


def scores(**values: float) -> dict[str, float]:
    return {name: values.get(name, 0.0) for name in EMOTIONS}


HAPPY = scores(Happiness=0.8)
ANGER = scores(Anger=0.7)
SAD = scores(Sadness=0.55)
CALM = scores(Neutral=0.75)


def run(
    classifier: PersonClassifier,
    signals: Signals,
    from_ms: int,
    to_ms: int,
    delay_sec: float = 3,
    emotion: Emotion = None,
) -> Reading:
    """Steps 66ms from from_ms and always finishes on an update at exactly to_ms."""
    for t in range(from_ms, to_ms, STEP_MS):
        classifier.update(signals, t, delay_sec, emotion)
    return classifier.update(signals, to_ms, delay_sec, emotion)


def labels_shown(
    classifier: PersonClassifier,
    signals_at: Callable[[int], Signals],
    from_ms: int,
    to_ms: int,
    delay_sec: float = 3,
    emotion_at: Callable[[int], Emotion] = lambda t: None,
) -> set[str]:
    """Every label shown from from_ms to to_ms, for "never shows" checks."""
    return {
        classifier.update(signals_at(t), t, delay_sec, emotion_at(t)).label
        for t in range(from_ms, to_ms + 1, STEP_MS)
    }


# Sleeping


def test_eyes_closed_3_seconds_gives_sleeping():
    classifier = PersonClassifier()

    assert run(classifier, CLOSED, 0, 2500).label != "sleeping"
    assert run(classifier, CLOSED, 2500, 3000) == Reading("sleeping", 100)


def test_delay_setting_of_1_gives_sleeping_after_1_second():
    classifier = PersonClassifier()

    assert run(classifier, CLOSED, 0, 900, 1).label != "sleeping"
    assert run(classifier, CLOSED, 900, 1000, 1).label == "sleeping"


def test_a_brief_open_eye_glitch_under_200ms_does_not_reset_the_count():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 2000)
    run(classifier, OPEN, 2066, 2200)

    assert run(classifier, CLOSED, 2266, 3000).label == "sleeping"


def test_opening_eyes_for_longer_resets_the_count():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 2000)
    run(classifier, OPEN, 2066, 2400)

    assert run(classifier, CLOSED, 2466, 4000).label != "sleeping"
    assert run(classifier, CLOSED, 4000, 5466).label == "sleeping"


def test_smile_with_closed_eyes_stays_laughing_at_4_seconds_and_becomes_sleeping_at_5():
    classifier = PersonClassifier()
    squeezed = replace(LAUGH, eye_closure=1)

    assert run(classifier, squeezed, 0, 4000, emotion=HAPPY).label == "laughing"
    assert run(classifier, squeezed, 4000, 4900, emotion=HAPPY).label == "laughing"
    assert run(classifier, squeezed, 4900, 5000, emotion=HAPPY).label == "sleeping"


def test_jaw_open_with_closed_eyes_still_becomes_sleeping_at_3_seconds():
    classifier = PersonClassifier()

    assert run(classifier, replace(CLOSED, jaw_open=0.6), 0, 3000).label == "sleeping"


def test_angry_with_closed_eyes_becomes_sleeping_at_5_seconds():
    classifier = PersonClassifier()
    angry_closed = replace(ANGRY, eye_closure=1)

    assert run(classifier, angry_closed, 0, 4000, emotion=ANGER).label == "angry"
    assert run(classifier, angry_closed, 4000, 5000, emotion=ANGER).label == "sleeping"


def test_a_strong_expression_waits_for_the_longer_of_the_delay_and_the_5_second_limit():
    classifier = PersonClassifier()
    squeezed = replace(LAUGH, eye_closure=1)

    assert run(classifier, squeezed, 0, 6000, 7, HAPPY).label == "laughing"
    assert run(classifier, squeezed, 6000, 7000, 7, HAPPY).label == "sleeping"


def test_closed_eyes_at_delay_7_show_neither_sleeping_nor_drowsy_until_7_seconds():
    classifier = PersonClassifier()
    labels = labels_shown(classifier, lambda t: CLOSED, 0, 6900, 7)

    assert "sleeping" not in labels
    assert "drowsy" not in labels
    assert run(classifier, CLOSED, 6900, 7000, 7).label == "sleeping"


def test_waking_from_sleeping_does_not_pass_through_drowsy():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 3500)
    labels = labels_shown(classifier, lambda t: OPEN, 3566, 7000)

    assert "drowsy" not in labels
    assert run(classifier, OPEN, 7000, 7066).label == "neutral"


def test_waking_keeps_the_last_closed_confidence_until_the_label_changes():
    classifier = PersonClassifier()
    run(classifier, replace(OPEN, eye_closure=0.9), 0, 3500)

    for t in range(3566, 4000, STEP_MS):
        assert classifier.update(OPEN, t, 3, None) == Reading("sleeping", 90)


def test_a_one_frame_smile_dip_with_squeezed_eyes_stays_out_of_sleeping_before_5_seconds():
    squeezed = replace(LAUGH, eye_closure=1)
    dip = replace(squeezed, smile=0.45)
    classifier = PersonClassifier()
    labels = labels_shown(classifier, lambda t: dip if t == 3300 else squeezed, 0, 4950)

    assert "sleeping" not in labels
    assert "drowsy" not in labels
    assert run(classifier, squeezed, 4950, 5000).label == "sleeping"


def test_a_smile_gone_for_500ms_no_longer_holds_sleeping_back():
    classifier = PersonClassifier()
    run(classifier, replace(LAUGH, eye_closure=1), 0, 3300)

    assert run(classifier, CLOSED, 3366, 3700).label != "sleeping"
    assert run(classifier, CLOSED, 3700, 3800).label == "sleeping"


def test_a_5_second_frame_gap_does_not_count_as_closed_time():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 2000)

    assert classifier.update(CLOSED, 7000, 3, None).label != "sleeping"


def test_a_face_lost_for_1_5_seconds_keeps_its_closed_count():
    """The 2 seconds before the gap persist, and the gap itself adds nothing."""
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 2000)

    assert classifier.update(CLOSED, 3500, 3, None).label != "sleeping"
    assert run(classifier, CLOSED, 3566, 4400).label != "sleeping"
    assert run(classifier, CLOSED, 4400, 4500).label == "sleeping"


def test_a_gap_then_open_eyes_ends_the_closure_without_sleeping():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 1500)

    labels = labels_shown(classifier, lambda t: OPEN, 3300, 5000)
    assert "sleeping" not in labels
    assert "drowsy" not in labels


def test_a_gap_just_before_the_delay_then_open_eyes_never_shows_sleeping():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 2950)

    assert "sleeping" not in labels_shown(classifier, lambda t: OPEN, 4000, 4500)


def test_closed_time_does_not_carry_over_a_closure_a_gap_ended():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 2500)
    classifier.update(OPEN, 3000, 3, None)

    assert run(classifier, CLOSED, 3066, 5900).label != "sleeping"


def test_a_frame_gap_within_the_open_grace_still_counts():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 2800)
    reading = classifier.update(CLOSED, 2800 + CONFIG.open_grace_ms, 3, None)

    assert reading.label == "sleeping"


def test_a_gap_does_not_count_towards_the_droopy_hold():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 2000)
    labels = labels_shown(classifier, lambda t: DROOPY, 4000, 4900)

    assert "drowsy" not in labels


def test_a_gap_does_not_count_towards_the_head_droop_hold():
    classifier = PersonClassifier()
    drooped = replace(OPEN, pitch_delta=CONFIG.head_droop_deg, eye_closure=0.5)
    run(classifier, drooped, 0, 1000)
    labels = labels_shown(classifier, lambda t: drooped, 2400, 2800)

    assert "drowsy" not in labels


# Drowsy


def slow_blink(classifier: PersonClassifier, from_ms: int) -> Reading:
    run(classifier, CLOSED, from_ms, from_ms + 500)
    return run(classifier, OPEN, from_ms + 566, from_ms + 900)


def test_normal_blinks_for_one_minute_never_give_sleeping_or_drowsy():
    labels = labels_shown(
        PersonClassifier(), lambda t: CLOSED if t % 4000 < 200 else OPEN, 0, 60000
    )

    assert "sleeping" not in labels
    assert "drowsy" not in labels


def test_two_slow_blinks_within_10_seconds_give_drowsy():
    classifier = PersonClassifier()

    assert slow_blink(classifier, 0).label != "drowsy"
    run(classifier, OPEN, 900, 5000)
    assert slow_blink(classifier, 5000).label == "drowsy"


def test_two_slow_blinks_further_apart_than_10_seconds_do_not_give_drowsy():
    classifier = PersonClassifier()
    slow_blink(classifier, 0)
    run(classifier, OPEN, 900, 11000)

    assert slow_blink(classifier, 11000).label != "drowsy"


def test_a_sleeping_length_closure_does_not_count_as_a_slow_blink():
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 3500)
    run(classifier, OPEN, 3566, 6500)

    assert slow_blink(classifier, 6500).label != "drowsy"


def test_double_blinks_every_5_seconds_for_one_minute_never_give_drowsy():
    def double_blink(t: int) -> Signals:
        phase = t % 5000
        return CLOSED if phase < 180 or 300 <= phase < 480 else OPEN

    labels = labels_shown(PersonClassifier(), double_blink, 0, 60000)

    assert "drowsy" not in labels
    assert "sleeping" not in labels


def test_a_glitch_inside_a_slow_blink_keeps_its_longest_unbroken_run():
    classifier = PersonClassifier()

    def slow_with_glitch(from_ms: int) -> Reading:
        run(classifier, CLOSED, from_ms, from_ms + 500)
        run(classifier, OPEN, from_ms + 566, from_ms + 632)
        run(classifier, CLOSED, from_ms + 698, from_ms + 800)
        return run(classifier, OPEN, from_ms + 866, from_ms + 1200)

    assert slow_with_glitch(0).label != "drowsy"
    assert slow_with_glitch(1266).label == "drowsy"


def test_drooping_eyes_for_3_seconds_give_drowsy():
    classifier = PersonClassifier()

    assert run(classifier, DROOPY, 0, 2900).label != "drowsy"
    assert run(classifier, DROOPY, 2900, 3000) == Reading("drowsy", 50)


@pytest.mark.parametrize("period_ms", [2000, 2500, 3300, 5000])
def test_droopy_eyes_with_200ms_blinks_reach_drowsy_and_stay_drowsy_for_30s(period_ms):
    classifier = PersonClassifier()
    droopy = replace(OPEN, eye_closure=0.28)

    def blinking(t: int) -> Signals:
        return CLOSED if t % period_ms < 200 else droopy

    assert "drowsy" not in labels_shown(classifier, blinking, 0, 3168)
    # The sign is false on closed frames, so a blink at the end of the hold delays it.
    labels_shown(classifier, blinking, 3234, 3564)
    assert labels_shown(classifier, blinking, 3630, 33630) == {"drowsy"}


def test_a_closed_frame_is_not_itself_the_droopy_sign():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 2970)

    assert run(classifier, CLOSED, 3036, 3170).label != "drowsy"
    assert run(classifier, DROOPY, 3236, 3302).label == "drowsy"


def test_a_closure_of_slow_blink_length_restarts_the_droopy_hold():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 2000)
    run(classifier, CLOSED, 2066, 2600)

    assert run(classifier, DROOPY, 2666, 5600).label != "drowsy"
    assert run(classifier, DROOPY, 5600, 5666).label == "drowsy"


def test_droopy_eyes_that_close_fully_stay_drowsy_until_sleeping_at_the_delay():
    classifier = PersonClassifier()
    assert run(classifier, DROOPY, 0, 3500, 7).label == "drowsy"

    assert labels_shown(classifier, lambda t: CLOSED, 3566, 10500, 7) == {"drowsy"}
    assert run(classifier, CLOSED, 10500, 10566, 7).label == "sleeping"


def test_head_droop_with_half_closed_eyes_for_1_5_seconds_gives_drowsy():
    classifier = PersonClassifier()
    nodding = replace(OPEN, eye_closure=0.5, pitch_delta=20)

    assert run(classifier, nodding, 0, 1400).label != "drowsy"
    assert run(classifier, nodding, 1400, 1500) == Reading("drowsy", 50)


def test_head_droop_with_closed_eyes_gives_drowsy_at_1_5_seconds_whatever_the_delay():
    classifier = PersonClassifier()
    nodded = replace(CLOSED, pitch_delta=20)

    assert run(classifier, nodded, 0, 1400, 7).label != "drowsy"
    assert run(classifier, nodded, 1400, 1500, 7) == Reading("drowsy", 100)


def test_looking_down_with_eyes_open_never_triggers():
    # Eyes read as a little closed when looking down, and the head tilts less than a nod.
    looking_down = replace(
        OPEN, pitch_delta=CONFIG.head_droop_deg - 1, look_down=0.7, eye_closure=0.2
    )
    labels = labels_shown(PersonClassifier(), lambda t: looking_down, 0, 30000)

    assert "sleeping" not in labels
    assert "drowsy" not in labels


def test_stays_until_its_signs_have_been_gone_for_2_seconds():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 3000)

    assert run(classifier, OPEN, 3066, 4900).label == "drowsy"
    assert run(classifier, OPEN, 4900, 5600).label == "neutral"


def test_a_strong_expression_suppresses_drowsy():
    smiling_droopy = replace(OPEN, eye_closure=0.28, smile=0.8)
    labels = labels_shown(PersonClassifier(), lambda t: smiling_droopy, 0, 10000)

    assert "drowsy" not in labels


def test_two_slow_blinks_are_one_event_so_drowsy_ends_2_seconds_after_the_second():
    classifier = PersonClassifier()
    slow_blink(classifier, 0)
    run(classifier, OPEN, 900, 3000)
    assert slow_blink(classifier, 3000).label == "drowsy"

    assert run(classifier, OPEN, 3900, 5500).label == "drowsy"
    assert run(classifier, OPEN, 5500, 6500).label == "neutral"


def test_a_third_slow_blink_after_the_event_needs_a_partner_to_fire_again():
    classifier = PersonClassifier()
    slow_blink(classifier, 0)
    run(classifier, OPEN, 900, 2000)
    slow_blink(classifier, 2000)
    run(classifier, OPEN, 2900, 6000)

    assert slow_blink(classifier, 6000).label != "drowsy"


def test_droopy_eyes_reach_and_keep_drowsy_despite_a_one_frame_brow_spike_every_2_seconds():
    def spiked(t: int) -> Signals:
        return replace(DROOPY, brow_down=0.7) if t % 1980 == 0 and t > 0 else DROOPY

    classifier = PersonClassifier()
    labels_shown(classifier, spiked, 0, 3300)

    assert labels_shown(classifier, spiked, 3366, 60000) == {"drowsy"}


def test_one_strong_frame_while_drowsy_does_not_stop_it():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 3300)
    classifier.update(replace(DROOPY, brow_down=0.7), 3366, 3, None)

    assert labels_shown(classifier, lambda t: DROOPY, 3432, 8000) == {"drowsy"}


def test_a_strong_expression_held_for_500ms_ends_drowsy():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 3300)
    laughing = replace(LAUGH, eye_closure=0.28)

    assert run(classifier, laughing, 3366, 4500, emotion=HAPPY).label == "laughing"


def test_a_laugh_starting_just_before_the_droopy_hold_ends_never_shows_drowsy():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 2800)
    laughing = replace(LAUGH, eye_closure=0.28)

    assert "drowsy" not in labels_shown(classifier, lambda t: laughing, 2866, 6000)


@pytest.mark.parametrize(("emotion", "label"), [(ANGER, "angry"), (CALM, "neutral")])
def test_known_limit_resting_brows_at_the_angry_level_with_droopy_eyes(emotion, label):
    """The blendshape rule protects either way, so with CALM a missed drowsy goes unseen."""
    frowning_droopy = replace(OPEN, eye_closure=0.28, brow_down=0.45)
    classifier = PersonClassifier()
    labels = labels_shown(
        classifier, lambda t: frowning_droopy, 0, 60000, emotion_at=lambda t: emotion
    )

    assert "drowsy" not in labels
    assert run(classifier, frowning_droopy, 60066, 60132, emotion=emotion).label == label


# A laugh the model sees while the blendshape smile stays under strong_smile_at


def test_model_laugh_with_squeezed_eyes_stays_out_of_sleeping_until_the_5_second_limit():
    squeezed = replace(MODEL_LAUGH, eye_closure=1)
    classifier = PersonClassifier()
    labels = labels_shown(classifier, lambda t: squeezed, 0, 4950, emotion_at=lambda t: HAPPY)

    assert "sleeping" not in labels
    assert "drowsy" not in labels
    assert run(classifier, squeezed, 4950, 5000, emotion=HAPPY).label == "sleeping"


def test_model_laugh_with_a_squint_never_gives_drowsy():
    squinting = replace(MODEL_LAUGH, eye_closure=0.28)
    labels = labels_shown(
        PersonClassifier(), lambda t: squinting, 0, 10000, emotion_at=lambda t: HAPPY
    )

    assert labels == {"neutral", "laughing"}


def test_model_laugh_with_repeated_600ms_squeezes_never_gives_drowsy_or_sleeping():
    def squeezing(t: int) -> Signals:
        return replace(MODEL_LAUGH, eye_closure=1 if t % 1000 < 600 else 0)

    labels = labels_shown(
        PersonClassifier(), squeezing, 0, 30000, emotion_at=lambda t: HAPPY
    )

    assert labels == {"neutral", "laughing"}


def test_model_laugh_with_repeated_squeezes_leaves_no_drowsy_behind():
    def squeezing(t: int) -> Signals:
        return replace(MODEL_LAUGH, eye_closure=1 if t % 1000 < 600 else 0) if t < 6000 else OPEN

    labels = labels_shown(
        PersonClassifier(),
        squeezing,
        0,
        9000,
        emotion_at=lambda t: HAPPY if t < 6000 else CALM,
    )

    assert "drowsy" not in labels
    assert "sleeping" not in labels


@pytest.mark.parametrize(
    ("signals", "emotion"),
    [
        pytest.param(CLOSED, HAPPY, id="happiness_with_closed_mouth"),
        pytest.param(
            replace(CLOSED, jaw_open=0.3),
            scores(Neutral=0.6, Happiness=0.4),
            id="open_jaw_but_happiness_is_not_top",
        ),
    ],
)
def test_only_a_model_laugh_holds_sleeping_back(signals, emotion):
    assert run(PersonClassifier(), signals, 0, 3000, emotion=emotion).label == "sleeping"


def test_a_laugh_whose_smile_dips_one_frame_in_five_still_protects_against_drowsy():
    def noisy_laugh(t: int) -> Signals:
        return replace(
            LAUGH,
            smile=0.45 if (t / STEP_MS) % 5 == 0 else 0.8,
            eye_closure=1 if t % 1000 < 600 else 0,
        )

    assert "drowsy" not in labels_shown(PersonClassifier(), noisy_laugh, 0, 30000)


def test_a_laugh_with_repeated_eye_squeezes_leaves_no_drowsy_behind():
    def squeezing(t: int) -> Signals:
        return replace(LAUGH, eye_closure=1 if t % 1000 < 600 else 0) if t < 6000 else OPEN

    assert "drowsy" not in labels_shown(PersonClassifier(), squeezing, 0, 9000)


# Expressions


@pytest.mark.parametrize(
    ("label", "signals", "emotion", "confidence"),
    [
        ("laughing", LAUGH, HAPPY, 80),
        ("smiling", replace(OPEN, smile=0.4), scores(Happiness=0.4), 40),
        ("angry", ANGRY, ANGER, 70),
        ("crying", CRYING, SAD, 55),
        ("neutral", replace(OPEN, smile=0.1, brow_down=0.25, brow_inner_up=0.2), CALM, 75),
    ],
)
def test_label_is_reached_from_its_scores(label, signals, emotion, confidence):
    reading = run(PersonClassifier(), signals, 0, 600, emotion=emotion)

    assert reading == Reading(label, confidence)


def test_a_smile_above_the_angry_limit_is_not_angry():
    classifier = PersonClassifier()
    reading = run(classifier, replace(ANGRY, smile=0.25), 0, 600, emotion=CALM)

    assert reading.label == "neutral"
    # The display now comes from the model, so the rule itself shows on the sleep path.
    assert classifier.debug_info()["strong_held"] is False


# Yawning


def test_yawn_held_1_5_seconds_shows_yawning():
    classifier = PersonClassifier()

    assert run(classifier, YAWN, 0, 1000).label != "yawning"
    # The yawn hold passes at 1.5s, and the normal label hold runs from there.
    assert run(classifier, YAWN, 1000, 1500).label != "yawning"
    assert run(classifier, YAWN, 1500, 1990).label != "yawning"
    assert run(classifier, YAWN, 1990, 2000) == Reading("yawning", 80)


def test_a_mouth_closing_before_1_5_seconds_restarts_the_yawn_hold():
    classifier = PersonClassifier()
    run(classifier, YAWN, 0, 1400)
    classifier.update(OPEN, 1466, 3, None)

    assert "yawning" not in labels_shown(classifier, lambda t: YAWN, 1532, 3000)


def test_wide_mouth_with_smile_is_not_a_yawn():
    wide_smile = replace(YAWN, smile=0.4)
    classifier = PersonClassifier()
    labels = labels_shown(classifier, lambda t: wide_smile, 0, 5000, emotion_at=lambda t: HAPPY)

    assert "yawning" not in labels
    assert run(classifier, wide_smile, 5000, 5066, emotion=HAPPY).label == "laughing"


def test_yawning_outranks_the_trained_expression():
    assert run(PersonClassifier(), YAWN, 0, 2100, emotion=ANGER).label == "yawning"


def test_yawning_with_closed_eyes_still_reaches_sleeping_at_delay():
    classifier = PersonClassifier()
    yawning_closed = replace(YAWN, eye_closure=1)

    assert run(classifier, yawning_closed, 0, 2500).label == "yawning"
    assert run(classifier, yawning_closed, 2500, 2900).label == "yawning"
    assert run(classifier, yawning_closed, 2900, 3000) == Reading("sleeping", 100)


def test_drowsy_outranks_yawning():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 3100)

    assert run(classifier, replace(DROOPY, jaw_open=0.8), 3166, 8000).label == "drowsy"


def test_a_yawn_hold_keeps_counting_while_drowsy_is_shown():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 3000)

    # Drowsy clears 2s after its last sign, by when the yawn has long passed its hold.
    assert run(classifier, YAWN, 3066, 4900).label == "drowsy"
    assert run(classifier, YAWN, 4900, 5600).label == "yawning"


def test_yawning_alone_never_gives_drowsy():
    labels = labels_shown(PersonClassifier(), lambda t: YAWN, 0, 30000)

    assert labels == {"neutral", "yawning"}


# Display


def test_a_label_flicker_under_500ms_is_not_displayed():
    smile = replace(OPEN, smile=0.4)

    def in_flicker(t: int) -> bool:
        return 1000 <= t < 1400

    labels = labels_shown(
        PersonClassifier(),
        lambda t: smile if in_flicker(t) else OPEN,
        0,
        3000,
        emotion_at=lambda t: HAPPY if in_flicker(t) else CALM,
    )

    assert labels == {"neutral"}


def alternating(a, b):
    return lambda t: a if t % 300 < 150 else b


JAW_CROSSING = alternating(replace(LAUGH, jaw_open=0.3), replace(LAUGH, jaw_open=0.2))
ALTERNATING_ROWS = [
    pytest.param(JAW_CROSSING, lambda t: HAPPY, {"laughing", "smiling"}, id="jaw_crossing_laugh"),
    pytest.param(
        alternating(replace(OPEN, smile=0.37), replace(OPEN, smile=0.33)),
        alternating(HAPPY, CALM),
        {"smiling", "neutral"},
        id="smile_around_the_smile_line",
    ),
    pytest.param(
        alternating(replace(OPEN, brow_down=0.42), replace(OPEN, brow_down=0.38)),
        alternating(ANGER, CALM),
        {"angry", "neutral"},
        id="brows_around_the_angry_line",
    ),
]


@pytest.mark.parametrize(("signals_at", "emotion_at", "allowed"), ALTERNATING_ROWS)
def test_two_alternating_labels_leave_sleeping_soon_after_waking(signals_at, emotion_at, allowed):
    classifier = PersonClassifier()
    run(classifier, CLOSED, 0, 3500)
    labels_shown(classifier, signals_at, 3566, 4600, emotion_at=emotion_at)

    assert labels_shown(classifier, signals_at, 4666, 20000, emotion_at=emotion_at) <= allowed


def test_a_jaw_crossing_laugh_leaves_neutral_within_a_second():
    classifier = PersonClassifier()
    labels_shown(classifier, JAW_CROSSING, 0, 1000, emotion_at=lambda t: HAPPY)

    labels = labels_shown(classifier, JAW_CROSSING, 1066, 10000, emotion_at=lambda t: HAPPY)
    assert "neutral" not in labels


def test_a_held_label_keeps_its_last_confidence_until_the_new_label_shows():
    classifier = PersonClassifier()
    run(classifier, ANGRY, 0, 600, emotion=ANGER)

    assert run(classifier, OPEN, 666, 1000, emotion=CALM) == Reading("angry", 70)
    assert run(classifier, OPEN, 1000, 1200, emotion=CALM).label == "neutral"


def test_confidence_stays_an_integer_from_0_to_100():
    classifier = PersonClassifier()
    inputs: list[tuple[Signals, Emotion]] = [
        (OPEN, None),
        (replace(OPEN, smile=0.333), scores(Happiness=0.333)),
        (replace(OPEN, smile=1.2, jaw_open=1.5), scores(Happiness=1.2)),
        (replace(OPEN, brow_down=1.4), scores(Anger=1.4)),
        (replace(OPEN, brow_inner_up=1.3, frown=1.3), scores(Sadness=1.3)),
        (replace(OPEN, eye_closure=1.2), None),
        (replace(OPEN, eye_closure=0.28), None),
        (replace(OPEN, jaw_open=1.5), None),
    ]

    t = 0
    for signals, emotion in inputs:
        end = t + 6000
        while t < end:
            confidence = classifier.update(signals, t, 3, emotion).confidence
            assert isinstance(confidence, int)
            assert 0 <= confidence <= 100
            t += STEP_MS


# Debug info


def test_debug_starts_quiet():
    classifier = PersonClassifier()
    classifier.update(OPEN, 0, 3, None)

    assert classifier.debug_info() == {
        "drowsy_sign": None,
        "strong_held": False,
        "strong_sustained": False,
        "closed_ms": 0,
        "closed_threshold": CONFIG.eyes_closed_at,
    }


def test_debug_names_the_droopy_sign_while_it_is_driving_drowsy_then_lets_go():
    classifier = PersonClassifier()
    run(classifier, DROOPY, 0, 3100)
    assert classifier.debug_info()["drowsy_sign"] == "droopy"

    run(classifier, OPEN, 3166, 5300)
    assert classifier.debug_info()["drowsy_sign"] is None


def test_debug_names_head_droop():
    classifier = PersonClassifier()
    run(classifier, replace(OPEN, eye_closure=0.5, pitch_delta=20), 0, 1600)

    assert classifier.debug_info()["drowsy_sign"] == "headDroop"


def test_debug_keeps_naming_slow_blinks_after_the_one_frame_event():
    classifier = PersonClassifier()
    for from_ms in (0, 1000):
        run(classifier, CLOSED, from_ms, from_ms + 500)
        run(classifier, OPEN, from_ms + 566, from_ms + 900)

    assert classifier.debug_info()["drowsy_sign"] == "slowBlinks"


def test_debug_tells_a_lone_strong_frame_from_a_sustained_one():
    classifier = PersonClassifier()
    classifier.update(ANGRY, 0, 3, None)
    info = classifier.debug_info()
    assert (info["strong_held"], info["strong_sustained"]) == (True, False)

    run(classifier, ANGRY, 66, 600)
    info = classifier.debug_info()
    assert (info["strong_held"], info["strong_sustained"]) == (True, True)


def test_debug_does_not_change_what_update_returns():
    reading = PersonClassifier().update(OPEN, 0, 3, None)

    assert isinstance(reading, Reading)
    assert set(vars(reading)) == {"label", "confidence"}


# Debug numbers


def test_debug_info_gives_the_closed_count_and_the_threshold_in_force():
    classifier = PersonClassifier()
    classifier.update(OPEN, 0, 3, None)
    assert classifier.debug_info()["closed_ms"] == 0
    assert classifier.debug_info()["closed_threshold"] == CONFIG.eyes_closed_at
    run(classifier, replace(CLOSED, look_down=0.9), 66, 1066)
    assert classifier.debug_info()["closed_ms"] == 1000
    assert classifier.debug_info()["closed_threshold"] == CONFIG.eyes_closed_at


# Real readings from the owner's on-camera test, wearing glasses (2026-11-07), against
# their baseline of closure 0.11 and pitch -8.8. "Looking down" sits high even at rest.
OWNER_OPEN = replace(OPEN, eye_closure=0.0, look_down=0.61, pitch_delta=0)
OWNER_CLOSED = replace(OPEN, eye_closure=0.48, look_down=0.77, pitch_delta=0.5)
OWNER_NOD = replace(OPEN, eye_closure=0.13, look_down=0.49, pitch_delta=21.9)
OWNER_KEYBOARD = replace(OPEN, eye_closure=0.135, look_down=0.65, pitch_delta=10.8)
OWNER_HALF_CLOSED = replace(OPEN, eye_closure=0.095, look_down=0.59, pitch_delta=7)


def test_owner_closed_eyes_behind_glasses_reach_sleeping_at_the_delay():
    classifier = PersonClassifier()
    run(classifier, OWNER_OPEN, 0, 3000)
    assert run(classifier, OWNER_CLOSED, 3066, 5900).label != "sleeping"
    assert run(classifier, OWNER_CLOSED, 5966, 6200).label == "sleeping"


def test_owner_nod_gives_drowsy():
    classifier = PersonClassifier()
    run(classifier, OWNER_OPEN, 0, 3000)
    assert run(classifier, OWNER_NOD, 3066, 5000).label == "drowsy"


def test_owner_open_eyes_never_alarm():
    classifier = PersonClassifier()
    assert run(classifier, OWNER_OPEN, 0, 30000).label == "neutral"


@pytest.mark.parametrize("pose", [OWNER_KEYBOARD, OWNER_HALF_CLOSED])
def test_owner_looking_at_the_keyboard_or_half_closed_never_alarms(pose):
    classifier = PersonClassifier()
    run(classifier, OWNER_OPEN, 0, 3000)
    labels = {run(classifier, pose, ms, ms + 66).label for ms in range(3066, 33000, 132)}
    assert labels == {"neutral"}
