from datetime import datetime

from app.core.events import diff_events, is_red, should_sound
from app.core.types import Reading

AT = datetime(2026, 11, 6, 12, 4, 31)


def r(label, conf=80):
    return Reading(label, conf)


def test_no_change_gives_no_events():
    assert diff_events({"A": "neutral"}, {"A": r("neutral")}, AT) == []


def test_label_change_gives_one_event():
    ev = diff_events({"B": "neutral"}, {"B": r("sleeping", 92)}, AT)
    assert len(ev) == 1
    assert (ev[0].at, ev[0].who, ev[0].what, ev[0].confidence) == (AT, "Person B", "fell asleep", 92)


def test_joined_then_left():
    joined = diff_events({}, {"C": r("sleeping")}, AT)
    assert [(e.who, e.what) for e in joined] == [("Person C", "joined")]
    left = diff_events({"C": "sleeping"}, {}, AT)
    assert [(e.who, e.what) for e in left] == [("Person C", "left")]


def test_started_yawning_wording():
    ev = diff_events({"A": "neutral"}, {"A": r("yawning")}, AT)
    assert ev[0].what == "started yawning"


def test_all_label_wording():
    words = {
        "sleeping": "fell asleep",
        "drowsy": "looks drowsy",
        "angry": "looks angry",
        "laughing": "started laughing",
        "smiling": "started smiling",
        "crying": "looks like crying",
        "neutral": "looks neutral",
    }
    for label, text in words.items():
        assert diff_events({"A": "yawning"}, {"A": r(label)}, AT)[0].what == text


def test_going_far_is_logged_with_no_confidence():
    ev = diff_events({"A": "neutral"}, {"A": r("far", 0)}, AT)
    assert (ev[0].what, ev[0].confidence) == ("is too far to read", None)


def test_coming_back_from_far_logs_the_new_label():
    ev = diff_events({"A": "far"}, {"A": r("neutral", 90)}, AT)
    assert (ev[0].what, ev[0].confidence) == ("looks neutral", 90)


def test_far_is_not_red_and_makes_no_sound():
    assert not is_red("far")
    assert not should_sound(["far"])


def test_is_red():
    assert is_red("sleeping") and is_red("drowsy")
    assert not is_red("yawning") and not is_red("neutral")


def test_should_sound():
    assert should_sound(["neutral", "sleeping"])
    assert should_sound(["drowsy"])
    assert not should_sound(["neutral", "yawning"])
    assert not should_sound([])
    assert not should_sound(["sleeping"], muted=True)
