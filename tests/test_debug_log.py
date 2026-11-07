from app.core.config import CONFIG
from app.core.debug_log import DebugBuffer

SIGNALS = dict(
    eye_closure=0.1, look_down=0.2, pitch_delta=3.0, smile=0, jaw_open=0,
    brow_down=0, brow_inner_up=0, frown=0,
)


def sample(t_ms, letter="A", **changes):
    return {
        "t_ms": t_ms,
        "letter": letter,
        "signals": {**SIGNALS, **changes},
        "label": "neutral",
        "confidence": 90,
    }


def test_keeps_about_the_last_90_seconds():
    b = DebugBuffer()
    for t in range(0, 100001, 1000):
        b.add(sample(t))
    times = [s["t_ms"] for s in b.samples()]
    assert times[0] == 100000 - CONFIG.debug_buffer_ms
    assert times[-1] == 100000


def test_keeps_every_person_in_the_window():
    b = DebugBuffer()
    b.add(sample(0, "A"))
    b.add(sample(0, "B"))
    assert [s["letter"] for s in b.samples()] == ["A", "B"]


def test_min_and_max_over_last_10_seconds_for_one_person():
    b = DebugBuffer()
    b.add(sample(0, "A", eye_closure=0.99))
    b.add(sample(5000, "A", eye_closure=0.8, look_down=0.1, pitch_delta=-4))
    b.add(sample(8000, "A", eye_closure=0.3, look_down=0.6, pitch_delta=12))
    b.add(sample(8000, "B", eye_closure=0))
    assert b.recent_ranges("A", 12000) == {
        "eye_closure": {"min": 0.3, "max": 0.8},
        "look_down": {"min": 0.1, "max": 0.6},
        "pitch_delta": {"min": -4, "max": 12},
    }


def test_no_ranges_for_person_with_no_recent_samples():
    b = DebugBuffer()
    b.add(sample(0, "A"))
    assert b.recent_ranges("B", 100) is None
    assert b.recent_ranges("A", 20000) is None


def test_ranges_skip_samples_with_no_signals():
    buffer = DebugBuffer()
    buffer.add({"t_ms": 0, "letter": "A", "kind": "identity"})
    assert buffer.recent_ranges("A", 0) is None


def test_buffer_keeps_90_seconds():
    assert CONFIG.debug_buffer_ms == 90000
