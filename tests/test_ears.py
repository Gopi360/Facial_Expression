from dataclasses import replace

from app.core import ears
from app.core.config import CONFIG
from app.core.ears import ear_square, sides_to_check


def test_straight_checks_both():
    assert sides_to_check(0) == ("left", "right")
    assert sides_to_check(19.9) == ("left", "right")


def test_turned_checks_facing_ear(monkeypatch):
    monkeypatch.setattr(ears, "CONFIG", replace(CONFIG, ear_yaw_sign=1))
    assert sides_to_check(25) == ("left",)
    assert sides_to_check(-25) == ("right",)
    monkeypatch.setattr(ears, "CONFIG", replace(CONFIG, ear_yaw_sign=-1))
    assert sides_to_check(25) == ("right",)
    assert sides_to_check(-25) == ("left",)


GEOMETRY = {"ear_offset_frac": 0.08, "ear_square_frac": 0.3}  # fixed here, so tuning config keeps these exact


def test_left_square_is_right_of_anchor(monkeypatch):
    monkeypatch.setattr(ears, "CONFIG", replace(CONFIG, **GEOMETRY))
    x0, y0, size = ear_square((400, 240), 200, "left", 640, 480)
    assert size == 60
    assert x0 + size / 2 == 400 + 0.08 * 200


def test_right_square_is_left_of_anchor(monkeypatch):
    monkeypatch.setattr(ears, "CONFIG", replace(CONFIG, **GEOMETRY))
    x0, y0, size = ear_square((240, 240), 200, "right", 640, 480)
    assert size == 60
    assert x0 + size / 2 == 240 - 16


def test_square_outside_frame_is_none(monkeypatch):
    monkeypatch.setattr(ears, "CONFIG", replace(CONFIG, **GEOMETRY))
    assert ear_square((630, 240), 200, "left", 640, 480) is None


from app.core.ears import EarWatch

CLEAR = {"left": 0.1, "right": 0.1}
COVERED = {"left": 0.9, "right": 0.1}


def _run(watch, checks):
    return [watch.update(c, t) for t, c in checks]


def test_starts_unknown():
    assert EarWatch().update(None, 0) == "unknown"
    assert EarWatch().update({}, 0) == "unknown"


def test_clear_check_is_visible():
    assert EarWatch().update(CLEAR, 0) == "visible"


def test_any_side_covered_counts():
    w = EarWatch()
    hold = CONFIG.ear_hidden_hold_ms
    assert w.update(COVERED, 0) != "hidden"
    assert w.update(COVERED, hold - 1) != "hidden"
    assert w.update(COVERED, hold) == "hidden"


def test_value_at_threshold_is_covered():
    w = EarWatch()
    w.update({"left": CONFIG.ear_covered_at}, 0)
    assert w.update({"left": CONFIG.ear_covered_at}, CONFIG.ear_hidden_hold_ms) == "hidden"


def test_hidden_needs_full_hold():
    w = EarWatch()
    out = _run(w, [(0, COVERED), (333, COVERED), (666, COVERED)])
    assert "hidden" not in out


def test_brief_cover_does_not_hide():
    w = EarWatch()
    out = _run(w, [(0, COVERED), (333, COVERED), (666, CLEAR), (1000, COVERED), (1333, COVERED)])
    assert "hidden" not in out


def _hidden_watch():
    w = EarWatch()
    for t in (0, 333, 666, 1000):
        status = w.update(COVERED, t)
    assert status == "hidden"
    return w


def test_clear_needs_hold():
    w = _hidden_watch()
    t = 1333
    assert w.update(CLEAR, t) == "hidden"
    assert w.update(CLEAR, t + CONFIG.ear_clear_hold_ms - 1) == "hidden"
    assert w.update(CLEAR, t + CONFIG.ear_clear_hold_ms) == "visible"


def test_covered_check_resets_clear_run():
    w = _hidden_watch()
    w.update(CLEAR, 1333)
    w.update(COVERED, 1500)
    assert w.update(CLEAR, 1700) == "hidden"
    assert w.update(CLEAR, 1700 + CONFIG.ear_clear_hold_ms) == "visible"


def test_frames_without_checks_keep_state():
    w = _hidden_watch()
    assert w.update(None, 1100) == "hidden"
    assert w.update({}, 1500) == "hidden"
    v = EarWatch()
    v.update(CLEAR, 0)
    assert v.update(None, 100) == "visible"


def test_grace_drops_to_unknown():
    w = _hidden_watch()
    assert w.update(None, 1000 + CONFIG.ear_unchecked_grace_ms - 1) == "hidden"
    assert w.update(None, 1000 + CONFIG.ear_unchecked_grace_ms) == "unknown"


def test_grace_resets_runs():
    w = EarWatch()
    w.update(COVERED, 0)
    w.update(None, CONFIG.ear_unchecked_grace_ms)
    assert w.update(COVERED, CONFIG.ear_unchecked_grace_ms + 100) != "hidden"
    assert w.update(COVERED, CONFIG.ear_unchecked_grace_ms + 100 + CONFIG.ear_hidden_hold_ms - 1) != "hidden"


def test_side_switch_keeps_state():
    w = EarWatch()
    checks = [(0, {"left": 0.9}), (333, {"left": 0.9, "right": 0.2}), (666, {"left": 0.9}), (1000, {"left": 0.9, "right": 0.2})]
    assert _run(w, checks)[-1] == "hidden"
