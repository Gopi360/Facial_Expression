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
