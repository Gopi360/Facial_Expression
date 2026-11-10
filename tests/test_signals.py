"""Port of tests/signals.test.ts, plus yaw."""
import math

import pytest

from app.core.config import CONFIG
from app.core.signals import Baseline, angles_from_matrix
from app.core.types import Box, RawFace

STEP_MS = 66


def close(value: float):
    """Same tolerance as vitest's toBeCloseTo."""
    return pytest.approx(value, abs=0.005)


def make_face(scores: dict[str, float], pitch_deg: float = 0) -> RawFace:
    return RawFace(
        box=Box(0, 0, 0.2, 0.2), scores=scores, pitch_deg=pitch_deg, yaw_deg=0, points=[]
    )


def face(raw_closure: float, pitch_deg: float = 0) -> RawFace:
    return make_face({"eyeBlinkLeft": raw_closure, "eyeBlinkRight": raw_closure}, pitch_deg)


def feed(baseline: Baseline, item: RawFace, from_ms: int, to_ms: int) -> None:
    for t in range(from_ms, to_ms, STEP_MS):
        baseline.to_signals(item, t)


def x_rotation(degrees: float) -> list[float]:
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    # Column-major 4x4.
    return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]


def y_rotation(degrees: float) -> list[float]:
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    # Column-major 4x4.
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]


# angles_from_matrix


def test_identity_matrix_gives_0_pitch():
    assert angles_from_matrix(x_rotation(0))[0] == close(0)


def test_20_degree_x_rotation_matrix_gives_20_pitch():
    assert angles_from_matrix(x_rotation(20))[0] == close(20)


def test_yaw_from_matrix():
    assert angles_from_matrix(x_rotation(0)) == (close(0), close(0))

    pitch, yaw = angles_from_matrix(y_rotation(30))
    assert abs(yaw) == close(30)
    assert pitch == close(0)


# Baseline


def test_averages_left_and_right_scores():
    signals = Baseline().to_signals(
        make_face(
            {
                "eyeBlinkLeft": 0.2,
                "eyeBlinkRight": 0.4,
                "eyeLookDownLeft": 0.1,
                "eyeLookDownRight": 0.3,
                "mouthSmileLeft": 0.5,
                "mouthSmileRight": 0.7,
                "browDownLeft": 0.6,
                "browDownRight": 0.8,
                "mouthFrownLeft": 0.0,
                "mouthFrownRight": 0.2,
                "jawOpen": 0.45,
                "browInnerUp": 0.55,
            }
        ),
        0,
    )

    assert signals.eye_closure == close(0.3)
    assert signals.look_down == close(0.2)
    assert signals.smile == close(0.6)
    assert signals.brow_down == close(0.7)
    assert signals.frown == close(0.1)
    assert signals.jaw_open == close(0.45)
    assert signals.brow_inner_up == close(0.55)


def test_baseline_raises_closure_scale_for_naturally_narrow_eyes():
    baseline = Baseline()
    feed(baseline, face(0.3), 0, 2500)

    assert baseline.to_signals(face(0.3), 2500).eye_closure == close(0)
    assert baseline.to_signals(face(0.65), 2566).eye_closure == close(0.5)
    assert baseline.to_signals(face(0.1), 2632).eye_closure == 0


def test_arriving_with_closed_eyes_does_not_freeze_a_bad_baseline():
    baseline = Baseline()
    feed(baseline, face(0.9), 0, 3000)

    assert baseline.to_signals(face(0.9), 3000).eye_closure == close(0.9)


def test_waits_for_enough_open_eye_samples_after_the_minimum_time():
    baseline = Baseline()
    feed(baseline, face(0.9), 0, 3000)
    feed(baseline, face(0.2), 3000, 3000 + 8 * STEP_MS)

    assert baseline.to_signals(face(0.2), 3600).eye_closure == close(0.2)  # 9th open sample
    assert baseline.to_signals(face(0.2), 3666).eye_closure == close(0)  # 10th freezes
    assert baseline.to_signals(face(0.6), 3732).eye_closure == close(0.5)


# When eyes never read as open


@pytest.mark.parametrize("resting", [0.52, 0.6])
def test_resting_closure_freezes_on_the_fallback_after_10s(resting):
    baseline = Baseline()
    feed(baseline, face(resting), 0, 9900)
    assert baseline.to_signals(face(resting), 9900).eye_closure == close(resting)

    feed(baseline, face(resting), 9966, 10100)
    assert baseline.to_signals(face(resting), 10100).eye_closure == close(0)
    assert baseline.to_signals(face(1), 10166).eye_closure == close(1)


def test_blinks_do_not_lift_the_fallback_because_it_uses_a_low_percentile():
    baseline = Baseline()
    for t in range(0, 10100, STEP_MS):
        baseline.to_signals(face(1 if t % 1000 < 200 else 0.6), t)

    assert baseline.to_signals(face(0.6), 10166).eye_closure == close(0)


def test_the_fallback_measures_pitch_from_the_samples_it_saw():
    baseline = Baseline()
    feed(baseline, face(0.6, 10), 0, 10100)

    assert baseline.to_signals(face(0.6, 28), 10166).pitch_delta == close(18)


def test_eyes_truly_closed_for_a_minute_are_never_taken_as_the_resting_level():
    baseline = Baseline()
    feed(baseline, face(0.9), 0, 60000)

    assert baseline.to_signals(face(0.9), 60000).eye_closure == close(0.9)


def test_someone_who_arrives_asleep_and_then_opens_their_eyes_gets_a_true_baseline():
    baseline = Baseline()
    feed(baseline, face(0.9), 0, 12000)
    feed(baseline, face(0.2), 12000, 13000)

    assert baseline.to_signals(face(0.2), 13000).eye_closure == close(0)


def test_someone_who_arrives_asleep_and_wakes_to_narrow_eyes_reaches_the_fallback():
    baseline = Baseline()
    feed(baseline, face(0.9), 0, 30000)
    feed(baseline, face(0.6), 30000, 40000)

    assert baseline.to_signals(face(0.6), 40000).eye_closure == close(0)


def test_pitch_delta_is_measured_from_the_persons_normal_pitch():
    baseline = Baseline()

    assert baseline.to_signals(face(0.1, 10), 0).pitch_delta == 0
    feed(baseline, face(0.1, 10), 66, 2500)
    assert baseline.to_signals(face(0.1, 28), 2500).pitch_delta == close(18)


def turned_face(raw_closure: float, pitch_deg: float, yaw_deg: float) -> RawFace:
    found = face(raw_closure, pitch_deg)
    found.yaw_deg = yaw_deg
    return found


def settled_baseline(pitch_deg: float = 10) -> Baseline:
    baseline = Baseline()
    feed(baseline, face(0.1, pitch_deg), 0, 2500)
    return baseline


def test_frontal_view_is_near_the_persons_normal_pitch_not_level():
    baseline = settled_baseline(pitch_deg=10)
    assert baseline.is_frontal(face(0.1, 10))
    assert baseline.is_frontal(face(0.1, 10 + CONFIG.frontal_pitch_delta_deg))
    assert baseline.is_frontal(face(0.1, 10 - CONFIG.frontal_pitch_delta_deg))
    assert not baseline.is_frontal(face(0.1, 10 + CONFIG.frontal_pitch_delta_deg + 1))
    assert not baseline.is_frontal(face(0.1, 10 - CONFIG.frontal_pitch_delta_deg - 1))


def test_frontal_view_needs_open_eyes_against_the_persons_own_normal():
    baseline = Baseline()
    feed(baseline, face(0.4), 0, 2500)
    # (0.55 - 0.4) / 0.6 is 0.25, under eyes_closed_at, though 0.55 is over it raw.
    assert baseline.is_frontal(face(0.55))
    assert not baseline.is_frontal(face(0.9))


def test_frontal_view_still_limits_yaw():
    baseline = settled_baseline(pitch_deg=0)
    assert baseline.is_frontal(turned_face(0.1, 0, CONFIG.frontal_yaw_deg))
    assert not baseline.is_frontal(turned_face(0.1, 0, CONFIG.frontal_yaw_deg + 1))


def test_frontal_view_with_no_baseline_judges_eyes_and_yaw_only():
    baseline = Baseline()
    assert baseline.is_frontal(face(0.1, 40))
    assert not baseline.is_frontal(face(0.9, 0))


def test_judging_a_view_does_not_feed_the_baseline():
    baseline = Baseline()
    for t in range(20):
        baseline.is_frontal(face(0.1, 40))
    assert baseline.normal() is None


def test_a_check_view_allows_more_pitch_than_a_frontal_view():
    baseline = settled_baseline(pitch_deg=10)
    tilted = face(0.1, 10 + CONFIG.frontal_pitch_delta_deg + 5)
    assert not baseline.is_frontal(tilted)
    assert baseline.is_checkable(tilted)
    assert baseline.is_checkable(face(0.1, 10 - CONFIG.identity_check_pitch_delta_deg + 1))


def test_a_check_view_stops_short_of_the_check_pitch_limit():
    baseline = settled_baseline(pitch_deg=0)
    assert baseline.is_checkable(face(0.1, CONFIG.identity_check_pitch_delta_deg - 1))
    assert not baseline.is_checkable(face(0.1, CONFIG.identity_check_pitch_delta_deg))
    assert not baseline.is_checkable(face(0.1, -CONFIG.identity_check_pitch_delta_deg))


def test_a_check_view_needs_open_eyes_and_limits_yaw():
    baseline = settled_baseline(pitch_deg=0)
    assert not baseline.is_checkable(face(0.9))
    assert baseline.is_checkable(turned_face(0.1, 0, CONFIG.frontal_yaw_deg))
    assert not baseline.is_checkable(turned_face(0.1, 0, CONFIG.frontal_yaw_deg + 1))


def test_a_nod_is_not_a_check_view_though_a_level_look_with_heavy_eyes_is():
    baseline = settled_baseline(pitch_deg=0)
    heavy = 0.1 + 0.9 * CONFIG.head_droop_eyes_at + 0.01
    assert not baseline.is_checkable(face(heavy, CONFIG.frontal_pitch_delta_deg + 1))
    assert baseline.is_checkable(face(heavy, CONFIG.frontal_pitch_delta_deg))
    assert baseline.is_checkable(face(0.1, CONFIG.head_droop_deg))


def test_every_frontal_view_can_be_checked():
    baseline = settled_baseline(pitch_deg=0)
    for pitch_deg in (-CONFIG.frontal_pitch_delta_deg, 0, CONFIG.frontal_pitch_delta_deg):
        assert baseline.is_frontal(face(0.1, pitch_deg))
        assert baseline.is_checkable(face(0.1, pitch_deg))


def test_a_check_view_with_no_baseline_judges_eyes_and_yaw_only():
    baseline = Baseline()
    assert baseline.is_checkable(face(0.1, 40))
    assert not baseline.is_checkable(face(0.9, 0))


def test_a_far_view_is_neither_frontal_nor_checkable():
    far_view = RawFace(Box(0, 0, 0.05, 0.05), {}, 0, 0, [], far=True)
    assert not settled_baseline().is_frontal(far_view)
    assert not settled_baseline().is_checkable(far_view)
    assert not Baseline().is_frontal(far_view)
    assert not Baseline().is_checkable(far_view)
