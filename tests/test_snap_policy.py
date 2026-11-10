from app.core.config import CONFIG
from app.core.snap_policy import SnapPolicy


def test_a_slow_frame_still_counts_as_the_interval_they_were_lost_in():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("A", True, 800)
    p.update("A", True, 800 + CONFIG.frame_gap_limit_ms, visible=False)
    assert p.drop("A")


def test_flicker_under_one_second_takes_no_snap():
    p = SnapPolicy()
    assert not p.update("A", True, 0)
    assert not p.update("A", True, 900)
    assert not p.update("A", False, 950)


def test_red_for_one_second_takes_exactly_one_snap():
    p = SnapPolicy()
    assert not p.update("A", True, 0)
    assert p.update("A", True, 1000)
    for t in range(1100, 61000, 1000):
        assert not p.update("A", True, t)


def test_going_normal_then_red_is_a_new_event():
    p = SnapPolicy()
    p.update("A", True, 0)
    assert p.update("A", True, 1000)
    p.update("A", False, 1100)
    assert not p.update("A", True, 1200)
    assert p.update("A", True, 2200)


def test_two_people_have_separate_snap_events():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("B", True, 500)
    assert p.update("A", True, 1000)
    assert not p.update("B", True, 1000)
    assert p.update("B", True, 1500)
    assert not p.update("A", True, 1600)


def test_drop_rearms_a_letter():
    p = SnapPolicy()
    p.update("A", True, 0)
    assert p.update("A", True, 1000)
    p.drop("A")
    assert not p.update("A", True, 5000)
    assert p.update("A", True, 6000)


def test_drop_forgets_red_start():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("A", True, 900)
    p.drop("A")
    assert not p.update("A", True, 1100)
    assert not p.update("A", True, 1100 + 200)


def test_hidden_red_person_takes_no_snap():
    p = SnapPolicy()
    assert not p.update("A", True, 0)
    assert not p.update("A", True, 300)
    for t in range(400, 3000, 100):
        assert not p.update("A", True, t, visible=False)


def test_time_hidden_does_not_count_towards_the_red_time():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("A", True, 300)
    p.update("A", True, 400, visible=False)
    p.update("A", True, 2000, visible=False)
    assert not p.update("A", True, 2100)
    assert not p.update("A", True, 2700)
    assert p.update("A", True, 2800)


def test_snap_due_while_hidden_waits_until_visible_time_is_reached():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("A", True, 900)
    assert not p.update("A", True, 1500, visible=False)
    assert not p.update("A", True, 1600)
    assert p.update("A", True, 1700)


def test_hiding_after_the_snap_keeps_the_event_open():
    p = SnapPolicy()
    p.update("A", True, 0)
    assert p.update("A", True, 1000)
    assert not p.update("A", True, 1100, visible=False)
    assert not p.update("A", True, 1200)
    assert not p.update("A", True, 5000)


def test_going_normal_while_hidden_rearms():
    p = SnapPolicy()
    p.update("A", True, 0)
    assert p.update("A", True, 1000)
    assert not p.update("A", False, 1100, visible=False)
    assert not p.update("A", True, 1200)
    assert p.update("A", True, 2200)


def test_drop_says_a_snap_is_owed_when_seen_red_for_the_limit_with_none_taken():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("A", True, 900)
    assert not p.update("A", True, 1000, visible=False)
    assert not p.update("A", True, 1100, visible=False)
    assert p.drop("A")


def test_drop_owes_nothing_for_under_the_limit():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("A", True, 300)
    p.update("A", True, 400, visible=False)
    assert not p.drop("A")


def test_drop_owes_nothing_after_a_snap_fired():
    p = SnapPolicy()
    p.update("A", True, 0)
    assert p.update("A", True, 1000)
    p.update("A", True, 1100, visible=False)
    assert not p.drop("A")


def test_drop_owes_nothing_for_an_unknown_letter():
    assert not SnapPolicy().drop("A")


def test_a_long_gap_before_hiding_is_not_red_time_seen():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("A", True, 900)
    p.update("A", True, 1500, visible=False)
    assert not p.drop("A")


def test_time_hidden_owes_nothing():
    p = SnapPolicy()
    p.update("A", True, 0)
    p.update("A", True, 500)
    for t in range(600, 2500, 100):
        p.update("A", True, t, visible=False)
    assert not p.drop("A")
