from app.core.snap_policy import SnapPolicy


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
