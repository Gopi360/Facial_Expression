from app.core.tracker import Tracker
from app.core.types import Box, RawFace


def face(x: float, y: float, size: float = 0.1) -> RawFace:
    return RawFace(
        box=Box(x, y, size, size), scores={}, pitch_deg=0.0, yaw_deg=0.0, points=[]
    )


def ids(result):
    return [i for i, _ in result]


def test_first_face_gets_one_second_gets_another():
    t = Tracker()
    r = t.update([face(0.1, 0.1), face(0.7, 0.7)], 0)
    assert len(set(ids(r))) == 2


def test_a_face_that_moves_slightly_keeps_its_id():
    t = Tracker()
    first = t.update([face(0.2, 0.2)], 0)
    r = t.update([face(0.22, 0.2)], 100)
    assert ids(r) == ids(first)


def test_two_faces_keep_ids_when_input_order_swaps():
    t = Tracker()
    a, b = ids(t.update([face(0.1, 0.1), face(0.7, 0.7)], 0))
    r = t.update([face(0.7, 0.7), face(0.1, 0.1)], 100)
    assert ids(r) == [b, a]


def test_face_gone_under_2_seconds_keeps_id_on_return():
    t = Tracker()
    first = ids(t.update([face(0.2, 0.2)], 0))
    t.update([], 1000)
    assert first[0] in t.active_ids()
    r = t.update([face(0.2, 0.2)], 1900)
    assert ids(r) == first


def test_face_gone_over_2_seconds_releases_its_id():
    t = Tracker()
    t.update([face(0.2, 0.2)], 0)
    t.update([], 2100)
    assert t.active_ids() == set()


def test_ids_are_never_reused():
    t = Tracker()
    first = ids(t.update([face(0.2, 0.2)], 0))
    t.update([], 2100)
    r = t.update([face(0.2, 0.2)], 2200)
    assert ids(r)[0] not in first


def test_fifth_face_drops_the_smallest():
    t = Tracker()
    faces = [
        face(0.0, 0.0, 0.2),
        face(0.5, 0.0, 0.2),
        face(0.0, 0.5, 0.2),
        face(0.5, 0.5, 0.2),
        face(0.9, 0.9, 0.05),
    ]
    r = t.update(faces, 0)
    assert len(r) == 4
    assert all(f.box.width == 0.2 for _, f in r)
    assert len(t.active_ids()) == 4


def test_face_beyond_max_match_distance_is_new():
    t = Tracker()
    first = ids(t.update([face(0.1, 0.1)], 0))
    r = t.update([face(0.9, 0.9)], 100)
    assert ids(r) != first
