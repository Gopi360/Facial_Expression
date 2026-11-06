import numpy as np

from app.core.config import CONFIG
from app.core.identity import Identity


def vec(i: int, n: int = 40) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


def seen(ident: Identity, ids: set[int], now: float, active: set[int] | None = None):
    ident.sync(ids, active if active is not None else ids, now)


def test_first_person_gets_a():
    ident = Identity()
    seen(ident, {1}, 0)
    assert ident.letter_for(1) is None
    ident.add_signature(1, vec(0), True, 0)
    assert ident.letter_for(1) == "A"
    assert ident.people_count() == 1


def test_returning_person_gets_old_letter_back():
    ident = Identity()
    seen(ident, {1}, 0)
    ident.add_signature(1, vec(0), True, 0)
    seen(ident, set(), 3000, active=set())
    seen(ident, {2}, 3100)
    ident.add_signature(2, vec(0), True, 3100)
    assert ident.letter_for(2) == "A"


def test_return_within_grace_at_new_position_keeps_letter():
    ident = Identity()
    seen(ident, {1}, 0)
    ident.add_signature(1, vec(0), True, 0)
    # track 1 hidden but still active, track 2 appears elsewhere
    ident.sync({2}, {1, 2}, 500)
    ident.add_signature(2, vec(0), True, 500)
    assert ident.letter_for(2) == "A"
    assert ident.letter_for(1) is None


def test_different_person_gets_next_letter():
    ident = Identity()
    seen(ident, {1}, 0)
    ident.add_signature(1, vec(0), True, 0)
    seen(ident, {1, 2}, 100)
    ident.add_signature(2, vec(1), True, 100)
    assert ident.letter_for(2) == "B"
    assert ident.people_count() == 2


def test_two_similar_faces_in_view_never_share_a_letter():
    ident = Identity()
    seen(ident, {1, 2}, 0)
    ident.add_signature(1, vec(0), True, 0)
    ident.add_signature(2, vec(0), True, 0)
    assert ident.letter_for(1) != ident.letter_for(2)
    assert ident.letter_for(2) == "B"


def test_no_signature_for_2_seconds_gets_next_letter():
    ident = Identity()
    seen(ident, {1}, 0)
    seen(ident, {1}, CONFIG.identity_pending_ms - 1)
    assert ident.letter_for(1) is None
    seen(ident, {1}, CONFIG.identity_pending_ms)
    assert ident.letter_for(1) == "A"


def test_after_z_letter_is_question_mark():
    ident = Identity()
    for i in range(27):
        seen(ident, set(range(i + 1)), i)
        ident.add_signature(i, vec(i), True, i)
    assert ident.letter_for(25) == "Z"
    assert ident.letter_for(26) == "?"


def test_people_after_z_have_separate_state_keys():
    ident = Identity()
    for i in range(28):
        ident.sync(set(range(i + 1)), set(range(i + 1)), i)
        ident.add_signature(i, vec(i), True, i)
    ident.sync({26, 27}, set(range(28)), 30)
    assert ident.letter_for(26) == "?" and ident.letter_for(27) == "?"
    assert ident.state_key(26) != ident.state_key(27)
    assert ident.state_key(26) == "?26"
    assert ident.state_key(25) == "Z"
    ident.sync({99}, {99}, 40)
    assert ident.state_key(99) is None


def test_side_view_signature_is_not_stored():
    ident = Identity()
    seen(ident, {1}, 0)
    ident.add_signature(1, vec(0), False, 0)
    assert ident.letter_for(1) == "A"  # first signature still decides a letter
    seen(ident, set(), 3000, active=set())
    seen(ident, {2}, 3100)
    ident.add_signature(2, vec(0), True, 3100)
    assert ident.letter_for(2) == "B"  # nothing stored for A


def test_stored_signatures_are_capped():
    ident = Identity()
    seen(ident, {1}, 0)
    first = (vec(0) + vec(1)) / np.sqrt(2)
    ident.add_signature(1, first, True, 0)
    for k in range(CONFIG.identity_max_signatures + 2):
        ident.add_signature(1, vec(1) + 0.05 * vec(k + 2), True, k + 1)
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    ident.add_signature(2, vec(0), True, 9100)  # only matched the dropped first
    assert ident.letter_for(2) != "A"
    seen(ident, set(), 20000, active=set())
    seen(ident, {3}, 20100)
    ident.add_signature(3, vec(1) + 0.05 * vec(2), True, 20100)
    assert ident.letter_for(3) == "A"


def test_lettered_track_never_flips_to_another_letter():
    ident = Identity()
    seen(ident, {1}, 0)
    ident.add_signature(1, vec(0), True, 0)
    seen(ident, {1, 2}, 100)
    ident.add_signature(2, vec(1), True, 100)  # B
    ident.sync({2}, {1, 2}, 200)  # A hidden
    ident.add_signature(2, vec(0), True, 1500)  # looks like A
    assert ident.letter_for(2) == "B"
    assert len(ident._signatures["B"]) == 1


def test_old_stripped_track_does_not_become_a_phantom_person():
    ident = Identity()
    seen(ident, {1}, 0)
    ident.add_signature(1, vec(0), True, 0)
    seen(ident, {1}, 5000)
    ident.sync({2}, {1, 2}, 5500)
    ident.add_signature(2, vec(0), True, 5500)
    assert ident.letter_for(2) == "A"
    for t in (5600, 6000, 9000):
        ident.sync({2}, {1, 2}, t)
    assert ident.letter_for(1) is None
    assert ident.people_count() == 1


def test_stripped_track_can_rebind_to_a_known_person():
    ident = Identity()
    seen(ident, {1}, 0)
    ident.add_signature(1, vec(0), True, 0)
    seen(ident, {1, 2}, 100)
    ident.add_signature(2, vec(1), True, 100)  # B
    ident.sync({2}, {1, 2}, 200)  # A hidden
    ident.sync({3}, {1, 2, 3}, 5000)  # B hidden too
    ident.add_signature(3, vec(0), True, 5000)  # A takes track 3, track 1 stripped
    ident.sync({1}, {1, 2, 3}, 5100)  # track 1 comes back into view as B
    ident.add_signature(1, vec(1), True, 5100)
    assert ident.letter_for(1) == "B"
    assert ident.letter_for(3) == "A"
    assert ident.people_count() == 2


def test_stranger_on_a_lettered_track_does_not_fill_its_store():
    ident = Identity()
    seen(ident, {1}, 0)
    ident.add_signature(1, vec(0), True, 0)
    ident.add_signature(1, vec(1), True, 1500)  # stranger inherits the track
    assert len(ident._signatures["A"]) == 1
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    ident.add_signature(2, vec(1), True, 9100)
    assert ident.letter_for(2) == "B"


def test_crossing_people_do_not_swap_stores():
    ident = Identity()
    seen(ident, {1, 2}, 0)
    ident.add_signature(1, vec(0), True, 0)  # A
    ident.add_signature(2, vec(1), True, 0)  # B
    ident.add_signature(1, vec(1), True, 1500)  # tracks swapped while crossing
    ident.add_signature(2, vec(0), True, 1500)
    assert len(ident._signatures["A"]) == 1
    assert len(ident._signatures["B"]) == 1
    seen(ident, set(), 9000, active=set())
    seen(ident, {3}, 9100)
    ident.add_signature(3, vec(0), True, 9100)
    assert ident.letter_for(3) == "A"


def test_needs_signature_rules():
    ident = Identity()
    seen(ident, {1}, 0)
    assert ident.needs_signature(1, 0)
    ident.add_signature(1, vec(0), True, 0)
    assert not ident.needs_signature(1, 500)
    assert ident.needs_signature(1, CONFIG.identity_interval_ms)
