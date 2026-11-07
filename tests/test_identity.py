import numpy as np

from app.core.config import CONFIG
from app.core.identity import Identity


def vec(i: int, n: int = 40) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


def near(i: int, j: int, lean: float = 0.3) -> np.ndarray:
    """Mostly person i, with a little of something else."""
    return vec(i) + lean * vec(j)


def blend(i: int, j: int, turn: float) -> np.ndarray:
    """turn 0 is all of i, turn 1 is all of j, moving smoothly between."""
    angle = turn * np.pi / 2
    return np.cos(angle) * vec(i) + np.sin(angle) * vec(j)


def lettered(ident: Identity, track: int, i: int, now: float = 0) -> None:
    """Puts a person on a track and fills their anchors with clean frontal views."""
    ident.sync({track}, {track}, now)
    for k in range(CONFIG.identity_anchor_signatures):
        ident.add_signature(track, vec(i), True, now + k)


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
    assert ident.letter_for(1) is None  # a side view cannot create a letter
    seen(ident, {1}, CONFIG.identity_pending_ms)
    assert ident.letter_for(1) == "A"
    ident.add_signature(1, vec(0), False, CONFIG.identity_pending_ms)
    seen(ident, set(), 3000, active=set())
    seen(ident, {2}, 3100)
    ident.add_signature(2, vec(0), True, 3100)
    assert ident.letter_for(2) == "B"  # nothing stored for A


def test_rolling_signatures_are_capped_and_anchors_stay():
    ident = Identity()
    seen(ident, {1}, 0)
    for k in range(CONFIG.identity_anchor_signatures):
        ident.add_signature(1, near(0, k + 1), True, k)
    first_rolling = near(0, 20)
    ident.add_signature(1, first_rolling, True, 10)
    for k in range(CONFIG.identity_max_signatures):
        ident.add_signature(1, near(0, k + 21), True, 11 + k)
    store = ident._signatures["A"]
    assert len(store.anchors) == CONFIG.identity_anchor_signatures
    assert len(store.rolling) == CONFIG.identity_max_signatures
    assert not any(np.array_equal(first_rolling, s) for s in store.rolling)


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


# -- permanent anchors ---------------------------------------------------


def test_first_good_views_become_anchors():
    ident = Identity()
    lettered(ident, 1, 0)
    store = ident._signatures["A"]
    assert len(store.anchors) == CONFIG.identity_anchor_signatures
    assert len(store.rolling) == 0


def test_a_slow_change_cannot_walk_the_store_away_from_the_anchors():
    ident = Identity()
    lettered(ident, 1, 0)
    for step in range(1, 21):
        ident.add_signature(1, blend(0, 1, step / 20), True, 10 + step)
    store = ident._signatures["A"]
    assert all(np.dot(s, vec(0)) >= CONFIG.identity_match_at for s in store.rolling)
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    ident.add_signature(2, vec(0), True, 9100)
    assert ident.letter_for(2) == "A"


def test_matching_a_return_also_uses_rolling_signatures():
    ident = Identity()
    lettered(ident, 1, 0)
    ident.add_signature(1, blend(0, 1, 0.6), True, 10)  # still like the anchors, stored rolling
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    ident.add_signature(2, vec(1), True, 9100)  # like the rolling view only
    assert ident.letter_for(2) == "A"


# -- a letter is decided by a frontal view -------------------------------


def test_return_turned_then_frontal_keeps_the_letter():
    ident = Identity()
    lettered(ident, 1, 0)
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    ident.add_signature(2, blend(0, 5, 0.9), False, 9100)  # too turned to match
    assert ident.letter_for(2) is None
    assert ident.needs_signature(2, 9101)
    seen(ident, {2}, 9400)
    ident.add_signature(2, vec(0), True, 9400)
    assert ident.letter_for(2) == "A"
    assert ident.people_count() == 1


def test_a_side_view_can_match_a_known_letter():
    ident = Identity()
    lettered(ident, 1, 0)
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    ident.add_signature(2, near(0, 5), False, 9100)
    assert ident.letter_for(2) == "A"


def test_stranger_seen_frontal_gets_a_new_letter_at_once():
    ident = Identity()
    lettered(ident, 1, 0)
    seen(ident, {1, 2}, 100)
    ident.add_signature(2, vec(1), True, 100)
    assert ident.letter_for(2) == "B"


def test_person_turned_for_the_whole_wait_gets_a_new_letter():
    ident = Identity()
    lettered(ident, 1, 0)
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    for t in range(9100, 9100 + CONFIG.identity_pending_ms, 100):
        seen(ident, {2}, t)
        ident.add_signature(2, vec(3), False, t)
        assert ident.letter_for(2) is None
    seen(ident, {2}, 9100 + CONFIG.identity_pending_ms)
    assert ident.letter_for(2) == "B"


def test_letter_from_the_wait_gives_way_when_the_face_turns_out_to_be_known():
    """That letter has never seen a frontal view, so nothing says it is anyone."""
    ident = Identity()
    lettered(ident, 1, 0)
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    seen(ident, {2}, 9100 + CONFIG.identity_pending_ms)
    assert ident.letter_for(2) == "B"
    ident.add_signature(2, vec(0), True, 11200)
    assert ident.letter_for(2) == "A"


# -- a track that stops looking like its letter --------------------------


def test_stranger_on_the_track_loses_the_letter_and_a_returns_as_a():
    ident = Identity()
    lettered(ident, 1, 0)
    ident.add_signature(1, vec(1), True, 1000)
    assert ident.letter_for(1) == "A"
    ident.add_signature(1, vec(1), True, 2000)
    assert ident.letter_for(1) == "B"
    ident.sync({1, 2}, {1, 2}, 2100)
    ident.add_signature(2, vec(0), True, 2100)
    assert ident.letter_for(2) == "A"
    assert ident.letter_for(1) == "B"


def test_one_bad_frontal_check_strips_nothing():
    ident = Identity()
    lettered(ident, 1, 0)
    ident.add_signature(1, vec(1), True, 1000)
    ident.add_signature(1, vec(0), True, 2000)
    ident.add_signature(1, vec(1), True, 3000)
    assert ident.letter_for(1) == "A"


def test_side_views_are_not_mismatch_checks():
    ident = Identity()
    lettered(ident, 1, 0)
    for t in (1000, 2000, 3000):
        ident.add_signature(1, vec(1), False, t)
    assert ident.letter_for(1) == "A"


def test_people_who_cross_get_their_own_letters_back():
    ident = Identity()
    ident.sync({1, 2}, {1, 2}, 0)
    for k in range(CONFIG.identity_anchor_signatures):
        ident.add_signature(1, vec(0), True, k)
        ident.add_signature(2, vec(1), True, k)
    assert (ident.letter_for(1), ident.letter_for(2)) == ("A", "B")
    for t in (1000, 2000):  # tracks swapped while crossing
        ident.add_signature(1, vec(1), True, t)
        ident.add_signature(2, vec(0), True, t + 1)
    assert (ident.letter_for(1), ident.letter_for(2)) == ("B", "A")
    assert ident.people_count() == 2


def test_real_person_returning_takes_the_letter_from_a_failing_track():
    ident = Identity()
    lettered(ident, 1, 0)
    ident.add_signature(1, vec(1), True, 1000)  # stranger on the track, one bad check
    ident.sync({1, 2}, {1, 2}, 1100)
    ident.add_signature(2, vec(0), True, 1100)
    assert ident.letter_for(2) == "A"
    assert ident.letter_for(1) is None


# -- checking is looser than storing -------------------------------------


def test_a_view_good_enough_to_check_is_not_stored():
    ident = Identity()
    lettered(ident, 1, 0)
    stored = len(ident._signatures["A"])
    decision = ident.add_signature(1, vec(0), False, 1000, checkable=True)
    assert decision.outcome == "kept"
    assert len(ident._signatures["A"]) == stored


def test_checkable_views_count_as_mismatch_checks():
    ident = Identity()
    lettered(ident, 1, 0)
    assert ident.add_signature(1, vec(1), False, 1000, checkable=True).outcome == "mismatch"
    second = ident.add_signature(1, vec(1), False, 2000, checkable=True)
    assert second.outcome == "stripped, waiting"
    assert ident.letter_for(1) is None
    assert set(ident._signatures) == {"A"}  # the loose view stored nothing, anywhere


def test_a_checkable_view_that_passes_clears_a_failed_check():
    ident = Identity()
    lettered(ident, 1, 0)
    ident.add_signature(1, vec(1), True, 1000)
    ident.add_signature(1, vec(0), False, 2000, checkable=True)
    ident.add_signature(1, vec(1), True, 3000)
    assert ident.letter_for(1) == "A"


def test_letter_from_the_wait_gives_way_to_a_checkable_view_without_storing_it():
    ident = Identity()
    lettered(ident, 1, 0)
    stored = len(ident._signatures["A"])
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    seen(ident, {2}, 9100 + CONFIG.identity_pending_ms)
    assert ident.letter_for(2) == "B"
    decision = ident.add_signature(2, vec(0), False, 11200, checkable=True)
    assert decision.outcome == "matched"
    assert ident.letter_for(2) == "A"
    assert len(ident._signatures["A"]) == stored


def test_a_view_too_poor_to_check_cannot_make_a_timeout_letter_give_way():
    ident = Identity()
    lettered(ident, 1, 0)
    seen(ident, set(), 9000, active=set())
    seen(ident, {2}, 9100)
    seen(ident, {2}, 9100 + CONFIG.identity_pending_ms)
    ident.add_signature(2, vec(0), False, 11200)
    assert ident.letter_for(2) == "B"


def test_a_side_view_cannot_take_a_letter_from_a_failing_track():
    ident = Identity()
    lettered(ident, 1, 0)
    ident.add_signature(1, vec(1), True, 1000)  # one bad check on track 1
    ident.sync({1, 2}, {1, 2}, 1100)
    decision = ident.add_signature(2, vec(0), False, 1100)
    assert decision.outcome == "waiting"
    assert ident.letter_for(1) == "A"
    assert ident.letter_for(2) is None


# -- decisions, for the debug log ----------------------------------------


def test_decision_reports_scores_from_before_the_view_was_learned():
    ident = Identity()
    seen(ident, {1}, 0)
    first = ident.add_signature(1, vec(0), True, 0)
    assert first.outcome == "new letter"
    assert first.letter == "A"
    assert first.best_letter is None
    assert first.best_score is None
    assert first.anchor_score is None


def test_decision_names_each_outcome():
    ident = Identity()
    lettered(ident, 1, 0)
    assert ident.add_signature(1, vec(0), True, 10).outcome == "kept"
    assert ident.add_signature(1, vec(1), True, 20).outcome == "mismatch"
    stripped = ident.add_signature(1, vec(1), True, 30)
    assert stripped.outcome == "stripped, new letter"
    assert stripped.letter == "B"
    seen(ident, {1, 2}, 40)
    waiting = ident.add_signature(2, vec(5), False, 40)
    assert waiting.outcome == "waiting"
    assert waiting.letter is None
    assert ident.add_signature(2, vec(0), True, 50).outcome == "matched"
