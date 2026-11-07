from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

import app.pipeline as pipeline_module
from app.core.config import CONFIG
from app.core.types import Box, FarDetectFailed, RawFace
from app.pipeline import Pipeline
from app.session_files import SessionFiles

START = datetime(2026, 11, 6, 16, 30, 0)
STEP_MS = 100
SIGNATURE_SIZE = 40
JPEG = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))[1].tobytes()


def at(ms):
    return START + timedelta(milliseconds=ms)


def face(marker=1, x=0.1, closed=False, yaw=0.0, pitch=0.0, blink=None, signature=None):
    blink = (0.95 if closed else 0.05) if blink is None else blink
    scores = {"eyeBlinkLeft": blink, "eyeBlinkRight": blink, "marker": marker}
    found = RawFace(Box(x, 0.1, 0.2, 0.2), scores, pitch, yaw, [(0.0, 0.0)] * 5)
    found.signature = signature
    return found


def between(first, second, turn):
    """A signature part way from one marker's to another's: 0 is first, 1 is second."""
    vector = np.zeros(SIGNATURE_SIZE)
    vector[first] = np.cos(turn * np.pi / 2)
    vector[second] = np.sin(turn * np.pi / 2)
    return vector


class FakeFinder:
    def __init__(self):
        self.faces = []
        self.error = None
        self.calls = 0

    def find(self, frame):
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.faces)


class FakeEmotion:
    def __init__(self):
        self.result = {"Neutral": 1.0}
        self.error = None
        self.failing_x = None
        self.calls = []

    def scores(self, frame_bgr, box):
        self.calls.append(box)
        if self.error or box.x == self.failing_x:
            raise self.error or ValueError("empty crop")
        return self.result


class FakeSigner:
    def __init__(self):
        self.failing = set()
        self.calls = []

    def signature(self, frame, f):
        marker = f.scores["marker"]
        self.calls.append(marker)
        if marker in self.failing:
            raise ValueError("no signature")
        if f.signature is not None:
            return f.signature
        vector = np.zeros(SIGNATURE_SIZE)
        vector[marker] = 1.0
        return vector


class Rig:
    def __init__(self, tmp_path, debug=False):
        self.root = tmp_path
        self.finder = FakeFinder()
        self.emotion = FakeEmotion()
        self.signer = FakeSigner()
        self.files = SessionFiles(tmp_path, START)
        self.pipe = Pipeline(self.finder, self.emotion, self.signer, self.files, debug=debug)
        self.t = 0
        self.result = None

    def frame(self, delay=1, jpeg=JPEG):
        self.result = self.pipe.process(jpeg, self.t, at(self.t), delay)
        self.t += STEP_MS
        return self.result

    def run(self, ms, *faces, delay=1):
        self.finder.faces = list(faces)
        for _ in range(ms // STEP_MS):
            self.frame(delay)
        return self.result

    def go_red(self, marker=1):
        self.run(2500, face(marker))
        self.run(2500, face(marker, closed=True))
        assert self.result.alarm

    def first_red_frame(self, marker=1):
        self.run(2500, face(marker))
        self.finder.faces = [face(marker, closed=True)]
        while not self.frame().alarm:
            pass

    def use_up_the_letters(self):
        for marker in range(1, 27):
            self.run(100, face(marker))
            self.run(CONFIG.letter_release_ms + 200)

    def log_text(self):
        return self.files.log_path.read_text(encoding="utf-8")

    def snaps(self):
        folder = self.root / "snaps"
        return sorted(folder.glob("*.jpg")) if folder.is_dir() else []


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def block_folder(path):
    path.write_text("a file where the folder should be")


# -- people coming and going ---------------------------------------------


def test_person_leaves_and_returns_keeps_letter(rig):
    rig.run(500, face(1))
    assert rig.result.people[0]["letter"] == "A"
    rig.run(10000)
    assert rig.result.people == []
    rig.run(200, face(1))
    assert [p["letter"] for p in rig.result.people] == ["A"]


def test_leave_and_return_keeps_baseline(rig):
    rig.run(500, face(1))
    baseline = rig.pipe._people["A"].baseline
    rig.run(10000)
    rig.run(200, face(1))
    assert rig.pipe._people["A"].baseline is baseline


def test_new_track_as_the_old_one_vanishes_takes_the_letter(rig):
    """Needs sync before add_signature, so the old track is already out of view."""
    rig.run(300, face(1, x=0.05))
    rig.run(100, face(1, x=0.7))
    assert [p["letter"] for p in rig.result.people] == ["A"]
    assert rig.result.people[0]["visible"] is True


def test_side_view_does_not_store_a_signature(rig):
    rig.run(100, face(1, yaw=CONFIG.frontal_yaw_deg + 1))
    assert rig.pipe._identity._signatures == {}
    rig.run(STEP_MS, face(1, yaw=-CONFIG.frontal_yaw_deg))
    assert len(rig.pipe._identity._signatures["A"]) == 1


def doze(rig, marker=1, asleep_marker=9):
    """10s upright, then a gradual droop, each view a little less like the last person."""
    rig.run(10000, face(marker))
    steps = 15
    for step in range(1, steps + 1):
        turned = between(marker, asleep_marker, step / steps)
        rig.run(500, face(marker, pitch=2.5 * step, blink=0.3, signature=turned))
    rig.run(7000, face(marker, pitch=2.5 * steps, closed=True, signature=vector_of(asleep_marker)))


def vector_of(marker):
    return between(marker, marker + 1, 0)


def test_dozing_does_not_cost_a_person_their_letter(rig):
    doze(rig)
    rig.run(5000, face(1))
    assert [p["letter"] for p in rig.result.people] == ["A"]
    rig.run(10000)
    rig.run(1000, face(1))
    assert [p["letter"] for p in rig.result.people] == ["A"]


def test_head_down_and_closed_eye_views_are_not_learned(rig):
    rig.run(5000, face(1))
    store = rig.pipe._identity._signatures["A"]
    learned = len(store)
    head_down = face(1, pitch=CONFIG.frontal_pitch_delta_deg + 1)
    rig.run(3000, head_down)
    assert len(store) == learned
    rig.run(3000, face(1, blink=0.9))
    assert len(store) == learned


def test_first_anchors_need_open_eyes(rig):
    rig.run(1500, face(1, closed=True))
    assert "A" not in rig.pipe._identity._signatures
    rig.run(1500, face(1))
    assert len(rig.pipe._identity._signatures["A"].anchors) >= 1


def test_head_down_view_of_someone_dozing_does_not_strip_their_letter(rig):
    rig.run(5000, face(1))
    rig.run(5000, face(1, pitch=30, signature=vector_of(9)))
    assert [p["letter"] for p in rig.result.people] == ["A"]


def test_return_turned_then_frontal_keeps_the_letter(rig):
    rig.run(3000, face(1))
    rig.run(10000)
    turned = between(1, 9, 0.9)
    rig.run(300, face(1, yaw=CONFIG.frontal_yaw_deg + 25, signature=turned))
    assert rig.result.lines == ["Identifying..."]
    rig.run(500, face(1))
    assert [p["letter"] for p in rig.result.people] == ["A"]
    assert "Person B" not in rig.log_text()


def test_stranger_in_the_seat_gets_their_own_letter_and_a_comes_back(rig):
    rig.run(5000, face(1))
    rig.run(1500)
    rig.run(3000, face(2))
    assert [p["letter"] for p in rig.result.people] == ["B"]
    assert "| Person A | left" in rig.log_text()
    rig.run(500, face(2), face(1, x=0.7))
    assert [p["letter"] for p in rig.result.people] == ["A", "B"]
    assert rig.result.people[0]["box"]["x"] == 0.7


def test_two_waiting_faces_both_get_their_signature_turn(rig):
    turned = CONFIG.frontal_yaw_deg + 1
    rig.run(300, face(1, x=0.1, yaw=turned), face(2, x=0.6, yaw=turned))
    assert set(rig.signer.calls) == {1, 2}


def test_people_past_z_show_as_unknown_and_keep_separate_state(rig):
    rig.use_up_the_letters()
    rig.run(300, face(27, x=0.1), face(28, x=0.6))
    assert [p["letter"] for p in rig.result.people] == ["?", "?"]
    assert rig.result.lines == ["Person ? looks neutral 100%"] * 2
    assert rig.log_text().count("| Person ? | joined") == 2
    assert len([key for key in rig.pipe._people if key.startswith("?")]) == 2


def test_red_person_past_z_snaps_as_unknown(rig):
    rig.use_up_the_letters()
    rig.go_red(marker=27)
    assert rig.snaps()[0].name.endswith("_unknown_sleeping.jpg")
    assert "| Person ? | sleeping | " in rig.log_text()


def test_people_are_listed_in_letter_order_with_unlettered_last(rig):
    rig.signer.failing = {3}
    rig.run(300, face(2, x=0.6), face(1, x=0.1))
    rig.run(100, face(3, x=0.35), face(2, x=0.6), face(1, x=0.1))
    assert [p["letter"] for p in rig.result.people] == ["A", "B", None]
    assert rig.result.count_text == "3 faces found"
    assert rig.result.lines[-1] == "Identifying..."


def test_person_dict_shape_for_a_normal_person(rig):
    rig.run(300, face(1))
    assert rig.result.people == [
        {
            "letter": "A",
            "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            "label": "neutral",
            "confidence": 100,
            "red": False,
            "reason": None,
            "visible": True,
        }
    ]
    assert rig.result.count_text == "1 face found"


def test_unknown_person_is_not_classified_until_lettered(rig):
    rig.signer.failing = {1}
    rig.finder.faces = [face(1, closed=True)]
    rig.frame()
    assert rig.result.lines == ["Identifying..."]
    assert rig.result.people[0]["letter"] is None
    assert rig.result.people[0]["red"] is False
    assert rig.result.people[0]["reason"] is None
    assert rig.result.alarm is False
    assert rig.pipe._people == {}


# -- a letter that stuck -------------------------------------------------


def test_letter_given_on_a_head_down_return_gives_way_when_they_look_up(rig):
    """The new letter's baseline froze head-down, so upright looks wrong to the strict gate."""
    rig.run(3000, face(1))
    rig.run(10000)
    rig.run(4000, face(1, pitch=25, signature=between(1, 9, 0.9)))
    assert [p["letter"] for p in rig.result.people] == ["B"]
    rig.run(3000, face(1))
    assert [p["letter"] for p in rig.result.people] == ["A"]


def test_letter_given_on_a_turned_head_down_return_gives_way_when_they_look_up(rig):
    rig.run(3000, face(1))
    rig.run(10000)
    away = CONFIG.frontal_yaw_deg + 25
    rig.run(4000, face(1, pitch=25, yaw=away, signature=between(1, 9, 0.9)))
    assert [p["letter"] for p in rig.result.people] == ["B"]
    rig.run(2000, face(1))
    assert [p["letter"] for p in rig.result.people] == ["A"]


def test_stranger_holding_a_steady_tilt_is_checked_and_loses_the_letter(rig):
    rig.run(5000, face(1))
    rig.run(3000, face(2, pitch=CONFIG.frontal_pitch_delta_deg + 8))
    assert [p["letter"] for p in rig.result.people] == ["B"]


def test_a_tilted_view_is_checked_but_never_stored(rig):
    rig.run(5000, face(1))
    store = rig.pipe._identity._signatures["A"]
    learned = len(store)
    rig.run(3000, face(1, pitch=CONFIG.frontal_pitch_delta_deg + 8))
    assert len(store) == learned
    assert [p["letter"] for p in rig.result.people] == ["A"]


def test_eyes_closed_doze_inside_the_check_pitch_keeps_the_letter(rig):
    rig.run(5000, face(1))
    rig.run(5000, face(1, pitch=20, closed=True, signature=vector_of(9)))
    assert [p["letter"] for p in rig.result.people] == ["A"]


def test_a_nod_with_heavy_eyes_is_not_a_check_so_the_sleep_rule_keeps_its_person(rig):
    """Real numbers from the owner: a nod reads pitch delta 22, closure 0.13."""
    rig.run(5000, face(1))
    rig.run(5000, face(1, pitch=22, blink=0.18, signature=vector_of(9)))
    assert [p["letter"] for p in rig.result.people] == ["A"]
    assert rig.result.alarm is True


# -- far faces -----------------------------------------------------------


FAR_LINE = "Person A is too far to read"


def far(marker=1, x=0.1, signature=None):
    """Real far faces have empty scores. The marker is only for FakeSigner."""
    found = RawFace(Box(x, 0.1, 0.05, 0.05), {"marker": marker}, 0, 0, [(0.0, 0.0)] * 5, far=True)
    found.signature = signature
    return found


def confirm_far(rig, *faces):
    rig.run(CONFIG.far_confirm_frames * STEP_MS, *faces)


def lettered_far(rig, marker=1):
    rig.run(CONFIG.far_confirm_frames * STEP_MS + CONFIG.identity_pending_ms, far(marker))


def test_far_face_shows_nothing_until_it_has_lasted(rig):
    rig.finder.faces = [far()]
    for _ in range(CONFIG.far_confirm_frames - 1):
        result = rig.frame()
        assert result.people == []
        assert result.lines == []
        assert result.count_text == "No faces found"
    assert "Person" not in rig.log_text()
    assert rig.signer.calls == []
    result = rig.frame()
    assert [p["letter"] for p in result.people] == [None]
    assert result.count_text == "1 face found"


def test_flickering_false_face_never_shows_and_costs_no_letter(rig):
    for _ in range(5):
        rig.run((CONFIG.far_confirm_frames - 1) * STEP_MS, far())
        assert rig.result.people == []
        rig.run(STEP_MS)
    assert rig.result.count_text == "No faces found"
    assert rig.pipe._identity.people_count() == 0
    assert rig.signer.calls == []
    assert "Person" not in rig.log_text()


def test_a_false_face_does_not_hide_or_add_to_the_near_people(rig):
    rig.run(500, face(1))
    rig.run((CONFIG.far_confirm_frames - 1) * STEP_MS, face(1), far(2, x=0.8))
    assert rig.result.lines == ["Person A looks neutral 100%"]
    assert rig.result.count_text == "1 face found"


def test_far_person_shows_a_box_and_letter_with_a_line_and_no_percent(rig):
    lettered_far(rig)
    assert rig.result.lines == [FAR_LINE]
    assert rig.result.people == [
        {
            "letter": "A",
            "box": {"x": 0.1, "y": 0.1, "width": 0.05, "height": 0.05},
            "label": "far",
            "confidence": 0,
            "red": False,
            "reason": None,
            "visible": True,
        }
    ]
    assert rig.result.alarm is False


def test_far_person_who_is_seen_on_and_off_stays_shown(rig):
    lettered_far(rig)
    rig.run(STEP_MS)
    assert rig.result.people[0]["visible"] is False
    rig.run(STEP_MS, far())
    assert rig.result.people[0]["visible"] is True


def test_far_person_never_gets_an_emotion_check(rig):
    lettered_far(rig)
    rig.run(1000, far())
    assert rig.emotion.calls == []


def test_far_person_takes_no_snap_or_alarm_and_stays_out_of_the_debug_numbers(tmp_path):
    rig = Rig(tmp_path, debug=True)
    lettered_far(rig)
    rig.run(3000, far())
    assert rig.result.alarm is False
    assert rig.snaps() == []
    assert rig.result.debug == []
    assert not [s for s in rig.pipe.debug_samples() if "signals" in s]


def test_going_far_and_back_keeps_the_letter_and_logs_it(rig):
    rig.run(3000, face(1))
    rig.run(1000, far())
    assert rig.result.lines == [FAR_LINE]
    assert "| Person A | is too far to read\n" in rig.log_text()
    rig.run(1000, face(1))
    assert rig.result.lines == ["Person A looks neutral 100%"]
    assert rig.pipe._identity.people_count() == 1
    assert "| Person A | left" not in rig.log_text()


def test_far_face_never_feeds_the_baseline_or_classifier(rig):
    rig.run(500, face(1))
    state = rig.pipe._people["A"]
    samples = len(state.baseline._closures)
    updated_ms = state.classifier._state.last_update_ms
    rig.run(1000, far())
    assert rig.result.lines == [FAR_LINE]
    assert len(state.baseline._closures) == samples
    assert state.baseline.normal() is None
    assert state.classifier._state.last_update_ms == updated_ms


def test_time_far_away_is_not_closed_eye_time(rig):
    """Eyes closed 1.5s, far 1.8s, back with eyes open."""
    rig.run(4000, face(1))
    rig.run(1500, face(1, closed=True), delay=3)
    rig.run(1800, far(), delay=3)
    assert rig.result.lines == [FAR_LINE]
    rig.finder.faces = [face(1)]
    for _ in range(15):
        assert not rig.frame(delay=3).alarm
        assert rig.result.people[0]["label"] != "sleeping"
    assert "fell asleep" not in rig.log_text()


def test_time_far_away_does_not_add_to_a_closure_that_carries_on(rig):
    """Eyes closed 1.5s, far 1.8s, then closed 1s more: only 2.5s were seen, under the delay."""
    rig.run(4000, face(1))
    rig.run(1500, face(1, closed=True), delay=3)
    rig.run(1800, far(), delay=3)
    rig.finder.faces = [face(1, closed=True)]
    for _ in range(10):
        assert not rig.frame(delay=3).alarm


def test_alarming_person_going_far_stops_the_alarm(rig):
    """Only after letter_release_ms of far views: a sleeper's tipped head looks the same at first."""
    rig.go_red()
    rig.run(CONFIG.letter_release_ms - 3 * STEP_MS, far())
    assert rig.result.alarm is True
    rig.run(5 * STEP_MS, far())
    assert rig.result.alarm is False
    assert rig.result.people[0]["red"] is False
    assert rig.log_text().count("| System | alarm stopped") == 1
    assert len(rig.snaps()) == 1


def test_sleeper_mediapipe_keeps_losing_never_reads_far(rig):
    """Moving head: MediaPipe catches 1 frame in 8, YuNet sees the rest."""
    rig.go_red()
    for _ in range(40):
        rig.finder.faces = [face(1, closed=True)]
        assert rig.frame().alarm is True
        rig.finder.faces = [far()]
        for _ in range(7):
            assert rig.frame().alarm is True
    text = rig.log_text()
    assert text.count("alarm started") == 1
    assert "alarm stopped" not in text
    assert "too far" not in text
    assert len(rig.snaps()) == 1


def test_a_gap_in_the_far_views_restarts_a_red_persons_wait(rig):
    rig.go_red()
    rig.run(CONFIG.letter_release_ms - 3 * STEP_MS, far())
    rig.run(STEP_MS)
    rig.run(CONFIG.letter_release_ms - 3 * STEP_MS, far())
    assert rig.result.alarm is True
    assert "too far" not in rig.log_text()


def test_a_person_who_is_not_red_still_reads_far_quickly(rig):
    rig.run(3000, face(1))
    rig.run(CONFIG.far_confirm_frames * STEP_MS, far())
    assert rig.result.lines == [FAR_LINE]


def test_red_just_short_of_the_snap_then_hidden_then_far_still_gets_its_proof(rig):
    rig.first_red_frame()
    rig.run(CONFIG.min_red_ms - STEP_MS, face(1, closed=True))
    assert rig.snaps() == []
    rig.run(CONFIG.letter_release_ms + 3 * STEP_MS, far())
    assert rig.result.lines == [FAR_LINE]
    assert len(rig.snaps()) == 1
    assert rig.pipe._red_proof == {}


def test_red_for_under_the_snap_limit_then_far_takes_no_snap(rig):
    rig.first_red_frame()
    rig.run(300, face(1, closed=True))
    rig.run(CONFIG.letter_release_ms + 3 * STEP_MS, far())
    assert rig.snaps() == []
    assert rig.pipe._red_proof == {}


def test_wall_picture_uncovered_as_a_leaves_cannot_keep_their_letter(rig):
    rig.run(3000, face(1, x=0.4))
    rig.run(1500)
    picture = far(7, x=0.45)
    rig.run(3000, picture)
    assert rig.result.lines == [FAR_LINE]  # the picture inherited the track, for now
    rig.run(500, face(1, x=0.1), picture)
    shown = {p["box"]["x"]: p["letter"] for p in rig.result.people}
    assert shown[0.1] == "A"
    assert shown[0.45] is None
    rig.run(CONFIG.identity_pending_ms + 500, face(1, x=0.1), picture)
    assert rig.result.lines == ["Person A looks neutral 100%", "Person B is too far to read"]
    assert list(rig.pipe._identity._signatures) == ["A"]


def test_far_view_strips_nothing_and_stores_nothing(rig):
    rig.run(5000, face(1))
    store = rig.pipe._identity._signatures["A"]
    learned = len(store)
    rig.run(4000, far(2))
    assert rig.result.lines == [FAR_LINE]
    assert len(store) == learned


def test_far_view_cannot_create_a_letter_early_or_store_one(rig):
    confirm_far(rig, far())
    rig.run(CONFIG.identity_pending_ms - 2 * STEP_MS, far())
    assert rig.result.lines == ["Identifying..."]
    rig.run(3 * STEP_MS, far())
    assert rig.result.lines == [FAR_LINE]
    assert rig.pipe._identity._signatures == {}


def test_far_view_can_match_a_known_letter(rig):
    rig.run(3000, face(1))
    rig.run(10000)
    confirm_far(rig, far())
    rig.run(2 * STEP_MS, far())
    assert rig.result.lines == [FAR_LINE]
    assert rig.pipe._identity.people_count() == 1


def test_letter_from_the_wait_gives_way_when_the_far_person_comes_close(rig):
    rig.run(3000, face(1))
    rig.run(10000)
    unreadable = between(1, 9, 0.9)
    rig.run(CONFIG.far_confirm_frames * STEP_MS, far(signature=unreadable))
    rig.run(CONFIG.identity_pending_ms, far(signature=unreadable))
    assert rig.result.lines == ["Person B is too far to read"]
    rig.run(1500, face(1))
    assert [p["letter"] for p in rig.result.people] == ["A"]


def test_near_person_lost_for_a_moment_is_hidden_not_called_far(rig):
    rig.go_red()
    rig.run((CONFIG.far_confirm_frames - 1) * STEP_MS, far())
    assert rig.result.alarm is True
    assert rig.result.people[0]["visible"] is False
    assert rig.result.people[0]["label"] == "sleeping"


def test_a_face_on_the_edge_of_near_and_far_never_reads_as_far(rig):
    rig.run(3000, face(1))
    for _ in range(20):
        rig.run(STEP_MS, far())
        rig.run(STEP_MS, face(1))
    assert "too far" not in rig.log_text()
    assert rig.result.lines == ["Person A looks neutral 100%"]


def test_far_faces_count_towards_max_faces(rig):
    near = [face(i, x=0.2 * i) for i in range(1, 4)]
    rig.run(1000 + CONFIG.identity_pending_ms, *near, far(4, x=0.9), far(5, x=0.05))
    assert rig.result.count_text == "4 faces found"
    assert rig.result.overflow is True


def test_far_detection_failure_warns_and_keeps_the_near_faces(rig):
    rig.finder.error = FarDetectFailed([face(1)])
    rig.run(300)
    assert rig.result.warning == pipeline_module.FAR_WARNING
    assert rig.result.lines == ["Person A looks neutral 100%"]
    assert rig.result.people[0]["visible"] is True


def test_one_frame_of_a_false_far_face_does_not_claim_a_crowd(rig):
    near = [face(i, x=0.2 * i) for i in range(1, 5)]
    rig.run(500, *near)
    rig.run(STEP_MS, *near, far(5, x=0.05))
    assert rig.result.overflow is False


# -- red marks, alarm and snaps ------------------------------------------


def test_closed_eyes_goes_red_and_sounds(rig):
    rig.go_red()
    person = rig.result.people[0]
    assert rig.result.alarm is True
    assert person["red"] is True
    assert person["label"] == "sleeping"
    assert person["reason"] == rig.result.lines[0]
    assert rig.result.lines[0].startswith("Person A is sleeping")


def test_red_event_writes_one_snap_and_links_it_in_the_log(rig):
    rig.go_red()
    rig.run(2000, face(1, closed=True))
    snaps = rig.snaps()
    assert len(snaps) == 1
    assert snaps[0].name.endswith("_A_sleeping.jpg")
    line = next(s for s in rig.log_text().splitlines() if f"snaps/{snaps[0].name}" in s)
    assert " | Person A | sleeping | " in line
    assert line in rig.result.log


def test_snap_is_rendered_from_the_decoded_frame(rig, monkeypatch):
    seen = []
    real = pipeline_module.render_snap

    def spy(frame, people):
        seen.append((frame.shape, [p.letter for p in people]))
        return real(frame, people)

    monkeypatch.setattr(pipeline_module, "render_snap", spy)
    rig.go_red()
    assert seen == [((64, 64, 3), ["A"])]


def test_alarm_stops_after_red_person_leaves(rig):
    rig.go_red()
    rig.run(1000)
    assert rig.result.alarm is True
    person = rig.result.people[0]
    assert person["visible"] is False
    assert person["label"] == "sleeping"
    rig.run(CONFIG.letter_release_ms)
    assert rig.result.alarm is False
    assert rig.result.people == []
    assert "| Person A | left" in rig.log_text()


def test_red_person_returning_within_grace_takes_no_second_snap(rig):
    rig.go_red()
    assert len(rig.snaps()) == 1
    rig.run(1000)
    rig.run(1000, face(1, closed=True))
    assert rig.result.alarm is True
    assert len(rig.snaps()) == 1


def test_red_person_returning_after_release_is_a_new_event(rig):
    rig.go_red()
    rig.run(CONFIG.letter_release_ms + 500)
    rig.run(3000, face(1, closed=True))
    assert rig.result.alarm is True
    assert len(rig.snaps()) == 2


def test_leaving_forgets_the_snap_even_if_the_reading_stays_red(rig):
    """Back after release but inside frame_gap_reset_ms, so the classifier is still red."""
    rig.go_red()
    rig.run(CONFIG.letter_release_ms + 200)
    assert rig.result.people == []
    rig.run(100, face(1, closed=True))
    assert rig.result.people[0]["red"] is True
    rig.run(CONFIG.min_red_ms + 100, face(1, closed=True))
    assert len(rig.snaps()) == 2


def test_hidden_red_person_takes_no_snap(rig):
    rig.first_red_frame()
    rig.run(300, face(1, closed=True))
    rig.run(CONFIG.letter_release_ms + 500)
    assert rig.result.people == []
    assert rig.snaps() == []


def test_seen_red_for_the_limit_then_lost_saves_the_last_visible_frame(rig, monkeypatch):
    seen = []
    real = pipeline_module.render_snap

    def spy(frame, people):
        seen.append((frame.shape, [(p.letter, p.visible) for p in people]))
        return real(frame, people)

    monkeypatch.setattr(pipeline_module, "render_snap", spy)
    small = cv2.imencode(".jpg", np.zeros((32, 32, 3), np.uint8))[1].tobytes()
    rig.first_red_frame()
    rig.run(CONFIG.min_red_ms - 2 * STEP_MS, face(1, closed=True))
    last_seen_ms = rig.t
    rig.frame(jpeg=small)
    rig.run(CONFIG.letter_release_ms - STEP_MS)
    assert rig.snaps() == []  # never while they are only in grace
    rig.run(3 * STEP_MS)
    assert rig.result.people == []
    assert seen == [((32, 32, 3), [("A", True)])]
    snaps = rig.snaps()
    assert len(snaps) == 1
    assert f"_{at(last_seen_ms):%H-%M-%S}_A_sleeping.jpg" in snaps[0].name
    assert f"snaps/{snaps[0].name}" in rig.log_text()


def test_lost_after_red_for_under_the_limit_takes_no_snap_but_keeps_the_log_line(rig):
    """The flicker rule: a red mark this short has no proof, and the README says so."""
    rig.first_red_frame()
    rig.run(300, face(1, closed=True))
    rig.run(CONFIG.letter_release_ms + 500)
    assert rig.snaps() == []
    assert "fell asleep" in rig.log_text()


def test_kept_frame_is_forgotten_when_the_person_stops_being_red(rig):
    rig.first_red_frame()
    rig.run(1000, face(1))
    rig.run(CONFIG.letter_release_ms + 500)
    assert rig.snaps() == []
    assert rig.pipe._red_proof == {}


def test_one_red_frame_then_finder_failure_takes_no_snap(rig):
    rig.first_red_frame()
    rig.finder.error = RuntimeError("detector broke")
    rig.run(CONFIG.letter_release_ms + 500)
    assert rig.result.alarm is False
    assert rig.snaps() == []


def test_returning_still_red_snaps_once_after_a_visible_second(rig):
    rig.first_red_frame()
    rig.run(300, face(1, closed=True))
    rig.run(1500)
    assert rig.result.people[0]["visible"] is False
    rig.run(500, face(1, closed=True))
    assert rig.snaps() == []
    rig.run(CONFIG.min_red_ms, face(1, closed=True))
    assert len(rig.snaps()) == 1
    rig.run(3000, face(1, closed=True))
    assert len(rig.snaps()) == 1


def test_time_with_no_face_is_not_closed_eye_time(rig):
    """Eyes closed 1.5s, face lost 1.8s, back with eyes open."""
    rig.run(4000, face(1))
    rig.run(1500, face(1, closed=True), delay=3)
    rig.run(1800, delay=3)
    rig.finder.faces = [face(1)]
    for _ in range(15):
        assert not rig.frame(delay=3).alarm
        assert rig.result.people[0]["label"] != "sleeping"
    assert "fell asleep" not in rig.log_text()
    assert rig.snaps() == []


def test_four_faces_rate_still_reaches_sleeping_and_the_alarm(rig):
    step_ms = 222  # about 4.5 frames a second
    rig.finder.faces = [face(1)]
    for _ in range(20):
        rig.pipe.process(JPEG, rig.t, at(rig.t), 3)
        rig.t += step_ms
    rig.finder.faces = [face(1, closed=True)]
    shut_ms = rig.t
    while not rig.pipe.process(JPEG, rig.t, at(rig.t), 3).alarm:
        rig.t += step_ms
        assert rig.t - shut_ms < 4000
    assert rig.t - shut_ms >= 3000


def test_alarm_start_and_stop_are_logged(rig):
    rig.go_red()
    rig.run(3000, face(1))
    text = rig.log_text()
    assert text.count("| System | alarm started") == 1
    assert text.count("| System | alarm stopped") == 1
    assert text.index("alarm started") < text.index("alarm stopped")


def test_unwritable_snap_folder_warns_and_keeps_detecting(rig):
    block_folder(rig.root / "snaps")
    rig.run(2500, face(1))
    rig.finder.faces = [face(1, closed=True)]
    warnings = [rig.frame().warning for _ in range(25)]
    assert sum(w is not None for w in warnings) == 1
    assert "| Person A | snap failed" in rig.log_text()
    assert rig.result.alarm is True
    assert rig.result.people[0]["label"] == "sleeping"


# -- bad frames ----------------------------------------------------------


def test_corrupt_frame_is_skipped_and_session_continues(rig):
    good = rig.run(300, face(1))
    calls = rig.finder.calls
    bad = rig.frame(jpeg=b"this is not a jpeg")
    assert bad.warning
    assert bad.lines == good.lines
    assert rig.finder.calls == calls
    assert rig.frame().warning is None


def test_empty_frame_is_skipped(rig):
    good = rig.run(300, face(1))
    calls = rig.finder.calls
    bad = rig.frame(jpeg=b"")
    assert bad.warning
    assert bad.people == good.people
    assert rig.finder.calls == calls


def test_oversized_frame_is_skipped(rig):
    bad = rig.frame(jpeg=JPEG + b"\0" * CONFIG.jpeg_max_bytes)
    assert bad.warning
    assert bad.count_text == "No faces found"
    assert rig.finder.calls == 0


# -- the heavy check -----------------------------------------------------


def test_one_heavy_check_per_frame(rig):
    rig.finder.faces = [face(i, x=0.2 * i) for i in range(1, 5)]
    for _ in range(40):
        before = len(rig.signer.calls) + len(rig.emotion.calls)
        rig.frame()
        assert len(rig.signer.calls) + len(rig.emotion.calls) - before <= 1


def test_signature_comes_before_emotion(rig):
    rig.run(100, face(1))
    assert rig.signer.calls == [1]
    assert rig.emotion.calls == []
    rig.run(100, face(1))
    assert len(rig.emotion.calls) == 1


def test_emotion_goes_to_the_most_overdue_person_at_the_interval(rig):
    rig.run(1200, face(1, x=0.1), face(2, x=0.6))
    xs = [box.x for box in rig.emotion.calls]
    assert xs[:4] == [0.1, 0.6, 0.1, 0.6]
    assert len(xs) <= 2 * (1200 // CONFIG.emotion_interval_ms + 1)


def test_more_than_four_faces_sets_overflow(rig):
    rig.finder.faces = [face(i, x=0.15 * i) for i in range(1, 6)]
    assert rig.frame().overflow is True
    rig.finder.faces = rig.finder.faces[:4]
    assert rig.frame().overflow is False


# -- model failures ------------------------------------------------------


def test_failing_signature_lets_the_person_wait_then_letters_them(rig):
    rig.signer.failing = {1}
    rig.run(CONFIG.identity_pending_ms - STEP_MS, face(1))
    assert rig.result.lines == ["Identifying..."]
    rig.run(2 * STEP_MS, face(1))
    assert rig.result.people[0]["letter"] == "A"


def test_signature_failure_warns_and_the_frame_continues(rig):
    rig.signer.failing = {1}
    rig.finder.faces = [face(1)]
    assert rig.frame().warning


def test_failing_signature_does_not_starve_other_people(rig):
    rig.signer.failing = {1}
    rig.run(300, face(1, x=0.1), face(2, x=0.6))
    assert [p["letter"] for p in rig.result.people] == ["A", None]
    assert rig.result.people[0]["box"]["x"] == 0.6
    assert rig.emotion.calls


def test_emotion_failure_keeps_last_emotion(rig):
    rig.emotion.result = {"Anger": 0.9}
    rig.run(1000, face(1))
    assert rig.result.people[0]["label"] == "angry"
    rig.emotion.error = ValueError("empty crop")
    rig.run(1000, face(1))
    assert rig.result.people[0]["label"] == "angry"
    assert rig.pipe._people["A"].emotion == {"Anger": 0.9}


def test_stale_emotion_is_cleared_when_a_person_returns_after_release(rig):
    rig.emotion.result = {"Anger": 0.9}
    rig.run(1000, face(1))
    assert rig.result.people[0]["label"] == "angry"
    rig.run(CONFIG.letter_release_ms + 200)
    rig.emotion.error = ValueError("empty crop")
    rig.run(1000, face(1))
    assert rig.pipe._people["A"].emotion is None
    assert rig.result.people[0]["label"] == "neutral"


def test_emotion_is_kept_through_the_grace(rig):
    rig.emotion.result = {"Anger": 0.9}
    rig.run(1000, face(1))
    rig.run(1000)
    rig.emotion.error = ValueError("empty crop")
    rig.run(300, face(1))
    assert rig.result.people[0]["label"] == "angry"


def test_failing_emotion_does_not_starve_other_people(rig):
    rig.emotion.failing_x = 0.1
    rig.run(500, face(1, x=0.1), face(2, x=0.6))
    assert 0.6 in [box.x for box in rig.emotion.calls]


def test_finder_failure_gives_no_faces_and_a_warning(rig):
    rig.run(300, face(1))
    rig.finder.error = RuntimeError("detector broke")
    result = rig.frame()
    assert result.warning
    assert result.overflow is False
    assert result.people[0]["visible"] is False
    rig.run(CONFIG.letter_release_ms + 200)
    assert rig.result.people == []
    assert rig.result.count_text == "No faces found"


# -- the log -------------------------------------------------------------


def test_session_started_is_written_and_shown(rig):
    assert "| System | session started" in rig.log_text()
    rig.run(100, face(1))
    assert rig.result.log[-1].endswith("| System | session started")


def test_panel_log_is_the_file_lines_newest_first(rig):
    rig.go_red()
    rig.run(3000, face(1))
    file_lines = rig.log_text().splitlines()
    assert sorted(rig.result.log) == sorted(file_lines)
    assert rig.result.log[-1] == file_lines[0]
    assert rig.result.log[0] == file_lines[-1]


def test_panel_log_is_capped(rig):
    rig.pipe._log = ["old line"] * CONFIG.log_cap
    rig.run(100, face(1))
    assert len(rig.result.log) == CONFIG.log_cap
    assert rig.result.log[0].endswith("| Person A | joined")


def test_session_log_write_failure_warns(tmp_path):
    block_folder(tmp_path / "logs")
    rig = Rig(tmp_path)
    rig.finder.faces = [face(1)]
    assert rig.frame().warning


# -- manual snapshot, debug and close ------------------------------------


def test_manual_snapshot_saves_file(rig):
    rig.run(300, face(1))
    assert rig.pipe.snapshot(at(rig.t)) is None
    snaps = rig.snaps()
    assert len(snaps) == 1
    assert snaps[0].name.endswith("_all_manual.jpg")
    assert f"snaps/{snaps[0].name}" in rig.log_text()


def test_manual_snapshot_before_any_frame_warns(rig):
    assert rig.pipe.snapshot(at(0))
    assert rig.snaps() == []


def test_failed_manual_snapshot_warns_and_logs(rig):
    block_folder(rig.root / "snaps")
    rig.run(300, face(1))
    assert rig.pipe.snapshot(at(rig.t))
    assert "| System | snap failed" in rig.log_text()


def test_debug_is_off_by_default(rig):
    rig.run(300, face(1))
    assert rig.result.debug is None
    assert rig.pipe.debug_samples() == []


def test_debug_lines_and_samples(tmp_path):
    rig = Rig(tmp_path, debug=True)
    rig.run(300, face(1))
    assert rig.result.debug[0].startswith("A: eye_closure 0.05  look_down 0.00")
    assert rig.result.debug[1].startswith("A last 10s: eye_closure 0.05 to 0.05  look_down")
    assert rig.result.debug[2] == "A drowsy: no sign"
    sample = rig.pipe.debug_samples()[-1]
    assert sample["letter"] == "A"
    assert sample["t_ms"] == 200
    assert sample["signals"]["eye_closure"] == pytest.approx(0.05)
    assert sample["label"] == "neutral"
    assert sample["confidence"] == 100
    assert sample["drowsy_sign"] is None
    assert sample["strong_held"] is False
    assert sample["strong_sustained"] is False


def test_debug_samples_carry_the_numbers_for_tuning(tmp_path):
    rig = Rig(tmp_path, debug=True)
    rig.emotion.result = {"Neutral": 0.7, "Anger": 0.3}
    rig.run(3000, face(1, yaw=4.0, pitch=2.0))
    rig.run(500, face(1, closed=True, yaw=4.0, pitch=2.0), delay=3)
    sample = rig.pipe.debug_samples()[-1]
    assert sample["raw_eye_blink"] == pytest.approx(0.95)
    assert sample["baseline_closure"] == pytest.approx(0.05)
    assert sample["baseline_pitch"] == pytest.approx(2.0)
    assert sample["closed_ms"] == 400
    assert sample["closed_threshold"] == CONFIG.eyes_closed_at
    assert sample["yaw_deg"] == 4.0
    assert sample["pitch_deg"] == 2.0
    assert sample["emotion"] == {"Neutral": 0.7, "Anger": 0.3}


def test_debug_sample_before_a_baseline_says_so(tmp_path):
    rig = Rig(tmp_path, debug=True)
    rig.run(200, face(1))
    sample = rig.pipe.debug_samples()[-1]
    assert sample["baseline_closure"] is None
    assert sample["baseline_pitch"] is None


def identity_samples(rig):
    return [s for s in rig.pipe.debug_samples() if s.get("kind") == "identity"]


def test_identity_decisions_are_in_the_debug_log_as_numbers_only(tmp_path):
    rig = Rig(tmp_path, debug=True)
    rig.run(100, face(1))
    rig.run(1100, face(1), face(2, x=0.6))
    first = identity_samples(rig)[0]
    assert first["track"] == 1
    assert first["outcome"] == "new letter"
    assert first["letter"] == "A"
    assert first["best_letter"] is None
    later = [s for s in identity_samples(rig) if s["track"] == 1][-1]
    assert later["outcome"] == "kept"
    assert later["best_letter"] == "A"
    assert later["best_score"] == pytest.approx(1.0)
    assert later["runner_up_letter"] == "B"
    assert later["runner_up_score"] == pytest.approx(0.0)
    assert later["anchor_score"] == pytest.approx(1.0)
    assert later["frontal"] is True
    assert not any(
        isinstance(value, (np.ndarray, list)) for s in identity_samples(rig) for value in s.values()
    )


def test_debug_lines_show_the_sleep_count_and_identity_score(tmp_path):
    rig = Rig(tmp_path, debug=True)
    rig.run(1200, face(1))
    rig.run(500, face(1, closed=True))
    lines = rig.result.debug
    assert "A sleep: closed 400 ms, closed at 0.33" in lines
    assert any(line.startswith("A identity: best A 1.00, ") for line in lines)
    assert any(line.startswith("A raw: eyeBlink 0.95") for line in lines)


def test_no_identity_decisions_are_kept_with_debug_off(rig):
    rig.run(300, face(1))
    assert rig.pipe.debug_samples() == []


def test_debug_wording_for_strong_expressions():
    text = pipeline_module._drowsy_text
    assert text({"strong_sustained": True, "strong_held": True, "drowsy_sign": None}) == (
        "strong expression blocking"
    )
    assert text({"strong_sustained": False, "strong_held": True, "drowsy_sign": "droopy"}) == (
        "sign droopy, strong expression delays sleeping"
    )


def test_close_writes_session_ended(rig):
    rig.pipe.close(at(1000))
    assert rig.log_text().splitlines()[-1].endswith("| System | session ended")


def test_close_during_alarm_logs_alarm_stopped(rig):
    rig.go_red()
    rig.pipe.close(at(rig.t))
    last_two = rig.log_text().splitlines()[-2:]
    assert last_two[0].endswith("| System | alarm stopped")
    assert last_two[1].endswith("| System | session ended")


def test_close_without_an_alarm_logs_no_alarm_stopped(rig):
    rig.run(300, face(1))
    rig.pipe.close(at(rig.t))
    assert "alarm stopped" not in rig.log_text()
