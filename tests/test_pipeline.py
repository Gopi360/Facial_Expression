from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

import app.pipeline as pipeline_module
from app.core.config import CONFIG
from app.core.types import Box, RawFace
from app.pipeline import Pipeline
from app.session_files import SessionFiles, format_line

START = datetime(2026, 11, 6, 16, 30, 0)
STEP_MS = 100
SIGNATURE_SIZE = 40
JPEG = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))[1].tobytes()


def at(ms):
    return START + timedelta(milliseconds=ms)


def face(marker=1, x=0.1, closed=False, yaw=0.0):
    blink = 0.95 if closed else 0.05
    scores = {"eyeBlinkLeft": blink, "eyeBlinkRight": blink, "marker": marker}
    return RawFace(Box(x, 0.1, 0.2, 0.2), scores, 0.0, yaw, [(0.0, 0.0)] * 5)


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
    rig.run(CONFIG.identity_interval_ms + 100, face(1, yaw=-CONFIG.frontal_yaw_deg))
    assert len(rig.pipe._identity._signatures["A"]) == 1


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


def test_log_uses_the_shared_formatter():
    assert pipeline_module.format_line is format_line


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
