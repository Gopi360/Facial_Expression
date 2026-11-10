from datetime import datetime

import numpy as np

from app.core.types import Box, LogEvent, Person, Reading
from app.session_files import GREEN_BGR, RED_BGR, SessionFiles, format_line, render_snap

START = datetime(2026, 11, 6, 16, 30, 0)
AT = datetime(2026, 11, 6, 16, 42, 11)
LOG = "session-2026-11-06_16-30-00.log"


def blank(h=8, w=8):
    return np.zeros((h, w, 3), np.uint8)


def test_log_line_format_with_snap():
    ev = LogEvent(AT, "Person A", "sleeping", 92, "snaps/2026-11-06_16-30-00_16-42-11_A_sleeping.jpg")
    assert format_line(ev) == (
        "16:42:11 | Person A | sleeping | 92% | snaps/2026-11-06_16-30-00_16-42-11_A_sleeping.jpg"
    )


def test_log_line_format_without_confidence():
    assert format_line(LogEvent(AT, "System", "alarm started")) == "16:42:11 | System | alarm started"


def test_each_session_has_its_own_file(tmp_path):
    a = SessionFiles(tmp_path, START)
    b = SessionFiles(tmp_path, datetime(2026, 11, 6, 17, 0, 0))
    a.write(LogEvent(AT, "System", "session started"))
    b.write(LogEvent(AT, "System", "session started"))
    names = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert names == [LOG, "session-2026-11-06_17-00-00.log"]


def test_line_is_on_disk_before_close(tmp_path):
    s = SessionFiles(tmp_path, START)
    assert s.write(LogEvent(AT, "Person A", "joined")) is None
    path = tmp_path / "logs" / LOG
    assert "16:42:11 | Person A | joined" in path.read_text(encoding="utf-8")
    s.close(AT)
    assert "session ended" in path.read_text(encoding="utf-8")


def test_snap_saved_with_expected_name(tmp_path):
    s = SessionFiles(tmp_path, START)
    rel, warn = s.save_snap(blank(48, 64), AT, "A", "sleeping")
    assert warn is None
    assert rel == "snaps/2026-11-06_16-30-00_16-42-11_A_sleeping.jpg"
    assert (tmp_path / rel).stat().st_size > 0


def test_manual_snap_name(tmp_path):
    s = SessionFiles(tmp_path, START)
    rel, _ = s.save_snap(blank(), AT, "all", "manual")
    assert rel == "snaps/2026-11-06_16-30-00_16-42-11_all_manual.jpg"


def test_question_mark_person_snap_uses_unknown(tmp_path):
    s = SessionFiles(tmp_path, START)
    rel, warn = s.save_snap(blank(), AT, "?", "sleeping")
    assert warn is None
    assert rel == "snaps/2026-11-06_16-30-00_16-42-11_unknown_sleeping.jpg"
    assert (tmp_path / rel).exists()


def test_unwritable_log_returns_warning_and_does_not_raise(tmp_path):
    (tmp_path / "logs").write_text("a file, not a folder")
    s = SessionFiles(tmp_path, START)
    warning = s.write(LogEvent(AT, "Person A", "joined"))
    assert isinstance(warning, str) and warning
    s.close(AT)  # must not raise either


def test_unwritable_snap_returns_warning_and_does_not_raise(tmp_path):
    (tmp_path / "snaps").write_text("a file, not a folder")
    s = SessionFiles(tmp_path, START)
    rel, warn = s.save_snap(blank(), AT, "A", "sleeping")
    assert rel is None and isinstance(warn, str) and warn


def person(label="neutral", x=0.1, visible=True, letter="A"):
    return Person(letter, Box(x, 0.2, 0.2, 0.4), Reading(label, 90), visible)


def test_render_snap_mirrors_box_position():
    frame = np.zeros((100, 200, 3), np.uint8)
    out = render_snap(frame, [person(x=0.1)])
    cols = np.argwhere(out.sum(axis=2) > 0)[:, 1]
    assert cols.size and cols.min() > 100  # box was on the left, now on the right
    assert frame.sum() == 0  # input untouched


def test_render_snap_skips_hidden_people():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert render_snap(frame, [person(visible=False)]).sum() == 0


def test_render_snap_uses_red_for_sleeping_and_green_otherwise():
    frame = np.zeros((100, 200, 3), np.uint8)
    red = {tuple(p) for p in render_snap(frame, [person("sleeping")]).reshape(-1, 3)}
    green = {tuple(p) for p in render_snap(frame, [person("neutral")]).reshape(-1, 3)}
    assert (68, 68, 239) in red
    assert (94, 197, 34) in green


def test_snap_box_red_for_ears():
    frame = np.zeros((100, 200, 3), np.uint8)
    hidden = person("neutral")
    hidden.ears_hidden = True
    assert RED_BGR in {tuple(p) for p in render_snap(frame, [hidden]).reshape(-1, 3)}
    assert GREEN_BGR not in {tuple(p) for p in render_snap(frame, [hidden]).reshape(-1, 3)}
