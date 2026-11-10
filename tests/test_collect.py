from dataclasses import replace

import cv2
import numpy as np
import pytest

from app import collect
from app.collect import EarCollector
from app.core.config import CONFIG
from app.core.ears import ear_square
from app.core.types import Box, RawFace

FRAME_W, FRAME_H = 640, 480
LEFT_ANCHOR = (400.0, 240.0)
RIGHT_ANCHOR = (240.0, 240.0)


def frame():
    return np.full((FRAME_H, FRAME_W, 3), 127, np.uint8)


def face(yaw=0.0, far=False, ears=True):
    return RawFace(
        box=Box(0.3, 0.3, 0.3, 0.4),
        scores={},
        pitch_deg=0.0,
        yaw_deg=yaw,
        points=[],
        far=far,
        ears={"left": LEFT_ANCHOR, "right": RIGHT_ANCHOR} if ears else {},
    )


def names(root, folder):
    return sorted(p.name for p in (root / "dataset" / "ears" / folder).glob("*.png"))


def test_left_covered_sorts_each_ear(tmp_path):
    collector = EarCollector(tmp_path)
    collector.start("left_covered", "run")
    collector.offer(frame(), [face()], 0)
    assert [n.rsplit("_", 1)[1] for n in names(tmp_path, "covered")] == ["left.png"]
    assert [n.rsplit("_", 1)[1] for n in names(tmp_path, "clear")] == ["right.png"]
    assert names(tmp_path, "covered")[0].startswith("run_")


def test_turned_saves_only_facing_ear(tmp_path):
    collector = EarCollector(tmp_path)
    collector.start("both_clear", "run")
    collector.offer(frame(), [face(yaw=30)], 0)
    assert len(names(tmp_path, "clear")) == 1
    assert names(tmp_path, "clear")[0].endswith("_left.png")
    assert names(tmp_path, "covered") == []


def test_interval_limits_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "CONFIG", replace(CONFIG, collect_interval_ms=300))
    collector = EarCollector(tmp_path)
    collector.start("both_clear", "run")
    for t in (0, 100, 300):
        collector.offer(frame(), [face()], t)
    assert len(names(tmp_path, "clear")) == 4  # two frames, two ears each


def test_stopped_saves_nothing(tmp_path):
    collector = EarCollector(tmp_path)
    collector.offer(frame(), [face()], 0)
    collector.start("both_clear", "run")
    collector.stop()
    collector.offer(frame(), [face()], 1000)
    assert collector.counts() == {"clear": 0, "covered": 0}


def test_far_face_skipped(tmp_path):
    collector = EarCollector(tmp_path)
    collector.start("both_clear", "run")
    collector.offer(frame(), [face(far=True), face(ears=False)], 0)
    assert collector.counts() == {"clear": 0, "covered": 0}


def test_counts_reads_folders(tmp_path):
    collector = EarCollector(tmp_path)
    assert collector.counts() == {"clear": 0, "covered": 0}
    collector.start("right_covered", "run")
    collector.offer(frame(), [face()], 0)
    assert collector.counts() == {"clear": 1, "covered": 1}
    assert EarCollector(tmp_path).counts() == {"clear": 1, "covered": 1}


def test_saved_crop_is_native_size(tmp_path):
    collector = EarCollector(tmp_path)
    collector.start("both_clear", "run")
    collector.offer(frame(), [face()], 0)
    size = ear_square(LEFT_ANCHOR, 0.3 * FRAME_W, "left", FRAME_W, FRAME_H)[2]
    path = tmp_path / "dataset" / "ears" / "clear" / "run_00000_left.png"
    image = cv2.imread(str(path))
    assert image.shape[:2] == (size, size)


def test_failed_write_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(collect.cv2, "imwrite", lambda *a: False)
    collector = EarCollector(tmp_path)
    collector.start("both_clear", "run")
    collector.offer(frame(), [face()], 0)
    assert "Could not save" in caplog.text
