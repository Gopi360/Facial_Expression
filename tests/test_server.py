import json
import re
import socket
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

import app.server as server
from app.core.config import CONFIG
from app.models import ModelsMissing
from app.pipeline import FrameResult
from app.server import create_app

JPEG = b"\xff\xd8 pretend jpeg"
SERVED_URL = "ws://127.0.0.1:8000/ws"
HANG_LIMIT_SEC = 3  # a fake hang ends by itself, so a failing test fails instead of sticking
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class FakePipeline:
    def __init__(self, debug=False, collect=False):
        self.debug = debug
        self.collect = collect
        self.collect_labels = []
        self.frames = []
        self.snapshots = []
        self.closed = []
        self.process_error = None
        self.snapshot_warning = None
        self.hang = None  # an Event: process blocks until it is set
        self.entered = threading.Event()
        self.in_process = False
        self.overlapped = False
        self.order = []

    def process(self, jpeg, now_ms, at, delay_sec):
        if self.process_error:
            raise self.process_error
        self.in_process = True
        self.entered.set()
        if self.hang:
            self.hang.wait(HANG_LIMIT_SEC)
        self.in_process = False
        self.order.append("frame")
        self.frames.append((jpeg, now_ms, at, delay_sec))
        return FrameResult(
            count_text="1 face found",
            lines=["Person A looks neutral 80%"],
            people=[{"letter": "A", "red": False}],
            log=["16:30:00 | System | session started"],
            alarm=False,
            warning=None,
            overflow=False,
            debug=["A: live"] if self.debug else None,
        )

    def snapshot(self, at):
        self.overlapped = self.overlapped or self.in_process
        self.snapshots.append(at)
        return self.snapshot_warning

    def set_collect(self, label):
        self.collect_labels.append(label)

    def debug_samples(self):
        return [{"t_ms": 1.0, "letter": "A"}]

    def close(self, at):
        self.order.append("close")
        self.closed.append(at)


class Factory:
    def __init__(self):
        self.made = []
        self.error = None

    def __call__(self, debug, collect=False):
        if self.error:
            raise self.error
        self.made.append(FakePipeline(debug, collect))
        return self.made[-1]


@pytest.fixture
def factory():
    return Factory()


@pytest.fixture
def client(factory):
    return TestClient(create_app(factory))


def start(ws, debug=False, collect=False):
    ws.send_json({"type": "start", "debug": debug, "collect": collect})
    assert ws.receive_json() == {"type": "ready"}


def assert_closes(ws):
    with pytest.raises(WebSocketDisconnect):
        ws.receive_json()


def test_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Start camera" in response.text


def test_start_then_frame_replies_with_frame_json(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        ws.send_bytes(JPEG)
        reply = ws.receive_json()
    assert reply == {
        "type": "frame",
        "count_text": "1 face found",
        "lines": ["Person A looks neutral 80%"],
        "people": [{"letter": "A", "red": False}],
        "log": ["16:30:00 | System | session started"],
        "alarm": False,
        "warning": None,
        "overflow": False,
    }
    jpeg, now_ms, at, delay_sec = factory.made[0].frames[0]
    assert jpeg == JPEG
    assert now_ms > 0
    assert isinstance(at, datetime)
    assert delay_sec == CONFIG.default_delay_sec


def test_debug_lines_are_sent_when_started_with_debug(client):
    with client.websocket_connect("/ws") as ws:
        start(ws, debug=True)
        ws.send_bytes(JPEG)
        assert ws.receive_json()["debug"] == ["A: live"]


def test_debug_dump_returns_the_samples(client):
    with client.websocket_connect("/ws") as ws:
        start(ws, debug=True)
        ws.send_json({"type": "debug_dump"})
        assert ws.receive_json() == {
            "type": "debug_dump",
            "samples": [{"t_ms": 1.0, "letter": "A"}],
        }


def test_collect_message_needs_collect_session(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        ws.send_json({"type": "collect", "label": "left_covered"})
        assert ws.receive_json() == {"type": "error", "message": server.BAD_MESSAGE}
    assert factory.made[0].collect_labels == []


def test_collect_session_passes_label(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws, collect=True)
        ws.send_json({"type": "collect", "label": "left_covered"})
        ws.send_json({"type": "collect", "label": None})
        ws.send_bytes(JPEG)
        assert ws.receive_json()["type"] == "frame"
    assert factory.made[0].collect is True
    assert factory.made[0].collect_labels == ["left_covered", None]


def test_collect_message_with_unknown_label_gets_an_error(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws, collect=True)
        ws.send_json({"type": "collect", "label": "nonsense"})
        assert ws.receive_json() == {"type": "error", "message": server.BAD_MESSAGE}
    assert factory.made[0].collect_labels == []


def test_second_client_is_told_busy(client, factory):
    with client.websocket_connect("/ws") as first:
        start(first)
        with client.websocket_connect("/ws") as second:
            assert second.receive_json() == {"type": "busy"}
            assert_closes(second)
        first.send_bytes(JPEG)
        assert first.receive_json()["type"] == "frame"
    assert len(factory.made) == 1


def test_disconnect_closes_session_and_next_client_is_accepted(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
    assert len(factory.made[0].closed) == 1

    with client.websocket_connect("/ws") as ws:
        start(ws)
    assert len(factory.made) == 2
    assert len(factory.made[1].closed) == 1


def test_delay_setting_reaches_the_pipeline(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        ws.send_json({"type": "settings", "delaySec": 7})
        ws.send_bytes(JPEG)
        ws.receive_json()
    assert factory.made[0].frames[0][3] == 7


@pytest.mark.parametrize(
    "sent, expected",
    [(0, CONFIG.min_delay_sec), (99, CONFIG.max_delay_sec), (10**400, CONFIG.max_delay_sec)],
)
def test_delay_setting_is_clamped(client, factory, sent, expected):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        ws.send_json({"type": "settings", "delaySec": sent})
        ws.send_bytes(JPEG)
        ws.receive_json()
    assert factory.made[0].frames[0][3] == expected


@pytest.mark.parametrize("value", ['"5"', "true", "null", "NaN", "[5]"])
def test_bad_delay_setting_gets_an_error_and_keeps_the_old_delay(client, factory, value):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        ws.send_text('{"type": "settings", "delaySec": ' + value + "}")
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(JPEG)
        assert ws.receive_json()["type"] == "frame"
    assert factory.made[0].frames[0][3] == CONFIG.default_delay_sec


def test_snapshot_message_reaches_the_pipeline(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        ws.send_json({"type": "snapshot"})
        ws.send_bytes(JPEG)
        assert ws.receive_json()["type"] == "frame"  # no reply of its own when it worked
    assert len(factory.made[0].snapshots) == 1


def test_snapshot_warning_is_sent_and_the_session_carries_on(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        factory.made[0].snapshot_warning = "Could not save the snap: disk full"
        ws.send_json({"type": "snapshot"})
        assert ws.receive_json() == {
            "type": "error",
            "message": "Could not save the snap: disk full",
        }
        ws.send_bytes(JPEG)
        assert ws.receive_json()["type"] == "frame"


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '"start"', "{}", '{"type": "dance"}'])
def test_bad_message_gets_an_error_and_the_session_carries_on(client, text):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        ws.send_text(text)
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(JPEG)
        assert ws.receive_json()["type"] == "frame"


@pytest.mark.parametrize(
    "send",
    [
        lambda ws: ws.send_bytes(JPEG),
        lambda ws: ws.send_json({"type": "snapshot"}),
        lambda ws: ws.send_json({"type": "debug_dump"}),
    ],
)
def test_session_messages_before_start_get_an_error(client, send):
    with client.websocket_connect("/ws") as ws:
        send(ws)
        assert ws.receive_json()["type"] == "error"
        start(ws)


def test_second_start_gets_an_error_and_keeps_the_session(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        ws.send_json({"type": "start", "debug": False})
        assert ws.receive_json()["type"] == "error"
    assert len(factory.made) == 1


def test_missing_models_sends_error_message(client, factory):
    factory.error = ModelsMissing("Missing model file. Run: python scripts/setup_models.py")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "debug": False})
        assert ws.receive_json() == {
            "type": "error",
            "message": "Missing model file. Run: python scripts/setup_models.py",
        }
        assert_closes(ws)

    factory.error = None
    with client.websocket_connect("/ws") as ws:
        start(ws)


def test_pipeline_exception_sends_error_and_closes_session(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        factory.made[0].process_error = ValueError("boom")
        ws.send_bytes(JPEG)
        reply = ws.receive_json()
        assert reply["type"] == "error"
        assert "boom" not in reply["message"]
        assert_closes(ws)
    assert len(factory.made[0].closed) == 1

    with client.websocket_connect("/ws") as ws:
        start(ws)


def test_failing_close_still_frees_the_app_for_the_next_client(client, factory):
    with client.websocket_connect("/ws") as ws:
        start(ws)
        factory.made[0].close = lambda at: 1 / 0

    with client.websocket_connect("/ws") as ws:
        start(ws)


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "http://evil.example:8000",
        "http://127.0.0.1:9999",
        "http://localhost:9999",
        "https://127.0.0.1:8000",
        "http://127.0.0.1:8000.evil.example",
        "null",
        "",
    ],
)
def test_foreign_origin_is_rejected(client, factory, origin):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(SERVED_URL, headers={"origin": origin}):
            pass
    assert factory.made == []


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8000", "http://localhost:8000"])
def test_own_origin_is_accepted(client, origin):
    with client.websocket_connect(SERVED_URL, headers={"origin": origin}) as ws:
        start(ws)


def test_rejected_origin_does_not_take_the_session(client, factory):
    foreign = {"origin": "http://evil.example"}
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(SERVED_URL, headers=foreign):
            pass

    with client.websocket_connect(SERVED_URL) as ws:
        start(ws)  # the rejected client left no session behind
        with pytest.raises(WebSocketDisconnect):  # refused outright, not told busy
            with client.websocket_connect(SERVED_URL, headers=foreign):
                pass
        ws.send_bytes(JPEG)
        assert ws.receive_json()["type"] == "frame"
    assert len(factory.made) == 1
    assert len(factory.made[0].closed) == 1


@pytest.fixture
def real_port(factory):
    """A real uvicorn, the way the app is run. TestClient never needs a WebSocket library."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    uvicorn_server = uvicorn.Server(uvicorn.Config(create_app(factory), log_config=None))
    thread = threading.Thread(target=uvicorn_server.run, args=([sock],), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uvicorn_server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert uvicorn_server.started
    yield sock.getsockname()[1]
    uvicorn_server.should_exit = True
    thread.join(10)
    sock.close()


def test_real_server_runs_a_session_over_a_real_websocket(real_port, factory):
    origin = f"http://127.0.0.1:{real_port}"
    with connect(f"ws://127.0.0.1:{real_port}/ws", origin=origin, proxy=None) as ws:
        ws.send(json.dumps({"type": "start", "debug": False}))
        assert json.loads(ws.recv(timeout=5)) == {"type": "ready"}
        ws.send(JPEG)
        assert json.loads(ws.recv(timeout=5))["type"] == "frame"
    assert factory.made[0].frames[0][0] == JPEG


def test_real_server_refuses_a_foreign_origin(real_port, factory):
    with pytest.raises(InvalidStatus) as refused:
        connect(f"ws://127.0.0.1:{real_port}/ws", origin="http://evil.example", proxy=None)
    assert refused.value.response.status_code == 403
    assert factory.made == []


def test_hung_pipeline_releases_the_session(client, factory, monkeypatch):
    monkeypatch.setattr(server, "CONFIG", replace(CONFIG, reply_timeout_ms=200))
    stuck = threading.Event()
    with client.websocket_connect("/ws") as ws:
        start(ws)
        factory.made[0].hang = stuck
        ws.send_bytes(JPEG)
        assert ws.receive_json() == {"type": "error", "message": server.STALLED_MESSAGE}
        assert_closes(ws)
    assert len(factory.made[0].closed) == 1

    with client.websocket_connect("/ws") as ws:
        start(ws)  # accepted while the old worker thread is still stuck
        assert not stuck.is_set()
    stuck.set()


def real_connect(port):
    return connect(f"ws://127.0.0.1:{port}/ws", origin=f"http://127.0.0.1:{port}", proxy=None)


def real_start(ws):
    ws.send(json.dumps({"type": "start", "debug": False}))
    assert json.loads(ws.recv(timeout=5)) == {"type": "ready"}


def wait_until(check, seconds):
    deadline = time.monotonic() + seconds
    while not check() and time.monotonic() < deadline:
        time.sleep(0.01)
    return check()


# TestClient cancels the app when its block ends, so only a real server shows a real disconnect.
def test_closed_socket_closes_the_session(real_port, factory):
    with real_connect(real_port) as ws:
        real_start(ws)
    assert wait_until(lambda: factory.made[0].closed, 2)


def test_closed_socket_releases_a_hung_session_before_the_timeout(real_port, factory, monkeypatch):
    monkeypatch.setattr(
        server, "CONFIG", replace(CONFIG, reply_timeout_ms=60000, frame_finish_grace_ms=50)
    )
    stuck = threading.Event()
    with real_connect(real_port) as ws:
        real_start(ws)
        factory.made[0].hang = stuck
        ws.send(JPEG)
        assert factory.made[0].entered.wait(2)
    assert wait_until(lambda: factory.made[0].closed, HANG_LIMIT_SEC / 2)

    with real_connect(real_port) as ws:
        real_start(ws)
    assert not stuck.is_set()
    stuck.set()


def test_close_waits_for_a_frame_that_is_about_to_finish(real_port, factory):
    finishing = threading.Event()
    with real_connect(real_port) as ws:
        real_start(ws)
        factory.made[0].hang = finishing
        ws.send(JPEG)
        assert factory.made[0].entered.wait(2)
    time.sleep(0.1)  # the server has seen the close by now, and the frame is still running
    finishing.set()
    assert wait_until(lambda: factory.made[0].closed, 2)
    assert factory.made[0].order == ["frame", "close"]


def test_snapshot_sent_during_a_frame_waits_for_it(client, factory):
    finishing = threading.Event()
    with client.websocket_connect("/ws") as ws:
        start(ws)
        factory.made[0].hang = finishing
        ws.send_bytes(JPEG)
        assert factory.made[0].entered.wait(2)
        ws.send_json({"type": "snapshot"})
        ws.send_json({"type": "settings", "delaySec": 7})
        time.sleep(0.1)
        assert factory.made[0].snapshots == []
        finishing.set()
        assert ws.receive_json()["type"] == "frame"
        ws.send_bytes(JPEG)
        assert ws.receive_json()["type"] == "frame"
    assert len(factory.made[0].snapshots) == 1
    assert not factory.made[0].overlapped
    assert factory.made[0].frames[1][3] == 7


def test_config_endpoint_matches_config(client):
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json() == {
        "target_fps": CONFIG.target_fps,
        "max_faces": CONFIG.max_faces,
        "min_delay_sec": CONFIG.min_delay_sec,
        "max_delay_sec": CONFIG.max_delay_sec,
        "default_delay_sec": CONFIG.default_delay_sec,
        "paused_note_ms": CONFIG.paused_note_ms,
        "warning_hold_ms": CONFIG.warning_hold_ms,
        "reply_timeout_ms": CONFIG.reply_timeout_ms,
        "overlay_line_frac": CONFIG.overlay_line_frac,
        "overlay_letter_frac": CONFIG.overlay_letter_frac,
        "frame_width": CONFIG.frame_width,
        "frame_height": CONFIG.frame_height,
        "jpeg_quality": CONFIG.jpeg_quality,
        "beep_hz": CONFIG.beep_hz,
        "beep_on_ms": CONFIG.beep_on_ms,
        "beep_off_ms": CONFIG.beep_off_ms,
        "beep_peak_gain": CONFIG.beep_peak_gain,
        "low_fps_warning": CONFIG.low_fps_warning,
        "low_fps_hold_ms": CONFIG.low_fps_hold_ms,
    }


def test_page_warns_when_it_runs_slowly():
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "Running slowly, so sleep detection may be late." in source
    assert "config.low_fps_warning" in source
    assert "config.low_fps_hold_ms" in source


def test_page_has_no_copy_of_config_values():
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = re.sub(r"'[^'\n]*'|`[^`]*`", "", code)
    code = re.sub(r"//.*", "", code)
    numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", code))
    # 0 and 1 are logic, 100 makes a percentage and 1000 is milliseconds in a second.
    assert numbers <= {"0", "1", "100", "1000"}

    assert not re.search(r"\d+ largest", source)  # a number in a message is still a copy

    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    delay_input = re.search(r'<input id="delay"[^>]*>', page).group()
    assert not re.search(r"[ ](min|max|value)=", delay_input)


class FakeModel:
    loads = 0

    def __init__(self, models_dir):
        type(self).loads += 1


class FakeFinder(FakeModel):
    pass


class FakeEmotion(FakeModel):
    pass


class FakeSigner(FakeModel):
    pass


@pytest.fixture
def real_factory(monkeypatch, tmp_path):
    for cls in (FakeFinder, FakeEmotion, FakeSigner):
        cls.loads = 0
    monkeypatch.setattr(server, "FaceFinder", FakeFinder)
    monkeypatch.setattr(server, "EmotionModel", FakeEmotion)
    monkeypatch.setattr(server, "FaceSigner", FakeSigner)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "_models", None)
    return tmp_path


def test_models_load_once_and_each_session_gets_a_new_pipeline(real_factory):
    first = server.make_pipeline(False, False)
    second = server.make_pipeline(False, False)
    assert first is not second
    assert (FakeFinder.loads, FakeEmotion.loads, FakeSigner.loads) == (1, 1, 1)
    assert list((real_factory / "logs").glob("session-*.log"))


def test_failed_model_load_is_tried_again(real_factory, monkeypatch):
    def missing(models_dir):
        raise ModelsMissing("missing")

    monkeypatch.setattr(server, "FaceSigner", missing)
    with pytest.raises(ModelsMissing):
        server.make_pipeline(False, False)

    monkeypatch.setattr(server, "FaceSigner", FakeSigner)
    server.make_pipeline(False, False)
