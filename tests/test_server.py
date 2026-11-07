from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app.server as server
from app.core.config import CONFIG
from app.models import ModelsMissing
from app.pipeline import FrameResult
from app.server import create_app

JPEG = b"\xff\xd8 pretend jpeg"


class FakePipeline:
    def __init__(self, debug=False):
        self.debug = debug
        self.frames = []
        self.snapshots = []
        self.closed = []
        self.process_error = None
        self.snapshot_warning = None

    def process(self, jpeg, now_ms, at, delay_sec):
        if self.process_error:
            raise self.process_error
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
        self.snapshots.append(at)
        return self.snapshot_warning

    def debug_samples(self):
        return [{"t_ms": 1.0, "letter": "A"}]

    def close(self, at):
        self.closed.append(at)


class Factory:
    def __init__(self):
        self.made = []
        self.error = None

    def __call__(self, debug):
        if self.error:
            raise self.error
        self.made.append(FakePipeline(debug))
        return self.made[-1]


@pytest.fixture
def factory():
    return Factory()


@pytest.fixture
def client(factory):
    return TestClient(create_app(factory))


def start(ws, debug=False):
    ws.send_json({"type": "start", "debug": debug})
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
    first = server.make_pipeline(False)
    second = server.make_pipeline(False)
    assert first is not second
    assert (FakeFinder.loads, FakeEmotion.loads, FakeSigner.loads) == (1, 1, 1)
    assert list((real_factory / "logs").glob("session-*.log"))


def test_failed_model_load_is_tried_again(real_factory, monkeypatch):
    def missing(models_dir):
        raise ModelsMissing("missing")

    monkeypatch.setattr(server, "FaceSigner", missing)
    with pytest.raises(ModelsMissing):
        server.make_pipeline(False)

    monkeypatch.setattr(server, "FaceSigner", FakeSigner)
    server.make_pipeline(False)
