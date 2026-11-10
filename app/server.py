"""The local web app: serves the page and runs one camera session over a WebSocket."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .core.config import CONFIG
from .models import EmotionModel, FaceFinder, FaceSigner, ModelsMissing
from .pipeline import Pipeline
from .session_files import SessionFiles

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

POLICY_VIOLATION = 1008  # WebSocket close code

BAD_MESSAGE = "The app did not understand that message."
BAD_DELAY = "The delay setting must be a number."
NOT_STARTED = "The session has not started yet."
ALREADY_STARTED = "The session has already started."
FATAL_MESSAGE = "The app hit a problem and stopped this session. Reload the page to try again."
STALLED_MESSAGE = "The app stopped responding. Reload the page, and restart the app if this repeats."

# What the page reads from GET /config, so it holds no copy of these numbers.
PAGE_CONFIG = (
    "target_fps",
    "max_faces",
    "min_delay_sec",
    "max_delay_sec",
    "default_delay_sec",
    "paused_note_ms",
    "warning_hold_ms",
    "reply_timeout_ms",
    "overlay_line_frac",
    "overlay_letter_frac",
    "frame_width",
    "frame_height",
    "jpeg_quality",
    "beep_hz",
    "beep_on_ms",
    "beep_off_ms",
    "beep_peak_gain",
    "low_fps_warning",
    "low_fps_hold_ms",
)

_models: tuple[FaceFinder, EmotionModel, FaceSigner] | None = None
_models_lock = threading.Lock()


def _loaded_models() -> tuple[FaceFinder, EmotionModel, FaceSigner]:
    """Load the models on first use and keep them for every later session."""
    global _models
    with _models_lock:
        if _models is None:
            models_dir = ROOT / "models"
            _models = (FaceFinder(models_dir), EmotionModel(models_dir), FaceSigner(models_dir))
        return _models


def make_pipeline(debug: bool) -> Pipeline:
    finder, emotion, signer = _loaded_models()
    return Pipeline(finder, emotion, signer, SessionFiles(ROOT, datetime.now()), debug)


def _is_number(value: object) -> bool:
    is_numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    return is_numeric and value == value  # NaN is the one number not equal to itself


def _origin_allowed(socket: WebSocket) -> bool:
    """Only this app's own page. Browsers always send Origin, so no header means no browser."""
    origin = socket.headers.get("origin")
    if origin is None:
        return True
    port = socket.scope["server"][1]  # the port really bound, which a request cannot fake
    return origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")


class _Stalled(Exception):
    """A pipeline call did not finish in time."""


class _Session:
    """One socket. Frames and control messages are handled one at a time, in order.

    A reader task takes messages off the socket into an inbox, so a close is seen even while
    a pipeline call is running. Only run() acts on the inbox, which keeps the calls in turn.
    """

    def __init__(self, socket: WebSocket, make: Callable[[bool], Pipeline]):
        self._socket = socket
        self._make = make
        self._pipeline: Pipeline | None = None
        self._debug = False
        self._delay_sec: float = CONFIG.default_delay_sec
        self._inbox: asyncio.Queue[dict] = asyncio.Queue()
        self._gone = asyncio.Event()

    async def run(self) -> None:
        reader = asyncio.create_task(self._read())
        try:
            while True:
                message = await self._inbox.get()
                if self._gone.is_set():
                    return
                frame = message.get("bytes")
                if frame is not None:
                    await self._on_frame(frame)
                else:
                    await self._on_text(message.get("text"))
        except WebSocketDisconnect:
            pass
        except ModelsMissing as e:
            await self._stop(str(e))
        except _Stalled:
            logger.error("A pipeline call passed %d ms", CONFIG.reply_timeout_ms)
            await self._stop(STALLED_MESSAGE)
        except Exception:
            logger.exception("Session failed")
            await self._stop(FATAL_MESSAGE)
        finally:
            self._close()
            reader.cancel()

    async def _read(self) -> None:
        try:
            while True:
                message = await self._socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                self._inbox.put_nowait(message)
        except Exception:
            logger.exception("Could not read from the socket")
        finally:
            self._gone.set()
            self._inbox.put_nowait({})  # wakes run(), which then sees the session has gone

    async def _guarded(self, call: Callable, *args: object) -> object:
        """Run a pipeline call in a thread, and give up on it if it hangs or the client leaves."""
        work = asyncio.create_task(asyncio.to_thread(call, *args))
        work.add_done_callback(lambda done: done.cancelled() or done.exception())
        gone = asyncio.create_task(self._gone.wait())
        try:
            await asyncio.wait(
                {work, gone},
                timeout=CONFIG.reply_timeout_ms / 1000,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            gone.cancel()
        if self._gone.is_set():
            # A healthy call ends in moments. Waiting for it keeps close from overlapping it.
            await asyncio.wait({work}, timeout=CONFIG.frame_finish_grace_ms / 1000)
            work.cancel()
            raise WebSocketDisconnect()
        if not work.done():
            work.cancel()  # a stuck thread cannot be stopped, only left behind
            raise _Stalled()
        return work.result()

    async def _on_frame(self, jpeg: bytes) -> None:
        if self._pipeline is None:
            await self._error(NOT_STARTED)
            return
        result = await self._guarded(
            self._pipeline.process, jpeg, time.monotonic() * 1000, datetime.now(), self._delay_sec
        )
        reply = {"type": "frame", **asdict(result)}
        if not self._debug:
            del reply["debug"]
        await self._socket.send_json(reply)

    async def _on_text(self, text: str | None) -> None:
        try:
            message = json.loads(text)
            kind = message["type"]
        except (TypeError, ValueError, KeyError):
            await self._error(BAD_MESSAGE)
            return

        if kind == "start":
            await self._on_start(message)
        elif kind == "settings":
            await self._on_settings(message)
        elif kind not in ("snapshot", "debug_dump"):
            await self._error(BAD_MESSAGE)
        elif self._pipeline is None:
            await self._error(NOT_STARTED)
        elif kind == "snapshot":
            warning = await self._guarded(self._pipeline.snapshot, datetime.now())
            if warning:
                await self._error(warning)
        else:
            samples = self._pipeline.debug_samples()
            await self._socket.send_json({"type": "debug_dump", "samples": samples})

    async def _on_start(self, message: dict) -> None:
        if self._pipeline is not None:
            await self._error(ALREADY_STARTED)
            return
        self._debug = message.get("debug") is True
        self._pipeline = await asyncio.to_thread(self._make, self._debug)
        await self._socket.send_json({"type": "ready"})

    async def _on_settings(self, message: dict) -> None:
        delay = message.get("delaySec")
        if not _is_number(delay):
            await self._error(BAD_DELAY)
            return
        self._delay_sec = min(max(delay, CONFIG.min_delay_sec), CONFIG.max_delay_sec)

    async def _error(self, text: str) -> None:
        await self._socket.send_json({"type": "error", "message": text})

    async def _stop(self, text: str) -> None:
        """Say why the session is ending. The client may already be gone."""
        with suppress(WebSocketDisconnect, RuntimeError):
            await self._error(text)
            await self._socket.close()

    def _close(self) -> None:
        """Runs on every exit, so it must not stop the app being freed for the next client."""
        if self._pipeline is None:
            return
        try:
            self._pipeline.close(datetime.now())
        except Exception:
            logger.exception("Could not close the session")


def create_app(make: Callable[[bool], Pipeline]) -> FastAPI:
    app = FastAPI()
    session_open = False

    @app.get("/config")
    def page_config() -> dict:
        return {name: getattr(CONFIG, name) for name in PAGE_CONFIG}

    @app.websocket("/ws")
    async def session(socket: WebSocket) -> None:
        nonlocal session_open
        if not _origin_allowed(socket):
            logger.warning("Refused a connection from origin %s", socket.headers.get("origin"))
            await socket.close(code=POLICY_VIOLATION)
            return
        await socket.accept()
        if session_open:
            with suppress(WebSocketDisconnect, RuntimeError):
                await socket.send_json({"type": "busy"})
                await socket.close()
            return
        session_open = True
        try:
            await _Session(socket, make).run()
        finally:
            session_open = False

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app(make_pipeline)
