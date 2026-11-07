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


class _Session:
    """One socket. Frames and control messages are handled one at a time, in order."""

    def __init__(self, socket: WebSocket, make: Callable[[bool], Pipeline]):
        self._socket = socket
        self._make = make
        self._pipeline: Pipeline | None = None
        self._debug = False
        self._delay_sec: float = CONFIG.default_delay_sec

    async def run(self) -> None:
        try:
            while True:
                message = await self._socket.receive()
                if message["type"] == "websocket.disconnect":
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
        except Exception:
            logger.exception("Session failed")
            await self._stop(FATAL_MESSAGE)
        finally:
            self._close()

    async def _on_frame(self, jpeg: bytes) -> None:
        if self._pipeline is None:
            await self._error(NOT_STARTED)
            return
        result = await asyncio.to_thread(
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
            warning = await asyncio.to_thread(self._pipeline.snapshot, datetime.now())
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
