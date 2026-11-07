"""Joins the models and the rules: one JPEG in, one result out."""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import cv2
import numpy as np

from .core.classifier import PersonClassifier
from .core.config import CONFIG
from .core.debug_log import WATCHED, DebugBuffer
from .core.events import diff_events, is_red, should_sound
from .core.identity import UNKNOWN, Identity, IdentityDecision
from .core.signals import Baseline, raw_eye_closure
from .core.snap_policy import SnapPolicy
from .core.text import face_count_text, person_line
from .core.tracker import Tracker
from .core.types import Box, FarDetectFailed, LogEvent, Person, RawFace, Reading, Signals
from .session_files import SessionFiles, format_line, render_snap

logger = logging.getLogger(__name__)

EMPTY_FRAME_WARNING = "Skipped an empty camera frame."
LARGE_FRAME_WARNING = "Skipped a camera frame that was too large."
BAD_FRAME_WARNING = "Skipped a camera frame that could not be read."
DETECT_WARNING = "Face detection stopped working. Reload the page to try again."
FAR_WARNING = "Could not look for far faces on this frame."
FACE_WARNING = "One face could not be read on this frame."
NO_FRAME_WARNING = "There is no camera frame to save yet."
SNAP_WARNING = "Could not save the snap"

WAITING_READING = Reading("neutral", 0)  # shown for a face with no letter, never classified
FAR_READING = Reading("far", 0)  # nothing can be read at this size, so the number means nothing
MANUAL_LETTER = "all"
MANUAL_REASON = "manual"


@dataclass
class FrameResult:
    count_text: str
    lines: list[str]
    people: list[dict]
    log: list[str]  # the same lines as the session file, newest first
    alarm: bool
    warning: str | None
    overflow: bool
    debug: list[str] | None


@dataclass
class _PersonState:
    """What is remembered for one person, for the whole session."""

    baseline: Baseline = field(default_factory=Baseline)
    classifier: PersonClassifier = field(default_factory=PersonClassifier)
    emotion: dict[str, float] | None = None
    emotion_tried_ms: float = -math.inf
    box: Box | None = None
    reading: Reading | None = None


@dataclass
class _FarRun:
    """One track's far views in a row. Once confirmed, it keeps through a missed frame."""

    since_ms: float
    views: int = 0
    confirmed: bool = False


@dataclass
class _RedProof:
    """The last frame that showed a red person, ready to save if they are lost before a snap."""

    frame: np.ndarray
    people: list[Person]
    person: Person
    at: datetime


def _decode(jpeg: bytes) -> tuple[np.ndarray | None, str | None]:
    if not jpeg:
        return None, EMPTY_FRAME_WARNING
    if len(jpeg) > CONFIG.jpeg_max_bytes:
        return None, LARGE_FRAME_WARNING
    try:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        frame = None
    return (frame, None) if frame is not None else (None, BAD_FRAME_WARNING)


def _shown_letter(key: str) -> str:
    """A state key past Z carries its track id. People only ever see the ?."""
    return UNKNOWN if key.startswith(UNKNOWN) else key


def _shown_event(event: LogEvent) -> LogEvent:
    if event.who.startswith(f"Person {UNKNOWN}"):
        return replace(event, who=f"Person {UNKNOWN}")
    return event


def _person_dict(p: Person) -> dict:
    red = is_red(p.reading.label)
    return {
        "letter": p.letter,
        "box": asdict(p.box),
        "label": p.reading.label,
        "confidence": p.reading.confidence,
        "red": red,
        "reason": person_line(p) if red else None,
        "visible": p.visible,
    }


def _drowsy_text(info: dict) -> str:
    if info["strong_sustained"]:
        return "strong expression blocking"
    sign = f"sign {info['drowsy_sign']}" if info["drowsy_sign"] else "no sign"
    return f"{sign}, strong expression delays sleeping" if info["strong_held"] else sign


def _identity_text(decision: IdentityDecision) -> str:
    def named(letter: str | None, score: float | None) -> str:
        return "none" if letter is None else f"{letter} {score:.2f}"

    best = named(decision.best_letter, decision.best_score)
    runner_up = named(decision.runner_up_letter, decision.runner_up_score)
    return f"best {best}, runner-up {runner_up}, {decision.outcome}"


def _number(value: float | None) -> str:
    return "learning" if value is None else f"{value:.2f}"


def _debug_lines(
    letter: str,
    signals: Signals,
    ranges: dict | None,
    sample: dict,
    decision: IdentityDecision | None,
) -> list[str]:
    live = "  ".join(f"{name} {value:.2f}" for name, value in asdict(signals).items())
    lines = [f"{letter}: {live}"]
    if ranges:
        parts = "  ".join(
            f"{name} {ranges[name]['min']:.2f} to {ranges[name]['max']:.2f}" for name in WATCHED
        )
        lines.append(f"{letter} last {CONFIG.debug_range_ms // 1000}s: {parts}")
    lines.append(f"{letter} drowsy: {_drowsy_text(sample)}")
    lines.append(
        f"{letter} sleep: closed {sample['closed_ms']:.0f} ms, "
        f"closed at {sample['closed_threshold']:.2f}"
    )
    lines.append(
        f"{letter} raw: eyeBlink {sample['raw_eye_blink']:.2f}  "
        f"pitch {sample['pitch_deg']:.1f}  yaw {sample['yaw_deg']:.1f}  "
        f"baseline closure {_number(sample['baseline_closure'])}  "
        f"pitch {_number(sample['baseline_pitch'])}"
    )
    if decision is not None:
        lines.append(f"{letter} identity: {_identity_text(decision)}")
    return lines


class Pipeline:
    def __init__(self, finder, emotion, signer, files: SessionFiles, debug: bool = False):
        self._finder = finder
        self._emotion = emotion
        self._signer = signer
        self._files = files
        self._debug_on = debug
        self._tracker = Tracker()
        self._identity = Identity()
        self._snaps = SnapPolicy()
        self._debug_buffer = DebugBuffer()
        self._people: dict[str, _PersonState] = {}  # by state key, never deleted
        self._signature_failed_ms: dict[int, float] = {}
        self._signature_tried_ms: dict[int, float] = {}
        self._far_runs: dict[int, _FarRun] = {}  # by track
        self._crowded_frames = 0
        self._labels: dict[str, str] = {}
        self._red_proof: dict[str, _RedProof] = {}
        self._decisions: dict[str, IdentityDecision] = {}  # the latest, for the debug view
        self._alarm = False
        self._log: list[str] = []
        self._frame: np.ndarray | None = None
        self._shown: list[Person] = []
        self._pending_warnings = self._record(
            [LogEvent(files.started, "System", "session started")]
        )
        self._last = FrameResult(
            face_count_text(0), [], [], list(self._log), False, None, False, [] if debug else None
        )

    def process(self, jpeg: bytes, now_ms: float, at: datetime, delay_sec: float) -> FrameResult:
        frame, problem = _decode(jpeg)
        if frame is None:
            return replace(self._last, warning=problem)

        warnings, self._pending_warnings = self._pending_warnings, []
        faces = self._find(frame, warnings)
        seen = dict(self._tracker.update(faces, now_ms))
        active_ids = self._tracker.active_ids()
        tracked = self._without_unconfirmed(seen, active_ids, now_ms)
        self._identity.sync(set(tracked), active_ids, now_ms)
        self._heavy_check(frame, tracked, now_ms, warnings)

        debug: list[str] = []
        present = self._read_people(tracked, active_ids, now_ms, delay_sec, debug)
        waiting = [
            Person(None, face.box, WAITING_READING, True)
            for track_id, face in tracked.items()
            if self._identity.state_key(track_id) is None
        ]
        people = [*present.values(), *waiting]

        events = self._events(frame, present, people, now_ms, at, warnings)
        warnings += self._record(events)

        self._frame = frame
        self._shown = people
        self._last = FrameResult(
            count_text=face_count_text(len(people)),
            lines=[person_line(p) for p in people],
            people=[_person_dict(p) for p in people],
            log=list(self._log),
            alarm=self._alarm,
            warning=" ".join(dict.fromkeys(warnings)) or None,
            overflow=self._overflow(faces),
            debug=debug if self._debug_on else None,
        )
        return self._last

    def snapshot(self, at: datetime) -> str | None:
        """Save the last frame as the screen showed it. Returns a warning, or None."""
        if self._frame is None:
            return NO_FRAME_WARNING
        path, warning = self._save_snap(self._frame, self._shown, at, MANUAL_LETTER, MANUAL_REASON)
        event = (
            LogEvent(at, "System", "snap failed")
            if warning
            else LogEvent(at, "System", "manual snapshot", snap=path)
        )
        write_warnings = self._record([event])
        return warning or (write_warnings[0] if write_warnings else None)

    def debug_samples(self) -> list[dict]:
        return self._debug_buffer.samples()

    def close(self, at: datetime) -> None:
        if self._alarm:
            self._alarm = False
            self._record([LogEvent(at, "System", "alarm stopped")])
        self._files.close(at)

    def _find(self, frame: np.ndarray, warnings: list[str]) -> list[RawFace]:
        try:
            return self._finder.find(frame)
        except FarDetectFailed as failure:
            logger.exception("Far face detection failed")
            warnings.append(FAR_WARNING)
            return failure.near
        except Exception:
            # Carry on with no faces, so people leave after their grace and the alarm cannot stick.
            logger.exception("Face detection failed")
            warnings.append(DETECT_WARNING)
            return []

    def _without_unconfirmed(
        self, seen: dict[int, RawFace], active_ids: set[int], now_ms: float
    ) -> dict[int, RawFace]:
        """A far face counts as not seen until its track has lasted, which keeps false faces
        off the screen, and a near person lost for a moment hidden rather than called far.
        A near frame starts the count again, so a face on the edge of the two stays near.
        """
        for track_id, run in list(self._far_runs.items()):
            if track_id not in active_ids or (track_id not in seen and not run.confirmed):
                del self._far_runs[track_id]
        for track_id, face in seen.items():
            if face.far:
                run = self._far_runs.setdefault(track_id, _FarRun(now_ms))
                run.views += 1
                run.confirmed = run.confirmed or self._far_is_proven(track_id, run, now_ms)
            else:
                self._far_runs.pop(track_id, None)
        return {
            track_id: face
            for track_id, face in seen.items()
            if not face.far or self._far_runs[track_id].confirmed
        }

    def _far_is_proven(self, track_id: int, run: _FarRun, now_ms: float) -> bool:
        """A sleeper's tipped head can lose MediaPipe and shrink under YuNet, which looks just
        like walking away. So a red person gets as long as a lost person would have.
        """
        if run.views < CONFIG.far_confirm_frames:
            return False
        state = self._people.get(self._identity.state_key(track_id))
        red = state is not None and state.reading is not None and is_red(state.reading.label)
        return not red or now_ms - run.since_ms >= CONFIG.letter_release_ms

    def _overflow(self, faces: list[RawFace]) -> bool:
        """Too many near faces say so at once. Far ones could be false, so they must last."""
        crowded = len(faces) > CONFIG.max_faces
        self._crowded_frames = self._crowded_frames + 1 if crowded else 0
        near = sum(not face.far for face in faces)
        return near > CONFIG.max_faces or self._crowded_frames >= CONFIG.far_confirm_frames

    def _heavy_check(
        self, frame: np.ndarray, tracked: dict[int, RawFace], now_ms: float, warnings: list[str]
    ) -> None:
        """At most one signature or emotion call, which keeps the frame cost bounded."""
        wanting = [tid for tid in tracked if self._wants_signature(tid, now_ms)]
        if wanting:
            # A face with no letter wants one on every frame, so take turns.
            track_id = min(wanting, key=lambda t: self._signature_tried_ms.get(t, -math.inf))
            self._signature_tried_ms[track_id] = now_ms
            self._sign(frame, track_id, tracked[track_id], now_ms, warnings)
            return

        due = []
        for track_id, face in tracked.items():
            key = self._identity.state_key(track_id)
            if key is None or face.far:
                continue
            state = self._people.setdefault(key, _PersonState())
            if now_ms - state.emotion_tried_ms >= CONFIG.emotion_interval_ms:
                due.append((state, face))
        if due:
            state, face = min(due, key=lambda pair: pair[0].emotion_tried_ms)
            self._read_emotion(frame, state, face, now_ms, warnings)

    def _wants_signature(self, track_id: int, now_ms: float) -> bool:
        """A face that just failed waits its turn, or it would block everyone behind it."""
        failed_ms = self._signature_failed_ms.get(track_id, -math.inf)
        return (
            self._identity.needs_signature(track_id, now_ms)
            and now_ms - failed_ms >= CONFIG.identity_interval_ms
        )

    def _sign(
        self,
        frame: np.ndarray,
        track_id: int,
        face: RawFace,
        now_ms: float,
        warnings: list[str],
    ) -> None:
        try:
            signature = self._signer.signature(frame, face)
        except Exception:
            logger.exception("Face signature failed")
            self._signature_failed_ms[track_id] = now_ms
            warnings.append(FACE_WARNING)
            return
        baseline = self._baseline_of(track_id)
        decision = self._identity.add_signature(
            track_id, signature, baseline.is_frontal(face), now_ms, baseline.is_checkable(face)
        )
        if self._debug_on:
            self._debug_buffer.add(
                {"t_ms": now_ms, "kind": "identity", "track": track_id, **asdict(decision)}
            )
            key = self._identity.state_key(track_id)
            if key is not None:
                self._decisions[key] = decision

    def _baseline_of(self, track_id: int) -> Baseline:
        """Views are judged against the person's own normal once the track has a letter."""
        state = self._people.get(self._identity.state_key(track_id))
        return state.baseline if state else Baseline()

    def _read_emotion(
        self,
        frame: np.ndarray,
        state: _PersonState,
        face: RawFace,
        now_ms: float,
        warnings: list[str],
    ) -> None:
        state.emotion_tried_ms = now_ms  # set first, so a failure does not stay most overdue
        try:
            state.emotion = self._emotion.scores(frame, face.box)
        except Exception:
            logger.exception("Emotion scoring failed")
            warnings.append(FACE_WARNING)

    def _read_people(
        self,
        tracked: dict[int, RawFace],
        active_ids: set[int],
        now_ms: float,
        delay_sec: float,
        debug: list[str],
    ) -> dict[str, Person]:
        """Everyone with a letter and an active track, by state key, in panel order."""
        present: dict[str, Person] = {}
        for track_id in sorted(active_ids):
            key = self._identity.state_key(track_id)
            if key is None:
                continue
            state = self._people.setdefault(key, _PersonState())
            face = tracked.get(track_id)
            if face is not None and face.far:
                # Baseline and classifier hear nothing, so to them this is a gap like any other.
                state.reading = FAR_READING
                state.box = face.box
            elif face is not None:
                signals = state.baseline.to_signals(face, now_ms)
                state.reading = state.classifier.update(signals, now_ms, delay_sec, state.emotion)
                state.box = face.box
                if self._debug_on:
                    debug += self._debug_sample(key, face, signals, state, now_ms)
            if state.reading is not None and state.box is not None:
                present[key] = Person(
                    _shown_letter(key), state.box, state.reading, face is not None
                )
        order = sorted(present, key=lambda key: (key.startswith(UNKNOWN), key))
        return {key: present[key] for key in order}

    def _debug_sample(
        self, key: str, face: RawFace, signals: Signals, state: _PersonState, now_ms: float
    ) -> list[str]:
        normal = state.baseline.normal()
        sample = {
            "t_ms": now_ms,
            "letter": key,
            "signals": asdict(signals),
            "label": state.reading.label,
            "confidence": state.reading.confidence,
            "raw_eye_blink": raw_eye_closure(face),
            "baseline_closure": normal.closure if normal else None,
            "baseline_pitch": normal.pitch if normal else None,
            "yaw_deg": face.yaw_deg,
            "pitch_deg": face.pitch_deg,
            "emotion": state.emotion,
            **state.classifier.debug_info(),
        }
        self._debug_buffer.add(sample)
        ranges = self._debug_buffer.recent_ranges(key, now_ms)
        return _debug_lines(key, signals, ranges, sample, self._decisions.get(key))

    def _events(
        self,
        frame: np.ndarray,
        present: dict[str, Person],
        people: list[Person],
        now_ms: float,
        at: datetime,
        warnings: list[str],
    ) -> list[LogEvent]:
        readings = {key: p.reading for key, p in present.items()}
        events = [_shown_event(e) for e in diff_events(self._labels, readings, at)]
        for key in self._labels:
            if key not in present:
                self._forget_emotion(key)
                events += self._owed_snap(key, at, warnings)
        self._labels = {key: r.label for key, r in readings.items()}

        alarm = should_sound(self._labels.values())
        if alarm != self._alarm:
            events.append(LogEvent(at, "System", "alarm started" if alarm else "alarm stopped"))
        self._alarm = alarm

        for key, person in present.items():
            if person.reading.label == "far":
                events += self._owed_snap(key, at, warnings)  # as lost to the camera as leaving
                continue
            red = is_red(person.reading.label)
            if self._snaps.update(key, red, now_ms, person.visible):
                events.append(self._snap_event(frame, people, person, at, at, warnings))
            if not red:
                self._red_proof.pop(key, None)
            elif person.visible:
                self._red_proof[key] = _RedProof(frame, people, person, at)
        return events

    def _owed_snap(self, key: str, at: datetime, warnings: list[str]) -> list[LogEvent]:
        """End a red run with nobody left to see, saving its last visible frame if it earned one."""
        owed = self._snaps.drop(key)
        proof = self._red_proof.pop(key, None)
        if not owed or proof is None:
            return []
        return [self._snap_event(proof.frame, proof.people, proof.person, proof.at, at, warnings)]

    def _forget_emotion(self, key: str) -> None:
        """Once released, the last scores are stale: they must not label a return."""
        state = self._people.get(key)
        if state is not None:
            state.emotion = None
            state.emotion_tried_ms = -math.inf

    def _snap_event(
        self,
        frame: np.ndarray,
        people: list[Person],
        person: Person,
        shown_at: datetime,
        at: datetime,
        warnings: list[str],
    ) -> LogEvent:
        """Save a red person's snap, named for when the frame was seen. Logged as of at."""
        who = f"Person {person.letter}"
        label = person.reading.label
        path, warning = self._save_snap(frame, people, shown_at, person.letter, label)
        if warning:
            warnings.append(warning)
            return LogEvent(at, who, "snap failed")
        return LogEvent(at, who, label, person.reading.confidence, path)

    def _save_snap(
        self, frame: np.ndarray, people: list[Person], at: datetime, letter: str, reason: str
    ) -> tuple[str | None, str | None]:
        try:
            return self._files.save_snap(render_snap(frame, people), at, letter, reason)
        except Exception as e:
            logger.exception("Snap failed")
            return None, f"{SNAP_WARNING}: {e}"

    def _record(self, events: list[LogEvent]) -> list[str]:
        """Write events to the file and the panel log. Returns any warnings."""
        written = [self._files.write(event) for event in events]
        lines = [format_line(event) for event in reversed(events)]
        self._log = [*lines, *self._log][: CONFIG.log_cap]
        return [warning for warning in written if warning]
