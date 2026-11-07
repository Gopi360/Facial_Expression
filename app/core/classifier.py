"""Turns one person's signals over time into a steady label and confidence."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import CONFIG
from app.core.expressions import expression_reading, is_model_laugh
from app.core.signals import percent
from app.core.types import DrowsySign, Reading, Signals

MS_PER_SEC = 1000


def _closed_threshold(s: Signals) -> float:
    return (
        CONFIG.eyes_closed_when_looking_down_at
        if s.look_down >= CONFIG.look_down_at
        else CONFIG.eyes_closed_at
    )


def _is_eyes_closed(s: Signals) -> bool:
    return s.eye_closure >= _closed_threshold(s)


def _is_angry(s: Signals) -> bool:
    return s.brow_down >= CONFIG.angry_brow_at and s.smile < CONFIG.angry_smile_max


def _is_crying(s: Signals) -> bool:
    return s.brow_inner_up >= CONFIG.cry_brow_at and s.frown >= CONFIG.cry_frown_at


def _is_strong_expression(s: Signals, emotion: dict[str, float] | None) -> bool:
    """What protects the sleep path: the blendshape rules, or a laugh the model is showing."""
    blendshapes = s.smile >= CONFIG.strong_smile_at or _is_angry(s) or _is_crying(s)
    return blendshapes or is_model_laugh(emotion, s)


def _is_yawn(s: Signals) -> bool:
    return s.jaw_open >= CONFIG.yawn_jaw_at and s.smile < CONFIG.smile_at


# Start times of everything that needs an unbroken stretch of watching.
_RUNNING_HOLDS = (
    "closed_since_ms",
    "closed_run_since_ms",
    "droopy_since_ms",
    "head_droop_since_ms",
    "yawn_since_ms",
)


@dataclass
class _State:
    last_update_ms: float | None = None
    closed_since_ms: float | None = None  # start of the closed count
    last_closed_ms: float = 0  # last frame with eyes closed, while the count runs
    last_closed_strength: float = 0  # eye_closure on that frame
    closed_run_since_ms: float | None = None  # start of the current unbroken closed run
    longest_closed_run_ms: float = 0  # longest unbroken run within the closed count
    open_since_ms: float | None = None  # first open frame while the count still runs
    slow_blink_ends_ms: list[float] = field(default_factory=list)
    last_strong_ms: float | None = None
    strong_run_since_ms: float | None = None  # start of the current strong run
    droopy_since_ms: float | None = None
    head_droop_since_ms: float | None = None
    yawn_since_ms: float | None = None
    drowsy_sign_ms: float | None = None  # last time any drowsy sign was true
    drowsy_sign: DrowsySign | None = None  # which sign that was
    shown: Reading | None = None
    differs_since_ms: float | None = None  # when the candidate first stopped matching shown


class PersonClassifier:
    """Turns one person's signals over time into a steady label and confidence."""

    def __init__(self) -> None:
        self._state = _State()
        self._debug: dict = {
            "drowsy_sign": None,
            "strong_held": False,
            "strong_sustained": False,
            "closed_ms": 0,
            "closed_threshold": CONFIG.eyes_closed_at,
        }

    def debug_info(self) -> dict:
        """What drove the last update, for the debug view. Has no effect on the Reading."""
        return self._debug

    def update(
        self, s: Signals, now_ms: float, delay_sec: float, emotion: dict[str, float] | None
    ) -> Reading:
        delay_ms = delay_sec * MS_PER_SEC
        last_update_ms = self._state.last_update_ms
        gap_ms = now_ms - last_update_ms if last_update_ms is not None else 0
        if gap_ms > CONFIG.frame_gap_reset_ms:
            self._state = _State()
        elif gap_ms > CONFIG.open_grace_ms:
            self._skip_gap(s, gap_ms, now_ms, delay_ms)
        self._state.last_update_ms = now_ms

        is_strong = _is_strong_expression(s, emotion)
        strong = self._held_strong(is_strong, now_ms)
        sustained = self._sustained_strong(is_strong, now_ms, strong)
        self._track_closure(s, now_ms, delay_ms)
        self._track_drowsy_signs(s, now_ms, sustained)
        self._track_yawn(s, now_ms)
        self._debug = {
            "drowsy_sign": self._state.drowsy_sign if self._drowsy_active(now_ms) else None,
            "strong_held": strong,
            "strong_sustained": sustained,
            "closed_ms": self._closed_ms(now_ms),
            "closed_threshold": _closed_threshold(s),
        }
        return self._display(s, self._candidate(s, now_ms, delay_ms, strong, emotion), now_ms)

    def _closed_ms(self, now_ms: float) -> float:
        closed_since_ms = self._state.closed_since_ms
        return now_ms - closed_since_ms if closed_since_ms is not None else 0

    def _skip_gap(self, s: Signals, gap_ms: float, now_ms: float, delay_ms: float) -> None:
        """Nobody saw this time, so no hold counts it. Whatever was counted before it stands,
        unless the eyes came back open, which ends the closure there and then.
        """
        st = self._state
        for name in _RUNNING_HOLDS:
            since_ms = getattr(st, name)
            if since_ms is not None:
                setattr(st, name, since_ms + gap_ms)
        st.last_closed_ms += gap_ms
        if st.closed_since_ms is not None and not _is_eyes_closed(s):
            self._end_closure(now_ms, delay_ms)

    def _held_strong(self, is_strong: bool, now_ms: float) -> bool:
        """A strong expression counts as present until it has been absent for label_hold_ms."""
        st = self._state
        if is_strong:
            st.last_strong_ms = now_ms
        return st.last_strong_ms is not None and now_ms - st.last_strong_ms < CONFIG.label_hold_ms

    def _sustained_strong(self, is_strong: bool, now_ms: float, held: bool) -> bool:
        """True once a strong expression has run for label_hold_ms, and until it is no longer
        held. A lone noisy frame never gets there, and a dip under strong_dip_grace_ms
        does not break the run, so a real laugh with a noisy smile still does.
        """
        st = self._state

        def reached() -> bool:
            return (
                st.strong_run_since_ms is not None
                and now_ms - st.strong_run_since_ms >= CONFIG.label_hold_ms
            )

        dip_over = (
            st.last_strong_ms is None or now_ms - st.last_strong_ms > CONFIG.strong_dip_grace_ms
        )
        if is_strong:
            if st.strong_run_since_ms is None:
                st.strong_run_since_ms = now_ms
        elif not held or (dip_over and not reached()):
            st.strong_run_since_ms = None
        return reached()

    def _track_closure(self, s: Signals, now_ms: float, delay_ms: float) -> None:
        st = self._state
        if _is_eyes_closed(s):
            if st.closed_since_ms is None:
                st.closed_since_ms = now_ms
            if st.closed_run_since_ms is None:
                st.closed_run_since_ms = now_ms
            st.last_closed_ms = now_ms
            st.last_closed_strength = s.eye_closure
            st.longest_closed_run_ms = max(
                st.longest_closed_run_ms, now_ms - st.closed_run_since_ms
            )
            st.open_since_ms = None
            return
        st.closed_run_since_ms = None
        if st.closed_since_ms is None:
            return

        if st.open_since_ms is None:
            st.open_since_ms = now_ms
        if now_ms - st.open_since_ms < CONFIG.open_grace_ms:
            return

        self._end_closure(now_ms, delay_ms)

    def _end_closure(self, now_ms: float, delay_ms: float) -> None:
        # The grace bridges the sleep count, but two quick blinks it joins are not
        # one slow blink, so a slow blink needs one unbroken run of that length.
        st = self._state
        counted_ms = st.last_closed_ms - st.closed_since_ms
        if st.longest_closed_run_ms >= CONFIG.slow_blink_ms and counted_ms < delay_ms:
            st.slow_blink_ends_ms.append(now_ms)
        st.closed_since_ms = None
        st.open_since_ms = None
        st.longest_closed_run_ms = 0

    def _track_drowsy_signs(self, s: Signals, now_ms: float, sustained_strong: bool) -> None:
        st = self._state
        if sustained_strong:
            # Clear everything, not just the label, so that eyes squeezed shut in a
            # laugh cannot set off drowsy the moment the laugh ends.
            st.slow_blink_ends_ms = []
            st.droopy_since_ms = None
            st.head_droop_since_ms = None
            st.drowsy_sign_ms = None
            st.drowsy_sign = None
            return

        st.slow_blink_ends_ms = [
            end_ms
            for end_ms in st.slow_blink_ends_ms
            if now_ms - end_ms <= CONFIG.slow_blink_window_ms
        ]
        closed = _is_eyes_closed(s)
        not_looking_down = s.look_down < CONFIG.look_down_at
        droopy = not_looking_down and s.eye_closure >= CONFIG.droopy_eyes_at
        head_droop = (
            not_looking_down
            and s.pitch_delta >= CONFIG.head_droop_deg
            and s.eye_closure >= CONFIG.head_droop_eyes_at
        )
        # A blink passes through the droopy hold, but a longer closure ends it.
        long_closure = (
            st.closed_run_since_ms is not None
            and now_ms - st.closed_run_since_ms >= CONFIG.slow_blink_ms
        )
        if not droopy or long_closure:
            st.droopy_since_ms = None
        elif not closed and st.droopy_since_ms is None:
            st.droopy_since_ms = now_ms
        if not head_droop:
            st.head_droop_since_ms = None
        elif st.head_droop_since_ms is None:
            st.head_droop_since_ms = now_ms

        # Fully closed eyes are not the droopy sign, so the delay setting governs them.
        # Only someone already drowsy stays drowsy through them, until sleeping takes over.
        droopy_held = (
            not closed
            and st.droopy_since_ms is not None
            and now_ms - st.droopy_since_ms >= CONFIG.droopy_hold_ms
        )
        slow_blinks = len(st.slow_blink_ends_ms) >= CONFIG.slow_blinks_for_drowsy
        head_droop_held = (
            st.head_droop_since_ms is not None
            and now_ms - st.head_droop_since_ms >= CONFIG.head_droop_hold_ms
        )
        stayed_closed = closed and st.shown is not None and st.shown.label == "drowsy"
        signs: list[tuple[bool, DrowsySign]] = [
            (slow_blinks, "slowBlinks"),
            (droopy_held, "droopy"),
            (head_droop_held, "headDroop"),
            (stayed_closed, "stayedClosed"),
        ]
        sign = next((name for present, name in signs if present), None)
        # A strong run neither ends drowsy nor starts it until sustained, so a laugh that
        # begins just as a hold runs out does not sound the alarm in its first moments.
        if sign is None or st.strong_run_since_ms is not None:
            return
        st.drowsy_sign_ms = now_ms
        st.drowsy_sign = sign
        if slow_blinks:
            st.slow_blink_ends_ms = []  # one event, so drowsy ends after drowsy_clear_ms

    def _track_yawn(self, s: Signals, now_ms: float) -> None:
        """Runs on every frame, so the hold keeps counting while a sleep label is the candidate."""
        st = self._state
        if not _is_yawn(s):
            st.yawn_since_ms = None
        elif st.yawn_since_ms is None:
            st.yawn_since_ms = now_ms

    def _yawn_held(self, now_ms: float) -> bool:
        yawn_since_ms = self._state.yawn_since_ms
        return yawn_since_ms is not None and now_ms - yawn_since_ms >= CONFIG.yawn_hold_ms

    def _drowsy_active(self, now_ms: float) -> bool:
        drowsy_sign_ms = self._state.drowsy_sign_ms
        return drowsy_sign_ms is not None and now_ms - drowsy_sign_ms < CONFIG.drowsy_clear_ms

    def _candidate(
        self,
        s: Signals,
        now_ms: float,
        delay_ms: float,
        strong: bool,
        emotion: dict[str, float] | None,
    ) -> Reading:
        st = self._state
        needed_ms = max(delay_ms, CONFIG.expression_limit_ms) if strong else delay_ms
        if st.closed_since_ms is not None and now_ms - st.closed_since_ms >= needed_ms:
            # Through the open grace this stays the strength of the last closed frame.
            return Reading("sleeping", percent(st.last_closed_strength))
        if self._drowsy_active(now_ms):
            return Reading("drowsy", percent(max(s.eye_closure, CONFIG.drowsy_min_confidence)))
        if self._yawn_held(now_ms):
            return Reading("yawning", percent(s.jaw_open))
        return expression_reading(emotion, s)

    def _display(self, s: Signals, candidate: Reading, now_ms: float) -> Reading:
        """A held label keeps the confidence it was last shown with."""
        st = self._state
        if st.shown is None:
            st.shown = expression_reading(None, s)

        if candidate.label == st.shown.label:
            st.differs_since_ms = None
            st.shown = candidate
            return candidate
        # Timed against the shown label, not one candidate, so that two other labels
        # alternating faster than the hold cannot pin an old label, or the alarm, on.
        if st.differs_since_ms is None:
            st.differs_since_ms = now_ms
        immediate = candidate.label in ("sleeping", "drowsy")
        if immediate or now_ms - st.differs_since_ms >= CONFIG.label_hold_ms:
            st.differs_since_ms = None
            st.shown = candidate
        return st.shown
