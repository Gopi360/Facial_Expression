"""Session face recognition: who is A, B, C. Signatures live in memory only."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.core.config import CONFIG

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNKNOWN = "?"


@dataclass
class _TrackState:
    first_seen_ms: float
    letter: str | None = None
    last_signature_ms: float | None = None
    mismatches: int = 0  # frontal checks in a row that failed this letter's anchors


def _rolling() -> deque[np.ndarray]:
    return deque(maxlen=CONFIG.identity_max_signatures)


@dataclass
class _Store:
    """One person's views. Anchors are the first good views and never roll off, so whatever
    the rolling slots fill with, there is always a fixed idea of what this person looks like.
    """

    anchors: list[np.ndarray] = field(default_factory=list)
    rolling: deque[np.ndarray] = field(default_factory=_rolling)

    def __len__(self) -> int:
        return len(self.anchors) + len(self.rolling)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else 0.0


def _best_cosine(sig: np.ndarray, stored) -> float:
    return max((_cosine(sig, s) for s in stored), default=-1.0)


class Identity:
    def __init__(self) -> None:
        self._tracks: dict[int, _TrackState] = {}
        self._signatures: dict[str, _Store] = {}
        self._issued: set[str] = set()
        self._visible: set[int] = set()
        self._unknown_count = 0

    # -- queries ---------------------------------------------------------
    def letter_for(self, track_id: int) -> str | None:
        t = self._tracks.get(track_id)
        return t.letter if t else None

    def state_key(self, track_id: int) -> str | None:
        """Key for per-person state. Unique per track once past Z."""
        letter = self.letter_for(track_id)
        if letter is None:
            return None
        return f"?{track_id}" if letter == UNKNOWN else letter

    def needs_signature(self, track_id: int, now_ms: float) -> bool:
        t = self._tracks.get(track_id)
        if t is None or t.letter is None or t.last_signature_ms is None:
            return True
        return now_ms - t.last_signature_ms >= CONFIG.identity_interval_ms

    def people_count(self) -> int:
        return len(self._issued) + self._unknown_count

    # -- updates ---------------------------------------------------------
    def sync(self, visible_ids: set[int], active_ids: set[int], now_ms: float) -> None:
        self._visible = set(visible_ids)
        for tid in active_ids:
            self._tracks.setdefault(tid, _TrackState(first_seen_ms=now_ms))
        for tid in [t for t in self._tracks if t not in active_ids]:
            del self._tracks[tid]
        for tid, t in self._tracks.items():
            if t.letter is not None:
                continue
            if tid not in self._visible:
                t.first_seen_ms = now_ms  # the pending clock only runs while visible
            elif now_ms - t.first_seen_ms >= CONFIG.identity_pending_ms:
                t.letter = self._take_next_letter()

    def add_signature(
        self, track_id: int, signature: np.ndarray, frontal: bool, now_ms: float
    ) -> None:
        track = self._tracks.setdefault(track_id, _TrackState(first_seen_ms=now_ms))
        track.last_signature_ms = now_ms
        sig = np.asarray(signature, dtype=float).copy()

        if track.letter is None:
            self._place(track_id, sig, frontal, now_ms)
        elif track.letter != UNKNOWN:
            self._check(track_id, sig, frontal, now_ms)

    # -- helpers ---------------------------------------------------------
    def _place(
        self,
        track_id: int,
        sig: np.ndarray,
        frontal: bool,
        now_ms: float,
        rejected: str | None = None,
    ) -> None:
        """Give a letterless track a letter if this view can decide one. Any view can match
        a known person, but only a frontal view can say this is someone new.
        """
        track = self._tracks[track_id]
        exclude = self._held_letters(except_id=track_id)
        if rejected is not None:
            exclude.add(rejected)
        match = self._best_match(sig, exclude)
        if match is not None:
            self._hand_over(track, match[0], now_ms)
        elif frontal:
            track.letter = self._take_next_letter()
        if frontal and track.letter is not None:
            self._learn(track.letter, sig)

    def _check(self, track_id: int, sig: np.ndarray, frontal: bool, now_ms: float) -> None:
        """Does a lettered track still look like its letter? Only frontal views can say no."""
        track = self._tracks[track_id]
        letter = track.letter
        store = self._signatures.get(letter)
        other = self._best_match(sig, self._visible_letters(except_id=track_id) | {letter})

        if frontal and store is None and other is not None:
            # The letter came from the pending timeout and has never seen a frontal view.
            self._hand_over(track, other[0], now_ms)
            self._learn(other[0], sig)
            return
        if frontal and store is not None:
            if _best_cosine(sig, store.anchors) >= CONFIG.identity_match_at:
                track.mismatches = 0
            else:
                track.mismatches += 1
                if track.mismatches >= CONFIG.identity_mismatch_checks:
                    self._strip(track, now_ms)
                    self._place(track_id, sig, frontal, now_ms, rejected=letter)
                return
        if other is not None and other[1] > self._score(sig, letter):
            return  # clearly someone else: do not flip, do not learn
        if frontal:
            self._learn(letter, sig)

    def _learn(self, letter: str, sig: np.ndarray) -> None:
        if letter == UNKNOWN:
            return
        store = self._signatures.setdefault(letter, _Store())
        if store.anchors and _best_cosine(sig, store.anchors) < CONFIG.identity_match_at:
            return  # does not look like this letter's person: do not learn it
        if len(store.anchors) < CONFIG.identity_anchor_signatures:
            store.anchors.append(sig)
        else:
            store.rolling.append(sig)

    def _hand_over(self, track: _TrackState, letter: str, now_ms: float) -> None:
        for other in self._tracks.values():
            if other is not track and other.letter == letter:
                self._strip(other, now_ms)
        track.letter = letter
        track.mismatches = 0

    @staticmethod
    def _strip(track: _TrackState, now_ms: float) -> None:
        track.letter = None
        track.first_seen_ms = now_ms
        track.mismatches = 0

    def _visible_tracks(self, except_id: int) -> list[_TrackState]:
        return [
            t
            for tid, t in self._tracks.items()
            if tid in self._visible and tid != except_id and t.letter is not None
        ]

    def _visible_letters(self, except_id: int) -> set[str]:
        return {t.letter for t in self._visible_tracks(except_id)}

    def _held_letters(self, except_id: int) -> set[str]:
        """Letters another face in view is sure of. One failing its own letter can lose it."""
        return {t.letter for t in self._visible_tracks(except_id) if t.mismatches == 0}

    def _score(self, sig: np.ndarray, letter: str) -> float:
        store = self._signatures.get(letter)
        if not store:
            return -1.0
        return max(_best_cosine(sig, store.anchors), _best_cosine(sig, store.rolling))

    def _best_match(self, sig: np.ndarray, exclude: set[str]) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        for letter in self._signatures:
            if letter in exclude:
                continue
            score = self._score(sig, letter)
            if score >= CONFIG.identity_match_at and (best is None or score > best[1]):
                best = (letter, score)
        return best

    def _take_next_letter(self) -> str:
        for letter in LETTERS:
            if letter not in self._issued:
                self._issued.add(letter)
                return letter
        self._unknown_count += 1
        return UNKNOWN
