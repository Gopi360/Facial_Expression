"""Session face recognition: who is A, B, C. Signatures live in memory only."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from app.core.config import CONFIG

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNKNOWN = "?"


@dataclass
class _TrackState:
    first_seen_ms: float
    letter: str | None = None
    last_signature_ms: float | None = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else 0.0


class Identity:
    def __init__(self) -> None:
        self._tracks: dict[int, _TrackState] = {}
        self._signatures: dict[str, deque[np.ndarray]] = {}
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
        for t in self._tracks.values():
            if t.letter is None and now_ms - t.first_seen_ms >= CONFIG.identity_pending_ms:
                t.letter = self._take_next_letter()

    def add_signature(
        self, track_id: int, signature: np.ndarray, frontal: bool, now_ms: float
    ) -> None:
        track = self._tracks.setdefault(track_id, _TrackState(first_seen_ms=now_ms))
        track.last_signature_ms = now_ms
        sig = np.asarray(signature, dtype=float).copy()

        if track.letter is None:
            match = self._best_match(sig, exclude=self._visible_letters(except_id=track_id))
            if match is not None:
                letter = match[0]
                for other in self._tracks.values():
                    if other is not track and other.letter == letter:
                        other.letter = None
                track.letter = letter
            else:
                track.letter = self._take_next_letter()
        elif track.letter != UNKNOWN:
            other_match = self._best_match(
                sig, exclude=self._visible_letters(except_id=track_id) | {track.letter}
            )
            if other_match is not None and other_match[1] > self._score(sig, track.letter):
                return  # clearly someone else: do not flip, do not learn

        if frontal and track.letter is not None and track.letter != UNKNOWN:
            store = self._signatures.setdefault(
                track.letter, deque(maxlen=CONFIG.identity_max_signatures)
            )
            store.append(sig)

    # -- helpers ---------------------------------------------------------
    def _visible_letters(self, except_id: int) -> set[str]:
        return {
            t.letter
            for tid, t in self._tracks.items()
            if tid in self._visible and tid != except_id and t.letter is not None
        }

    def _score(self, sig: np.ndarray, letter: str) -> float:
        stored = self._signatures.get(letter)
        if not stored:
            return -1.0
        return max(_cosine(sig, s) for s in stored)

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
