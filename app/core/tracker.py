"""Follow faces between frames by position. Tracks have integer ids that are never reused."""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.config import CONFIG
from app.core.types import RawFace


@dataclass
class _Track:
    track_id: int
    cx: float
    cy: float
    last_seen_ms: float


class Tracker:
    def __init__(self) -> None:
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, faces: list[RawFace], now_ms: float) -> list[tuple[int, RawFace]]:
        self._tracks = [
            t for t in self._tracks if now_ms - t.last_seen_ms <= CONFIG.letter_release_ms
        ]

        if len(faces) > CONFIG.max_faces:
            largest = set(
                sorted(
                    range(len(faces)),
                    key=lambda i: faces[i].box.width * faces[i].box.height,
                    reverse=True,
                )[: CONFIG.max_faces]
            )
            faces = [f for i, f in enumerate(faces) if i in largest]

        centres = [(f.box.x + f.box.width / 2, f.box.y + f.box.height / 2) for f in faces]

        pairs: list[tuple[float, int, int]] = []
        for fi, (cx, cy) in enumerate(centres):
            for ti, track in enumerate(self._tracks):
                dist = math.hypot(cx - track.cx, cy - track.cy)
                if dist <= CONFIG.max_match_distance:
                    pairs.append((dist, fi, ti))
        pairs.sort()

        assigned: dict[int, int] = {}
        used: set[int] = set()
        for _, fi, ti in pairs:
            if fi in assigned or ti in used:
                continue
            assigned[fi] = ti
            used.add(ti)

        result: list[tuple[int, RawFace]] = []
        for fi, face in enumerate(faces):
            cx, cy = centres[fi]
            if fi in assigned:
                track = self._tracks[assigned[fi]]
            else:
                track = _Track(self._next_id, cx, cy, now_ms)
                self._next_id += 1
                self._tracks.append(track)
            track.cx, track.cy, track.last_seen_ms = cx, cy, now_ms
            result.append((track.track_id, face))
        return result

    def active_ids(self) -> set[int]:
        """Ids of tracks in view or still inside their grace period."""
        return {t.track_id for t in self._tracks}
