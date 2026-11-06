import { CONFIG } from './config';
import type { Letter, RawFace, TrackedFace } from './types';

const LETTERS: readonly Letter[] = ['A', 'B', 'C', 'D'];

interface Track {
  letter: Letter;
  cx: number;
  cy: number;
  lastSeenMs: number;
}

export class Tracker {
  private tracks: Track[] = [];

  update(faces: RawFace[], nowMs: number): TrackedFace[] {
    this.tracks = this.tracks.filter((t) => nowMs - t.lastSeenMs <= CONFIG.letterReleaseMs);

    const centres = faces.map((f) => ({
      cx: f.box.x + f.box.width / 2,
      cy: f.box.y + f.box.height / 2,
    }));

    const pairs: { fi: number; track: Track; dist: number }[] = [];
    centres.forEach((c, fi) => {
      for (const track of this.tracks) {
        const dist = Math.hypot(c.cx - track.cx, c.cy - track.cy);
        if (dist <= CONFIG.maxMatchDistance) pairs.push({ fi, track, dist });
      }
    });
    pairs.sort((a, b) => a.dist - b.dist);

    const assigned = new Map<number, Track>();
    const usedTracks = new Set<Track>();
    for (const p of pairs) {
      if (assigned.has(p.fi) || usedTracks.has(p.track)) continue;
      assigned.set(p.fi, p.track);
      usedTracks.add(p.track);
    }

    const result: TrackedFace[] = [];
    faces.forEach((face, fi) => {
      const c = centres[fi];
      if (!c) return;
      let track = assigned.get(fi);
      if (!track) {
        const letter = LETTERS.slice(0, CONFIG.maxFaces).find(
          (l) => !this.tracks.some((t) => t.letter === l),
        );
        if (!letter) return; // no free letter: ignore this face
        track = { letter, cx: c.cx, cy: c.cy, lastSeenMs: nowMs };
        this.tracks.push(track);
      }
      track.cx = c.cx;
      track.cy = c.cy;
      track.lastSeenMs = nowMs;
      result.push({ letter: track.letter, face });
    });
    return result;
  }

  activeLetters(): Letter[] {
    return this.tracks.map((t) => t.letter).sort();
  }
}
