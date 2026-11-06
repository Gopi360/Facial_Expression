import { describe, it, expect } from 'vitest';
import { diffEvents, appendLog, formatEntry } from '../src/core/activityLog';
import type { Label, Letter, LogEntry } from '../src/core/types';

const at = new Date(2026, 10, 6, 12, 4, 31);
const m = (...e: [Letter, Label][]) => new Map<Letter, Label>(e);

describe('diffEvents', () => {
  it('no change gives no log entries', () => {
    expect(diffEvents(m(['A', 'neutral']), m(['A', 'neutral']), at)).toEqual([]);
  });
  it('a label change gives one entry', () => {
    const r = diffEvents(m(['B', 'neutral']), m(['B', 'sleeping']), at);
    expect(r).toEqual([{ at, text: 'B fell asleep' }]);
  });
  it('a new letter gives joined, then a missing letter gives left', () => {
    expect(diffEvents(m(), m(['C', 'sleeping']), at)).toEqual([{ at, text: 'C joined' }]);
    expect(diffEvents(m(['C', 'sleeping']), m(), at)).toEqual([{ at, text: 'C left' }]);
  });
});

describe('appendLog', () => {
  it('is newest first and capped at 100', () => {
    const mk = (i: number): LogEntry => ({ at, text: `e${i}` });
    let log: LogEntry[] = [];
    for (let i = 0; i < 120; i++) log = appendLog(log, [mk(i)]);
    expect(log).toHaveLength(100);
    expect(log[0]?.text).toBe('e119');
    const two = appendLog([mk(0)], [mk(1), mk(2)]);
    expect(two.map((e) => e.text)).toEqual(['e1', 'e2', 'e0']);
  });
});

describe('formatEntry', () => {
  it('uses 24 hour HH:MM:SS', () => {
    expect(formatEntry({ at, text: 'B fell asleep' })).toBe('12:04:31 B fell asleep');
    expect(formatEntry({ at: new Date(2026, 10, 6, 15, 5, 9), text: 'x' })).toBe('15:05:09 x');
  });
});
