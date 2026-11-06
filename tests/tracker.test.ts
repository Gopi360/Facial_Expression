import { describe, it, expect } from 'vitest';
import { Tracker } from '../src/core/tracker';
import type { RawFace } from '../src/core/types';

function face(x: number, y: number): RawFace {
  return { box: { x, y, width: 0.1, height: 0.1 }, scores: {}, pitchDeg: 0 };
}

describe('Tracker', () => {
  it('first face gets A, second gets B', () => {
    const t = new Tracker();
    const r = t.update([face(0.1, 0.1), face(0.7, 0.7)], 0);
    expect(r.map((f) => f.letter)).toEqual(['A', 'B']);
  });

  it('a face that moves slightly keeps its letter', () => {
    const t = new Tracker();
    t.update([face(0.2, 0.2)], 0);
    const r = t.update([face(0.22, 0.2)], 100);
    expect(r[0]?.letter).toBe('A');
  });

  it('two faces keep letters when input order swaps', () => {
    const t = new Tracker();
    t.update([face(0.1, 0.1), face(0.7, 0.7)], 0);
    const r = t.update([face(0.7, 0.7), face(0.1, 0.1)], 100);
    expect(r.map((f) => f.letter)).toEqual(['B', 'A']);
  });

  it('face gone under 2 seconds keeps its letter on return', () => {
    const t = new Tracker();
    t.update([face(0.2, 0.2)], 0);
    t.update([], 1000);
    expect(t.activeLetters()).toContain('A');
    const r = t.update([face(0.2, 0.2)], 1900);
    expect(r[0]?.letter).toBe('A');
  });

  it('face gone over 2 seconds releases its letter', () => {
    const t = new Tracker();
    t.update([face(0.2, 0.2)], 0);
    t.update([], 2100);
    expect(t.activeLetters()).toEqual([]);
  });

  it('new face takes the lowest free letter', () => {
    const t = new Tracker();
    t.update([face(0.1, 0.1), face(0.7, 0.7)], 0);
    t.update([face(0.7, 0.7)], 1000);
    t.update([face(0.7, 0.7)], 2100);
    expect(t.activeLetters()).toEqual(['B']);
    const r = t.update([face(0.7, 0.7), face(0.1, 0.5)], 2200);
    expect(r.map((f) => f.letter)).toEqual(['B', 'A']);
  });

  it('a fifth face is ignored and steals no letter', () => {
    const t = new Tracker();
    const r = t.update(
      [face(0.0, 0.0), face(0.5, 0.0), face(0.0, 0.5), face(0.5, 0.5), face(0.9, 0.9)],
      0,
    );
    expect(r.map((f) => f.letter)).toEqual(['A', 'B', 'C', 'D']);
  });

  it('a face beyond maxMatchDistance is treated as new', () => {
    const t = new Tracker();
    t.update([face(0.1, 0.1)], 0);
    const r = t.update([face(0.9, 0.9)], 100);
    expect(r[0]?.letter).toBe('B');
  });
});
