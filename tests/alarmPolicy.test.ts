import { describe, it, expect } from 'vitest';
import { shouldSound } from '../src/core/alarmPolicy';

describe('shouldSound', () => {
  it('is true for sleeping or drowsy, false otherwise, and false when muted', () => {
    expect(shouldSound(['neutral', 'sleeping'], false)).toBe(true);
    expect(shouldSound(['drowsy'], false)).toBe(true);
    expect(shouldSound(['angry', 'smiling', 'crying', 'laughing', 'neutral'], false)).toBe(false);
    expect(shouldSound(['sleeping'], true)).toBe(false);
  });
  it('is false for an empty list', () => {
    expect(shouldSound([], false)).toBe(false);
  });
});
