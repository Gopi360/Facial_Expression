import { describe, it, expect } from 'vitest';
import { faceCountText, personLine } from '../src/core/text';
import type { Label, Person } from '../src/core/types';

const person = (label: Label, confidence = 92): Person => ({
  letter: 'A',
  box: { x: 0, y: 0, width: 1, height: 1 },
  reading: { label, confidence },
});

describe('faceCountText', () => {
  it('words 0, 1 and 2', () => {
    expect(faceCountText(0)).toBe('No faces found');
    expect(faceCountText(1)).toBe('1 face found');
    expect(faceCountText(2)).toBe('2 faces found');
  });
});

describe('personLine', () => {
  it('words all seven labels with confidence', () => {
    expect(personLine(person('sleeping'))).toBe('Person A is sleeping 92%');
    expect(personLine(person('drowsy'))).toBe('Person A is about to sleep 92%');
    expect(personLine(person('angry'))).toBe('Person A is angry 92%');
    expect(personLine(person('laughing'))).toBe('Person A is laughing 92%');
    expect(personLine(person('smiling'))).toBe('Person A is smiling 92%');
    expect(personLine(person('crying'))).toBe('Person A is crying 92%');
    expect(personLine(person('neutral'))).toBe('Person A looks neutral 92%');
  });
});
