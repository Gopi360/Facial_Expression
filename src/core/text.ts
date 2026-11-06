import type { Label, Person } from './types';

const LINE_WORDS: Record<Label, string> = {
  sleeping: 'is sleeping',
  drowsy: 'is about to sleep',
  angry: 'is angry',
  laughing: 'is laughing',
  smiling: 'is smiling',
  crying: 'is crying',
  neutral: 'looks neutral',
};

export function faceCountText(n: number): string {
  if (n <= 0) return 'No faces found';
  return n === 1 ? '1 face found' : `${n} faces found`;
}

export function personLine(p: Person): string {
  return `Person ${p.letter} ${LINE_WORDS[p.reading.label]} ${Math.round(p.reading.confidence)}%`;
}
