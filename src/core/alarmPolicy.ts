import type { Label } from './types';

export function shouldSound(labels: Label[], muted: boolean): boolean {
  if (muted) return false;
  return labels.some((l) => l === 'sleeping' || l === 'drowsy');
}
