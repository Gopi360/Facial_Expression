import { CONFIG } from './config';
import type { Label, Letter, LogEntry } from './types';

const LOG_WORDS: Record<Label, string> = {
  sleeping: 'fell asleep',
  drowsy: 'looks drowsy',
  angry: 'looks angry',
  laughing: 'started laughing',
  smiling: 'started smiling',
  crying: 'looks like crying',
  neutral: 'looks neutral',
};

export function diffEvents(prev: Map<Letter, Label>, next: Map<Letter, Label>, at: Date): LogEntry[] {
  const out: LogEntry[] = [];
  for (const [letter, label] of next) {
    const before = prev.get(letter);
    if (before === undefined) out.push({ at, text: `${letter} joined` });
    else if (before !== label) out.push({ at, text: `${letter} ${LOG_WORDS[label]}` });
  }
  for (const letter of prev.keys()) {
    if (!next.has(letter)) out.push({ at, text: `${letter} left` });
  }
  return out;
}

export function appendLog(log: LogEntry[], entries: LogEntry[]): LogEntry[] {
  return [...entries, ...log].slice(0, CONFIG.logCap);
}

const pad = (n: number): string => String(n).padStart(2, '0');

export function formatEntry(e: LogEntry): string {
  return `${pad(e.at.getHours())}:${pad(e.at.getMinutes())}:${pad(e.at.getSeconds())} ${e.text}`;
}
