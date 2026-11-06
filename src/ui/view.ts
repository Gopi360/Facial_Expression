import { formatEntry } from '../core/activityLog';
import { CONFIG } from '../core/config';
import { faceCountText, personLine } from '../core/text';
import type { Label, LogEntry, Person } from '../core/types';

const BOX_COLOUR = '#22c55e';
const ALERT_BOX_COLOUR = '#ef4444';

function isAlert(label: Label): boolean {
  return label === 'sleeping' || label === 'drowsy';
}

/** Draws only a thin box and a small letter per face. Pass only faces visible this frame. */
export function drawOverlay(canvas: HTMLCanvasElement, people: Person[]): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  const lineWidth = height * CONFIG.overlayLineFrac;
  const letterSize = height * CONFIG.overlayLetterFrac;
  ctx.lineWidth = lineWidth;
  ctx.font = `600 ${letterSize}px 'Segoe UI', system-ui, sans-serif`;
  ctx.textAlign = 'left';

  for (const { letter, box, reading } of people) {
    const colour = isAlert(reading.label) ? ALERT_BOX_COLOUR : BOX_COLOUR;
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;

    // The video is mirrored by CSS and this canvas is not, so flip the position here.
    const x = (1 - box.x - box.width) * width;
    const y = box.y * height;
    ctx.strokeRect(x, y, box.width * width, box.height * height);

    const fitsAbove = y >= letterSize + lineWidth;
    ctx.textBaseline = fitsAbove ? 'bottom' : 'top';
    ctx.fillText(letter, x, fitsAbove ? y - lineWidth : y + lineWidth);
  }
}

export function renderStrip(el: HTMLElement, people: Person[]): void {
  const count = document.createElement('div');
  count.className = 'strip-count';
  count.textContent = faceCountText(people.length);

  const lines = document.createElement('div');
  lines.className = 'strip-people';
  for (const person of people) {
    const line = document.createElement('div');
    line.className = isAlert(person.reading.label) ? 'strip-person alert' : 'strip-person';
    line.textContent = personLine(person);
    lines.append(line);
  }
  el.replaceChildren(count, lines);
}

export function renderLog(el: HTMLElement, log: LogEntry[]): void {
  el.replaceChildren(
    ...log.map((entry) => {
      const item = document.createElement('li');
      item.textContent = formatEntry(entry);
      return item;
    }),
  );
}

export function showMessage(el: HTMLElement, text: string | null): void {
  el.textContent = text ?? '';
  el.hidden = text === null;
}
