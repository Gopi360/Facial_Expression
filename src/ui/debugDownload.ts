import { fileStamp } from './snapshot';

/** Saves the debug log through the browser's normal download. Nothing is uploaded. */
export function saveDebugLog(json: string): void {
  const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
  try {
    const link = document.createElement('a');
    link.href = url;
    link.download = `debug-log-${fileStamp(new Date())}.json`;
    link.click();
  } finally {
    // The download has its own hold on the data, so the URL can go on the next turn.
    setTimeout(() => URL.revokeObjectURL(url));
  }
}
