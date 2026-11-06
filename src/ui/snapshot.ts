const pad = (n: number): string => String(n).padStart(2, '0');

function fileName(d: Date): string {
  const date = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
  const time = `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return `snapshot-${date}-${time}.png`;
}

/** Saves the picture as seen on screen. Throws if the canvas cannot be used. */
export function saveSnapshot(video: HTMLVideoElement, overlay: HTMLCanvasElement): void {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No 2D canvas context');

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);

  const link = document.createElement('a');
  link.href = canvas.toDataURL('image/png');
  link.download = fileName(new Date());
  link.click();
}
