function errName(err: unknown): string {
  if (typeof err === 'object' && err !== null && 'name' in err) {
    const n = (err as { name: unknown }).name;
    return typeof n === 'string' ? n : '';
  }
  return '';
}

export function cameraErrorMessage(err: unknown): string {
  switch (errName(err)) {
    case 'NotAllowedError':
      return 'Camera access was blocked. Please allow camera access in the browser and reload the page.';
    case 'NotFoundError':
      return 'No camera found. Connect a camera and reload the page.';
    case 'NotReadableError':
      return 'The camera is in use by another app. Close it and reload the page.';
    default:
      return 'The camera could not be started. Check it is connected and reload the page.';
  }
}

export function modelErrorMessage(_err: unknown): string {
  return 'The face model could not be loaded. Run npm run setup:assets, then reload the page.';
}
