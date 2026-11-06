import { describe, it, expect } from 'vitest';
import { cameraErrorMessage, modelErrorMessage } from '../src/core/errors';

describe('cameraErrorMessage', () => {
  it('explains denied permission', () => {
    expect(cameraErrorMessage(new DOMException('x', 'NotAllowedError'))).toContain('allow camera access');
  });
  it('explains missing camera', () => {
    expect(cameraErrorMessage(new DOMException('x', 'NotFoundError'))).toContain('No camera found');
  });
  it('explains busy camera', () => {
    expect(cameraErrorMessage(new DOMException('x', 'NotReadableError'))).toContain('another app');
  });
  it('gives a general message for unknown errors', () => {
    for (const e of [new Error('boom'), 'str', null, undefined, {}]) {
      expect(cameraErrorMessage(e).length).toBeGreaterThan(0);
    }
  });
});

describe('modelErrorMessage', () => {
  it('points to the setup script', () => {
    expect(modelErrorMessage(new Error('x'))).toContain('npm run setup:assets');
  });
});
