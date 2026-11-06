import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { CONFIG } from '../core/config';
import { pitchFromMatrix } from '../core/signals';
import type { RawFace } from '../core/types';

export class Detector {
  private constructor(private readonly landmarker: FaceLandmarker) {}

  static async create(): Promise<Detector> {
    const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    const make = (delegate: 'GPU' | 'CPU') =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: '/mediapipe/face_landmarker.task', delegate },
        runningMode: 'VIDEO',
        numFaces: CONFIG.maxFaces,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
    try {
      return new Detector(await make('GPU'));
    } catch {
      return new Detector(await make('CPU'));
    }
  }

  detect(video: HTMLVideoElement, nowMs: number): RawFace[] {
    const result = this.landmarker.detectForVideo(video, nowMs);
    const faces: RawFace[] = [];
    result.faceLandmarks.forEach((landmarks, i) => {
      if (landmarks.length === 0) return;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of landmarks) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      const scores: Record<string, number> = {};
      for (const c of result.faceBlendshapes?.[i]?.categories ?? []) {
        scores[c.categoryName] = c.score;
      }
      const matrix = result.facialTransformationMatrixes?.[i];
      const pitchDeg = matrix ? pitchFromMatrix(matrix.data) : 0;
      faces.push({ box: { x: minX, y: minY, width: maxX - minX, height: maxY - minY }, scores, pitchDeg });
    });
    return faces;
  }
}
