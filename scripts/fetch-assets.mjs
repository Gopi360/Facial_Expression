import { cpSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const wasmSrc = resolve(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const wasmDest = resolve(root, 'public/mediapipe/wasm');
const modelDest = resolve(root, 'public/mediapipe/face_landmarker.task');
const modelUrl =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

async function main() {
  if (!existsSync(wasmSrc)) {
    throw new Error(`WASM folder not found at ${wasmSrc}. Run "npm install" first.`);
  }
  mkdirSync(wasmDest, { recursive: true });
  cpSync(wasmSrc, wasmDest, { recursive: true });
  console.log('Copied MediaPipe WASM files to public/mediapipe/wasm');

  if (existsSync(modelDest)) {
    console.log('Model already present, skipping download');
    return;
  }
  console.log('Downloading face landmarker model...');
  const res = await fetch(modelUrl);
  if (!res.ok) {
    throw new Error(`Model download failed: HTTP ${res.status} ${res.statusText}`);
  }
  writeFileSync(modelDest, Buffer.from(await res.arrayBuffer()));
  console.log('Saved public/mediapipe/face_landmarker.task');
}

main().catch((err) => {
  console.error(`setup:assets failed: ${err instanceof Error ? err.message : err}`);
  process.exit(1);
});
