export type Label = 'sleeping' | 'drowsy' | 'angry' | 'laughing' | 'smiling' | 'crying' | 'neutral';
export type Letter = 'A' | 'B' | 'C' | 'D';
export interface Box { x: number; y: number; width: number; height: number } // 0..1 of the frame
export interface RawFace { box: Box; scores: Record<string, number>; pitchDeg: number }
export interface TrackedFace { letter: Letter; face: RawFace }
export interface Signals { eyeClosure: number; lookDown: number; pitchDelta: number; smile: number; jawOpen: number; browDown: number; browInnerUp: number; frown: number }
export interface Reading { label: Label; confidence: number } // confidence 0..100
export interface Person { letter: Letter; box: Box; reading: Reading }
export interface LogEntry { at: Date; text: string }
