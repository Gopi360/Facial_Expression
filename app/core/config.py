"""All tunable numbers live here. Values carried over from the TypeScript demo (snake_case)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # tracking and baseline
    max_faces: int = 4
    min_frame_gap_ms: int = 66
    letter_release_ms: int = 2000
    max_match_distance: float = 0.25
    baseline_ms: int = 2000
    baseline_min_samples: int = 10
    baseline_max_wait_ms: int = 10000
    baseline_fallback_percentile: float = 0.2
    baseline_fallback_max_closure: float = 0.7
    baseline_open_eyes_below: float = 0.5
    # sleep rules
    eyes_closed_at: float = 0.55
    eyes_closed_when_looking_down_at: float = 0.8
    look_down_at: float = 0.4
    open_grace_ms: int = 200
    slow_blink_ms: int = 400
    slow_blink_window_ms: int = 10000
    slow_blinks_for_drowsy: int = 2
    droopy_eyes_at: float = 0.35
    droopy_hold_ms: int = 3000
    head_droop_deg: float = 15
    head_droop_eyes_at: float = 0.45
    head_droop_hold_ms: int = 1500
    drowsy_clear_ms: int = 2000
    drowsy_min_confidence: float = 0.5
    # labels
    expression_limit_ms: int = 5000
    label_hold_ms: int = 500
    strong_dip_grace_ms: int = 200
    frame_gap_reset_ms: int = 2500
    smile_at: float = 0.35
    strong_smile_at: float = 0.5
    laugh_jaw_at: float = 0.25
    angry_brow_at: float = 0.4
    cry_brow_at: float = 0.4
    cry_frown_at: float = 0.3
    angry_smile_max: float = 0.2
    pitch_sign: int = 1
    # alarm and log
    default_delay_sec: int = 3
    min_delay_sec: int = 1
    max_delay_sec: int = 10
    log_cap: int = 100
    paused_note_ms: int = 3000
    debug_buffer_ms: int = 30000
    debug_range_ms: int = 10000
    # python rebuild additions
    target_fps: int = 15
    yawn_jaw_at: float = 0.5
    yawn_hold_ms: int = 1500
    emotion_interval_ms: int = 333
    identity_interval_ms: int = 1000
    identity_match_at: float = 0.363
    identity_pending_ms: int = 2000
    identity_max_signatures: int = 5
    frontal_yaw_deg: float = 20
    min_red_ms: int = 1000
    jpeg_max_bytes: int = 2000000


CONFIG = Config()
