"""All tunable numbers live here. Values carried over from the TypeScript demo (snake_case)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # tracking and baseline
    max_faces: int = 4
    letter_release_ms: int = 2000
    max_match_distance: float = 0.25
    baseline_ms: int = 2000
    baseline_min_samples: int = 10
    baseline_max_wait_ms: int = 10000
    baseline_fallback_percentile: float = 0.2
    baseline_fallback_max_closure: float = 0.7
    baseline_open_eyes_below: float = 0.5
    # sleep rules
    eyes_closed_at: float = 0.33
    open_grace_ms: int = 200
    slow_blink_ms: int = 400
    slow_blink_window_ms: int = 10000
    slow_blinks_for_drowsy: int = 2
    droopy_eyes_at: float = 0.22
    droopy_hold_ms: int = 3000
    head_droop_deg: float = 16
    head_droop_eyes_at: float = 0.08
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
    debug_buffer_ms: int = 90000
    debug_range_ms: int = 10000
    # python rebuild additions
    target_fps: int = 15
    yawn_jaw_at: float = 0.5
    yawn_hold_ms: int = 1500
    emotion_interval_ms: int = 333
    identity_interval_ms: int = 1000
    identity_match_at: float = 0.363
    identity_pending_ms: int = 2000
    identity_max_signatures: int = 5  # rolling slots, on top of the anchors
    identity_anchor_signatures: int = 3  # first good views, kept for the whole session
    identity_mismatch_checks: int = 2  # frontal checks in a row that must fail to lose a letter
    frontal_yaw_deg: float = 20
    frontal_pitch_delta_deg: float = 12  # the gate for storing a view
    identity_check_pitch_delta_deg: float = 30  # looser, for checking a letter: must be under this
    min_red_ms: int = 1000
    jpeg_max_bytes: int = 2000000
    overlay_line_frac: float = 0.003
    overlay_letter_frac: float = 0.045
    # the page and the session
    frame_width: int = 640
    frame_height: int = 480
    jpeg_quality: float = 0.8
    beep_hz: int = 880
    beep_on_ms: int = 300
    beep_off_ms: int = 200
    beep_peak_gain: float = 0.3
    warning_hold_ms: int = 5000
    reply_timeout_ms: int = 5000
    frame_finish_grace_ms: int = 1000


CONFIG = Config()
