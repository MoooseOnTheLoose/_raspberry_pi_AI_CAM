# DRONE_TUNING.md

## Purpose

This document provides **drone-specific camera and detection tuning guidance** for the _Rasp_ project.

Unlike static cameras, drone and mobile platforms introduce:
- Continuous motion
- Vibration
- Rapid exposure changes
- CPU contention under flight load

Drone tuning therefore focuses on **temporal stability, inference suppression during motion, and controlled recovery**, not image aesthetics.

---

## Design Goals for Drone Operation

For drones, the project prioritizes:

- Stable frame cadence under motion
- Suppression of inference during vibration
- Fast but controlled recovery after motion stabilizes
- Bounded CPU usage
- Predictable detection behavior

Visual smoothness and sharpness are secondary.

---

## Baseline Drone Camera Configuration

### Resolution
Lower resolution reduces latency and stabilizes inference timing.

**Recommended:**
- `960×540` (default)
- Reduce further if CPU headroom is limited

Avoid 1080p unless flight time is short and load is well characterized.

### Frame Rate (FPS)

Higher FPS on drones is used to:
- Reduce motion blur
- Improve temporal alignment during movement

**Recommended:**
- `25–30 FPS` at reduced resolution (day)
- `20–25 FPS` at reduced resolution (night)

FPS should never be increased without reducing resolution.

### Exposure Strategy

Exposure behavior is critical during flight.

**Guidelines:**
- Start with auto-exposure enabled
- Prefer shorter exposure times
- Accept increased noise rather than blur

Blur during motion causes missed detections and unstable confidence scores.

### Gain / Noise Tradeoff

- Allow moderate gain
- Avoid unbounded gain ranges
- Noise is preferable to blur

If gain becomes excessive, reduce FPS or resolution before fixing exposure.

---

## Motion Gating (Critical for Drones)

The drone script uses **motion-based inference suppression** to avoid false triggers during flight.

### Why motion gating is required
- Frame-to-frame differences are dominated by movement
- Inference during vibration is unreliable
- Confirmation windows are invalid under motion

---

## Motion Metric Configuration

### Center-ROI Motion Metric

The project uses a **center ROI motion metric** to reduce false gating from:
- Edge flow
- Sky exposure changes
- Prop-induced flicker

**Recommended:**
- `--motion-roi-scale 0.5`

### Motion Thresholds (Hysteresis)

Use separate enter/exit thresholds to prevent oscillation.

**Typical starting points:**
- Day: `18 / 12`
- Night: `14–16 / 9–11`

### Consecutive Frame Gating

Motion state changes require multiple consecutive frames.

**Recommended:**
- Day: `2 / 2`
- Night: `2–3 / 3–4`

### Settle Frames

After motion calms, detection remains suppressed briefly.

**Recommended:**
- Day: `4`
- Night: `6–8`

---

## Inference Pacing (CPU Protection)

### Max Inference FPS

Hard-caps inference to prevent starvation of capture and recording.

**Recommended:**
- Day: `6–8`
- Night: `4–6`

### Adaptive Frame Skipping

Under load, detection frequency is reduced automatically.

**Recommended:**
- Enable `--adaptive-skip`
- Day: `max-skip-factor 4–6`
- Night: `max-skip-factor 6–8`

---

## Field Deployment Presets (Recommended)

Presets allow repeatable, auditable field deployment without re-tuning CLI flags.

### Day / General Flight  
**`DRONE_PRESET.json`**
```json
{
  "profile_name": "drone_field_balanced",
  "camera": {
    "width": 960,
    "height": 540,
    "fps": 30,
    "ae": 1,
    "awb": 1
  },
  "inference": {
    "max_infer_fps": 8,
    "adaptive_skip": true,
    "max_skip_factor": 4,
    "infer_ema_alpha": 0.2
  },
  "motion_gate": {
    "enabled": true,
    "motion_threshold": 18.0,
    "motion_threshold_exit": 12.0,
    "motion_high_frames": 2,
    "motion_low_frames": 2,
    "motion_settle_frames": 4,
    "motion_roi_scale": 0.5,
    "motion_downscale": 0.25
  }
}
```

### Night Drone – Balanced  
**`DRONE_PRESET_NIGHT_BALANCED.json`**
```json
{
  "profile_name": "drone_field_night_balanced",
  "camera": {
    "width": 960,
    "height": 540,
    "fps": 25,
    "ae": 1,
    "awb": 1
  },
  "inference": {
    "max_infer_fps": 6,
    "adaptive_skip": true,
    "max_skip_factor": 6,
    "infer_ema_alpha": 0.2
  },
  "motion_gate": {
    "enabled": true,
    "motion_threshold": 14.0,
    "motion_threshold_exit": 9.0,
    "motion_high_frames": 2,
    "motion_low_frames": 3,
    "motion_settle_frames": 6,
    "motion_roi_scale": 0.5,
    "motion_downscale": 0.25
  }
}
```

**Use when:**
- Low light but not extreme
- Moderate vibration
- You want detection recovery within a reasonable window

### Night Drone – Conservative  
**`DRONE_PRESET_NIGHT_CONSERVATIVE.json`**
```json
{
  "profile_name": "drone_field_night_conservative",
  "camera": {
    "width": 832,
    "height": 468,
    "fps": 20,
    "ae": 1,
    "awb": 1
  },
  "inference": {
    "max_infer_fps": 4,
    "adaptive_skip": true,
    "max_skip_factor": 8,
    "infer_ema_alpha": 0.2
  },
  "motion_gate": {
    "enabled": true,
    "motion_threshold": 16.0,
    "motion_threshold_exit": 11.0,
    "motion_high_frames": 3,
    "motion_low_frames": 4,
    "motion_settle_frames": 8,
    "motion_roi_scale": 0.5,
    "motion_downscale": 0.25
  }
}
```

**Use when:**
- Very low light
- Noise dominates the image
- CPU headroom is limited
- Stability is more important than detection speed

---

## Recommended Preset Usage Pattern

1. Load preset at startup  
2. Apply values as defaults  
3. Allow explicit CLI flags to override preset values  
4. Log the effective configuration at startup  

This ensures reproducibility and safe overrides.

---

## Drone Field-Tuning Checklist

- FPS remains stable during flight  
- Inference suppresses fully during vibration  
- Detection resumes only after motion settles  
- CPU usage remains bounded  
- No confirmation accumulation during motion  
- False positives remain minimal after motion  

---

## Key Principle for Drone Operation

> **On a drone, most frames are unusable for detection.  
> A good system knows when *not* to infer.**
