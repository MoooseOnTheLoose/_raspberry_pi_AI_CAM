# AICAM (Raspberry Pi) — Event-Driven Vision System

A **stateful, auditable, event-driven vision system** with **deterministic outputs** and **controlled resource ownership**.

This project is intentionally designed to:
- survive long runtimes
- produce explainable artifacts
- evolve without fragile coupling

---

## What This Project Is / Is Not

### What this project **is**
- A single-purpose, offline vision system
- Optimized for Raspberry Pi using Picamera2
- Designed for long-running, unattended operation
- Focused on **events first**, media second

### What this project **is not**
This project is intentionally **not**:
- a real-time notification or alerting system
- a cloud-dependent or streaming solution
- a frame-by-frame inference logger
- a generic camera wrapper

Alerts are represented as **logged events with attached artifacts**, not pop-ups or messages.

---

## Core Design Philosophy

Signal > Frames  
Events > Video  
System > Scripts  

The system emphasizes **reliable, explainable events** over raw media capture.  
Video and images are treated as **artifacts attached to events**, not the primary output.

---

## Architecture at a Glance

Two complementary intelligence layers exist in the project:

- **Tier 1:** IMX500 (on-sensor AI) — efficient, best for stills, but fragile metadata coupling.
- **Tier 2:** Picamera2 + **ONNX Runtime (YOLO ONNX)** — portable, debuggable, production-consistent (**primary production path**).

OpenCV DNN remains supported in legacy modules but is no longer the preferred inference backend.

---

## Intelligence Layers

### Tier 1 — IMX500 (On-Sensor AI)

IMX500 (On-Sensor AI)  
│  
│  stdout / metadata  
▼  
Python (parse)  
│  
▼  
OpenCV (draw)  
│  
▼  
JPEG  

**Strengths**
- Efficient, low power
- Great for stills
- Hardware-accelerated inference

**Tradeoffs**
- Fragile stdout/metadata coupling
- Best-effort overlays depend on consistent metadata formats
- Single-owner constraints apply

---

### Tier 2 — Picamera2 + ONNX Runtime (Production-Consistent)

Picamera2  
│  
├── Main Stream (RGB)  
│       │  
│       ▼  
│   Burn-in Overlay (OpenCV)  
│       │  
│       ▼  
│   JPEG / MP4  
│  
└── Lores / Downscaled Stream (RGB)  
        │  
        ▼  
   ONNX Runtime (YOLO)  
        │  
        ▼  
   Detection Results  

**Strengths**
- Portable and debuggable
- Stable for long-running services
- Clean separation between capture / inference / overlay / events
- Deterministic, auditable outputs

---

## Critical Rule: Controlled Resource Ownership

ONE CAMERA OWNER  
start → run → stop (finally)  
NO parallel access  

This rule prevents:
- `pipeline handler in use by another process`
- race conditions in libcamera / Picamera2
- unpredictable capture failures

---

## Applications

### AICAM_photo — Still Capture + Detection

Still Capture  
│  
▼  
ML Inference (ONNX / legacy DNN)  
│  
▼  
Overlay Render  
│  
├── JPEG (annotated image)  
├── JSON (detections + metadata)  
└── Event Log (if policy triggers)  

**What it produces**
- A single annotated image (JPEG)
- A structured metadata record (JSON)
- An append-only event log entry

---

### AICAM_VIDSec — Continuous Video Security Recorder

Video Stream  
│  
▼  
ML Inference (throttled)  
│  
▼  
State Machine  
(person present?)  
│  
├── HUMAN_ENTER → log event  
├── HUMAN_EXIT  → log event  
│  
▼  
Burn-in Overlay  
│  
▼  
Segmented MP4 Clips  

**What it produces**
- Continuous segmented MP4 clips
- Burned-in overlay (boxes + labels + confidence)
- Event log entries with clip-relative timestamps

---

## State & Policy

The system operates as a **state machine**, not a frame-by-frame logger.

STATE:  
person_present = True / False  

POLICY:  
HUMAN_ENTER → log + annotate  
HUMAN_EXIT  → log only  

In this system, **events are the alert mechanism**.  
A logged state transition (e.g., `HUMAN_ENTER`, `TRIGGER`) is the authoritative signal that something occurred.

---

## Long-Running Assumptions

This system is designed to run:
- unattended
- for long durations
- with predictable resource usage

Design choices such as state machines, clip segmentation, cooldowns, and append-only logs exist specifically to prevent drift, leaks, and event spam over time.

---

## Determinism & Traceability (Audit Layer)

Deterministic Outputs  
├── Model SHA256 hashes  
├── OpenCV / Python / OS versions  
├── Clip-relative timestamps  
└── Append-only event logs  

This ensures:
- reproducibility across machines
- traceable changes when performance shifts
- credible, explainable records over time

---

## Output Artifacts

### Directory selection
Outputs are written to:
- Preferred: `/media/user/disk/...` (when mounted and writable)
- Fallback: `/home/user/...`

### Typical artifacts

**Still mode**
- `image_<ts>.jpg` (annotated)
- `meta_<ts>.json` (detections + versions + hashes)
- `events.log` (append-only)

**Video mode**
- `secAI_<session>_<seq>_<ts>.mp4`
- `events.log` containing:
  - `HUMAN_ENTER` / `HUMAN_EXIT`
  - `clip=<filename>`
  - `t=<seconds>` (clip-relative timestamp)
  - confidence + bbox for enter events

---

## Production Compatibility Notes

### Picamera2 API drift (FFmpeg output)

Picamera2 versions vary. This project uses **capability detection**, not brittle version checks:

```python
def make_ffmpeg_output(mp4_path: str):
    try:
        return FfmpegOutput(mp4_path, audio=False, options=["-movflags", "+faststart"])
    except TypeError:
        try:
            return FfmpegOutput(mp4_path, audio=False)
        except TypeError:
            return FfmpegOutput(mp4_path)
```


---

## Event Glossary

This section defines the canonical meaning of events written to `events.log` and `events.jsonl`.
Events are the **authoritative alert mechanism** in AICAM.

### TRIGGER
A confirmed detection that satisfies all gating conditions (confidence, persistence, policy).
A `TRIGGER` event indicates that the system has decided an event is **real** and actionable.

- For single-class pipelines (e.g., drone detection), `TRIGGER` implicitly means **target detected**
- A clip is started when a `TRIGGER` occurs

---

### HUMAN_ENTER
A state transition indicating that a person has entered the scene.

- Emitted once per presence interval
- Used to avoid repeated alerts while a person remains visible
- Always logged with clip context when applicable

---

### HUMAN_EXIT
A state transition indicating that a previously detected person has left the scene.

- Emitted once per presence interval
- Marks the end of an active presence state
- Does not generate new media by itself

---

### CLIP_START
Indicates that a recording segment has started.

- Includes clip filename and context
- May be caused by a `TRIGGER` or by clip segmentation logic

---

### CLIP_END
Indicates that a recording segment has ended.

Common reasons include:
- `clip_len` — maximum clip duration reached
- `lost` — target no longer present
- `shutdown` — system stopping

---

### reason: clip_len
A clip termination reason meaning:

> The clip ended because it reached the configured maximum duration.

This is expected behavior during long detections and prevents unbounded file growth.

---

### events.log vs events.jsonl

- `events.log`  
  Human-readable, append-only log for review and auditing.

- `events.jsonl`  
  Machine-readable, append-only event stream suitable for parsing, correlation, or automation.

Both files describe the **same events** at different fidelity levels.


---

## Quick Start (Modern / Recommended)

These commands use the **modern ONNX Runtime (YOLO) pipeline** and are suitable for most deployments.

### 1) Minimal run (headless, safest)
Use this first to confirm the system works end-to-end.

```bash
python3 AICAM_Drone_YOLO_Only.py \
  --yolo-model /home/user/models/best.onnx
```

- No preview
- No overlays
- Writes clips and logs only when events trigger
- Recommended for unattended operation

---

### 2) Tuned run (recommended defaults)
Balanced configuration for outdoor drone detection.

```bash
python3 AICAM_Drone_YOLO_Only.py \
  --yolo-model /home/user/models/best.onnx \
  --conf 0.35 \
  --min-area-px 2000 \
  --confirm-hits 4 \
  --confirm-window-s 1.5
```

This reduces false positives by requiring:
- sufficient confidence
- minimum object size
- persistence over time

---

### 3) Save clean + annotated clips (verification mode)
Use this to visually verify detections while keeping clean evidence.

```bash
python3 AICAM_Drone_YOLO_Only.py \
  --yolo-model /home/user/models/best.onnx \
  --annotate-clips secondary
```

Produces:
- clean clip (`*.mp4`)
- annotated clip (`*_annot.mp4`)

---

### 4) Live preview (system Python only)
Preview requires a desktop/X session and **system Python** (not a venv).

```bash
deactivate
python3 AICAM_Drone_YOLO_Only.py \
  --yolo-model /home/user/models/best.onnx \
  --preview \
  --annotate-preview
```

If preview fails inside a virtual environment, this is expected behavior.

---

### 5) Where to find outputs
By default, outputs are written to:

- Preferred: `/media/user/disk/...`
- Fallback: `/home/user/...`

See `logs/events.log` for authoritative alerts and state transitions.


---

## Troubleshooting (Common Failures)

This section documents the most common operational issues seen on Raspberry Pi deployments and their resolutions.

### 1) Preview aborts with Qt / xcb errors

**Symptoms**
- Program exits immediately
- Errors mentioning `Qt`, `xcb`, or `platform plugin`

**Cause**
- Preview uses `cv2.imshow()` which depends on Qt/X11
- Python virtual environments do not reliably inherit GUI libraries
- No active desktop session (`DISPLAY` unset)

**Resolution**
- Run headless (no `--preview`) and rely on annotated clips, **or**
- Exit the virtual environment and run using system Python from a desktop/X session

This behavior is expected.

---

### 2) Picamera2 import errors or NumPy ABI issues

**Symptoms**
- Import errors involving `picamera2` or `simplejpeg`
- Errors referencing NumPy ABI mismatch

**Cause**
- Mixing `pip`-installed NumPy with `apt`-installed Picamera2 dependencies
- Upgrading NumPy inside a venv breaks binary compatibility

**Resolution**
- Use system Python with `apt`-managed NumPy for runtime
- Avoid upgrading NumPy via `pip` on Raspberry Pi
- Reserve virtual environments for training/export only

---

### 3) Permission errors on external mounts

**Symptoms**
- Clips or logs fail to write
- Permission denied errors
- Base directory falls back unexpectedly

**Cause**
- External mount not writable by the current user
- Running the script with `sudo` changes ownership and permissions

**Resolution**
- Run as the normal user (not `sudo`)
- Confirm mount permissions (`ls -ld /media/user/disk`)
- Rely on automatic fallback behavior if the mount is unavailable

---

### Final Note

Most issues are environmental rather than code defects.  
The system is designed to **fail safely**, log clearly, and continue operating whenever possible.
