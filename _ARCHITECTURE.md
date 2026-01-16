# 🧠 AICAM Architecture

> **A stateful, auditable, event-driven vision system**
>
> *Media is evidence. Events are the product.*

---

This document defines the **internal architecture**, **state machine**, **thresholding model**, and **artifact schemas** for the AICAM project.

It is intended as a **long-term reference** for learning, maintenance, extension, and audit.

---

## 📚 Table of Contents

1. Architectural Goals  
2. System Layers  
3. Camera Ownership Model  
4. Inference Model  
   - 4.1 Detection Engine  
   - 4.2 Inference Throttling  
5. Classification Tiers & Thresholds  
6. State Machine (Human Presence)  
7. Event Logging Model  
8. Overlay Rendering  
9. Artifact Schemas  
10. Determinism & Auditability  
11. Failure & Shutdown Behavior  
12. Summary  

---

## 1. Architectural Goals

AICAM is designed to be:

- **Stateful** — decisions depend on temporal context, not single frames  
- **Auditable** — every output can be traced to a model, version, and policy  
- **Event-driven** — events are the primary product, not media  
- **Deterministic** — identical inputs produce identical outputs  
- **Resource-safe** — camera and disk ownership are strictly controlled  

---

## 2. System Layers

```
┌────────────────────────────┐
│        Hardware Layer      │
│  Camera + (optional) IMX500│
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│     Capture Layer          │
│  Picamera2 (single owner)  │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│   Inference Layer          │
│  OpenCV DNN (throttled)    │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│   State Machine Layer      │
│  Presence / transitions   │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│   Output Layer             │
│  Overlay / Logs / Artifacts│
└────────────────────────────┘
```

Each layer has a **single responsibility** and communicates via **explicit data structures**, not implicit side effects.

---

## 3. Camera Ownership Model

### Rules
- Only **one process** may access the camera at a time  
- No background preview processes  
- No concurrent still + video pipelines  

### Enforced by
- Single Picamera2 instance  
- Explicit lifecycle control with `finally` cleanup  
- Optional systemd unit locking  

Prevents:
- `pipeline handler in use` errors  
- libcamera race conditions  
- undefined hardware behavior  

---

## 4. Inference Model

### 4.1 Detection Engine

- OpenCV DNN  
- Caffe MobileNet-SSD  
- Inference runs on the **lores stream**  
- Bounding boxes are scaled to main-stream coordinates  

---

### 4.2 Inference Throttling

Inference does **not** run on every frame.

```
INFER_EVERY_N_FRAMES = N
```

At 15 FPS:
- `N = 8` → ~1.9 inferences/sec  
- Prevents CPU starvation  
- Keeps video encoding stable  

---

## 5. Classification Tiers & Thresholds

### 5.1 Class Tiers

**Primary Classes (high importance)**

```
person
dog
cat
car
bus
motorbike
bicycle
```

**Secondary Classes (contextual)**

```
bird
chair
sofa
bottle
tvmonitor
```

All other classes are ignored by default.

---

### 5.2 Confidence Thresholds

| Tier           | Minimum Confidence |
|----------------|--------------------|
| Primary        | 0.50               |
| Secondary      | 0.60               |
| Event (person) | 0.50               |

Thresholds favor **precision and explainability** over recall.

---

## 6. State Machine (Human Presence)

AICAM operates on **temporal state**, not per-frame reactions.

### 6.1 State Variable

```python
person_present: bool
```

Represents whether a human is currently considered present in the scene.

---

### 6.2 Timers & Guards

| Parameter                  | Purpose |
|---------------------------|--------|
| PERSON_EXIT_HOLD          | Time without detection required to trigger exit |
| PERSON_REENTER_COOLDOWN   | Prevents rapid enter/exit oscillation |
| INFER_EVERY_N_FRAMES      | Controls inference frequency |

Timers ensure stable behavior under noisy detections.

---

### 6.3 State Transitions

```
ABSENT
  │
  │  person detected (conf ≥ threshold)
  ▼
PRESENT
  │
  │  no detection for EXIT_HOLD duration
  ▼
ABSENT
```

Transitions are **edge-triggered**, not continuous.

---

### 6.4 Events

| Event       | Description |
|------------|-------------|
| HUMAN_ENTER | First valid detection after absence |
| HUMAN_EXIT  | Sustained absence after presence |

Only state changes produce events.

---

## 7. Event Logging Model

Events are **append-only**, human-readable, and auditable.

### 7.1 Log Format

```
ISO8601_TIMESTAMP | EVENT | clip=<filename> | t=<seconds> | conf | bbox
```

### 7.2 Examples

```
2026-01-14T19:32:10 | HUMAN_ENTER | clip=secAI_0003.mp4 | t=42.31 | 0.873 | 122,88,402,612
2026-01-14T19:35:51 | HUMAN_EXIT  | clip=secAI_0003.mp4 | t=263.08
```

Notes:
- `t` is clip-relative time, not wall-clock  
- ENTER includes confidence and bounding box  
- EXIT is informational only  

---

## 8. Overlay Rendering

Overlays are burned into output media to ensure portability.

### 8.1 Header Overlay

Header includes:
- System identifier  
- Wall-clock timestamp  
- Detection summary (top N detections)  

### 8.2 Bounding Boxes

- Color-coded by tier  
  - Primary: green  
  - Secondary: yellow  

- Label format:
```
<class> <confidence%>
```

Rendered **after state filtering**, not raw inference.

---

## 9. Artifact Schemas

### 9.1 Still Capture Artifacts

**Outputs**
- Annotated JPEG  
- JSON metadata  
- Optional event log entry  

**Metadata Schema (Still)**

```json
{
  "timestamp": "YYYYMMDD_HHMMSS",
  "camera": { "size": [1280, 720] },
  "detections": [
    {
      "label": "person",
      "confidence": 0.87,
      "bbox_xyxy": [x1, y1, x2, y2],
      "tier": "primary"
    }
  ],
  "versions": {
    "python": "3.x",
    "opencv": "4.x",
    "os": "Linux"
  },
  "model": {
    "name": "MobileNet-SSD",
    "sha256": "..."
  }
}
```

---

### 9.2 Video Session Artifacts

**Outputs**
- Segmented MP4 clips  
- Session metadata JSON  
- Append-only event log  

**Session Metadata**

```json
{
  "session_ts": "YYYYMMDD_HHMMSS",
  "clip_seconds": 600,
  "fps": 15,
  "model_hashes": {
    "prototxt": "...",
    "caffemodel": "..."
  },
  "thresholds": {
    "primary": 0.50,
    "secondary": 0.60,
    "event": 0.50
  }
}
```

---

## 10. Determinism & Auditability

Determinism is ensured by recording:
- Model SHA256 hashes  
- OpenCV / Python / OS versions  
- Fixed inference parameters  
- Clip-relative timestamps  
- Append-only event logs  

This enables **post-hoc reconstruction** of system behavior.

---

## 11. Failure & Shutdown Behavior

On:
- low disk  
- SIGINT / SIGTERM  
- unexpected exception  

The system guarantees:
- encoder stop  
- camera release  
- final event log entry  

No orphaned pipelines.  
No silent corruption.

---

## 12. Summary

AICAM is not a demo pipeline.

It is a **long-running, production-oriented vision system** designed around:
- explicit state  
- explainable decisions  
- controlled resources  
- durable artifacts  

**Media is evidence.  
Events are the product.**
