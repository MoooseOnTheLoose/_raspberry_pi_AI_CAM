#!/usr/bin/env python3
"""
AICAM_Drone.py — Airborne object clipper (Drone/Airplane)

Legacy Caffe-SSD implementation
Maintained for compatibility and reference
Not recommended for new deployments. Manipukates the classes
bird and aeroplane to trick that it is a drone. 

Outputs (project parity target)
- logs/aicam_drone.log        : rotating operational log (unchanged)
- logs/events.jsonl           : rotating structured events (unchanged)
- logs/events.log             : project-compatible text events (added)
- clips/<clip>.mp4            : recorded clip
- clips/<clip>.json           : per-clip metadata (added)

Design alignment 
- detect -> confirm -> trigger -> record -> cooldown
- explicit state machine with shared Target abstraction

Scope constraints
- Video-only recorder. No autopilot/actuation logic.
- Detector: OpenCV DNN + Caffe SSD (MobileNet-SSD style)
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
import enum
import json
import logging
import logging.handlers
from pathlib import Path
import shutil
import signal
import time
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2
Polygon = List[Tuple[int, int]]


# ----------------------------
# Defaults (project config style)
# ----------------------------

# Storage
MOUNT = "/media/user/disk"
BASE_SUBDIR = "aicam_drone"
TMP_SUBDIR = "tmp"          # internal temp files
FALLBACK_DIR = "/home/user/aicam_drone"
EVENT_LOG_NAME = "events.log"
MIN_FREE_GB = 10.0          # skip triggers if below this
DEVICE_ID = "aircam_01"    # device/camera identifier

# AI model (Caffe MobileNet-SSD)
MODEL_DIR = "/home/user/models"
PROTOTXT = str(Path(MODEL_DIR) / "deploy.prototxt")
CAFFEMODEL = str(Path(MODEL_DIR) / "mobilenet_iter_73000.caffemodel")


# ----------------------------
# Data model (project parity)
# ----------------------------

class State(str, enum.Enum):
    INIT = "INIT"
    STANDBY = "STANDBY"
    ACQUIRE_TARGET = "ACQUIRE_TARGET"
    TRACK = "TRACK"
    LOST_TARGET = "LOST_TARGET"
    FAILSAFE = "FAILSAFE"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class Target:
    """Shared target abstraction (kept intentionally minimal)."""
    id: str
    classification: str
    confidence: float
    position: Tuple[float, float, float]  # x, y, z (z=0 for 2D video)
    velocity: Tuple[float, float, float]  # vx, vy, vz (vz=0)
    last_seen_mono: float
    bbox: Optional[Box] = None


@dataclass
class Detection:
    cls_name: str
    conf: float
    box: Box


# ----------------------------
# Time helpers
# ----------------------------

def ts_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ts_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ts_utc_compact() -> str:
    # 20260116T235959Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ts_local_human() -> str:
    # 2026-01-16 18:01:02
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# Storage selection 
# ----------------------------

def pick_base_dir(mount: str, base_subdir: str, fallback_dir: str) -> Path:
    """
    Prefer removable mount if present; else fallback.
    Creates base_subdir under the chosen base.
    """
    mount_path = Path(mount)
    if mount and mount_path.exists() and mount_path.is_dir():
        base = mount_path / base_subdir
    else:
        base = Path(fallback_dir) / base_subdir

    base.mkdir(parents=True, exist_ok=True)
    (base / "clips").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / TMP_SUBDIR).mkdir(parents=True, exist_ok=True)
    return base


def free_space_gb(path: Path) -> float:
    """Return free space (GiB) for the filesystem containing path."""
    try:
        usage = shutil.disk_usage(str(path))
        return float(usage.free) / (1024.0 ** 3)
    except Exception:
        return 0.0


# ----------------------------
# Logging (operational + JSONL + text events)
# ----------------------------

def setup_rotating_logger(name: str, path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if logger.handlers:
        return logger

    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class JsonlEvents:
    def __init__(self, path: Path, max_bytes: int = 2_000_000, backup_count: int = 5) -> None:
        self._logger = logging.getLogger("events_aicam_drone_jsonl")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(fh)

    def emit(self, payload: Dict) -> None:
        try:
            self._logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            pass


class TextEvents:
    """
    project-compatible events.log.

    Format:
      YYYY-mm-dd HH:MM:SS | EVENT | key=value | key=value ...
    """
    def __init__(self, path: Path, max_bytes: int = 2_000_000, backup_count: int = 5) -> None:
        self._logger = logging.getLogger("events_aicam_drone_text")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(fh)

    def line(self, event: str, **fields: object) -> None:
        try:
            parts = [ts_local_human(), "|", event]
            for k, v in fields.items():
                parts.append("|")
                parts.append(f"{k}={v}")
            self._logger.info(" ".join(parts))
        except Exception:
            pass

    # Convenience aliases used throughout the project logs
    def info(self, event: str, **fields: object) -> None:
        self.line(event, **fields)

    def error(self, event: str, **fields: object) -> None:
        self.line(event, **fields)


# ----------------------------
# Geometry / ROI
# ----------------------------

def center_of(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def area_of(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def parse_roi_polygon(s: str) -> Optional[Polygon]:
    """
    Parse polygon like: "x1,y1 x2,y2 x3,y3 ..."
    """
    s = (s or "").strip()
    if not s:
        return None
    pts: Polygon = []
    for chunk in s.replace(";", " ").split():
        if "," not in chunk:
            continue
        xs, ys = chunk.split(",", 1)
        try:
            pts.append((int(xs), int(ys)))
        except Exception:
            continue
    if len(pts) < 3:
        return None
    return pts


def build_roi_mask(frame_shape: Tuple[int, int, int], roi: Optional[Polygon]) -> Optional[np.ndarray]:
    if roi is None:
        return None
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 255)
    return mask


def apply_roi_mask_bgr(frame_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return frame_bgr
    out = frame_bgr.copy()
    out[mask == 0] = 0
    return out


def in_roi_center(box: Box, roi_mask: Optional[np.ndarray]) -> bool:
    if roi_mask is None:
        return True
    cx, cy = center_of(box)
    x = int(max(0, min(roi_mask.shape[1] - 1, round(cx))))
    y = int(max(0, min(roi_mask.shape[0] - 1, round(cy))))
    return bool(roi_mask[y, x] != 0)


# ----------------------------
# Detector (Caffe SSD)
# ----------------------------

def load_names(names_path: Path) -> List[str]:
    return [ln.strip() for ln in names_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def parse_classes_csv(s: str) -> List[str]:
    return [c.strip() for c in (s or "").split(",") if c.strip()]


class SsdCaffe:
    def __init__(
        self,
        prototxt_path: Path,
        caffemodel_path: Path,
        names: Optional[List[str]] = None,
        inp_size: Tuple[int, int] = (300, 300),
        scale: float = 0.007843,
        mean: float = 127.5,
    ) -> None:
        if not prototxt_path.exists():
            raise FileNotFoundError(str(prototxt_path))
        if not caffemodel_path.exists():
            raise FileNotFoundError(str(caffemodel_path))
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        self.names = names
        self.inp_size = inp_size
        self.scale = scale
        self.mean = mean

    def predict(
        self,
        frame_bgr: np.ndarray,
        allow: Sequence[str],
        conf_thres: float,
        max_det: int,
    ) -> List[Detection]:
        allow_set = {c.lower().strip() for c in allow if c.strip()}
        h, w = frame_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=float(self.scale),
            size=self.inp_size,
            mean=(float(self.mean), float(self.mean), float(self.mean)),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()  # [1,1,N,7]

        dets: List[Detection] = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf < float(conf_thres):
                continue
            cls_id = int(out[0, 0, i, 1])
            if self.names and 0 <= cls_id < len(self.names):
                cls_name = self.names[cls_id]
            else:
                cls_name = str(cls_id)

            if allow_set and cls_name.lower() not in allow_set:
                continue

            x1 = int(max(0, min(w - 1, round(float(out[0, 0, i, 3]) * w))))
            y1 = int(max(0, min(h - 1, round(float(out[0, 0, i, 4]) * h))))
            x2 = int(max(0, min(w - 1, round(float(out[0, 0, i, 5]) * w))))
            y2 = int(max(0, min(h - 1, round(float(out[0, 0, i, 6]) * h))))
            if x2 <= x1 or y2 <= y1:
                continue

            dets.append(Detection(cls_name=cls_name, conf=conf, box=(x1, y1, x2, y2)))

        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets[: int(max_det)]


# ----------------------------
# Clip writer (OpenCV VideoWriter)
# ----------------------------

class ClipWriter:
    def __init__(self, out_path: Path, fps: float, frame_size: Tuple[int, int], fourcc: str = "mp4v") -> None:
        self.out_path = out_path
        self.fps = float(max(1.0, fps))
        self.frame_size = frame_size
        self.fourcc = fourcc

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        cc = cv2.VideoWriter_fourcc(*fourcc)
        self.vw = cv2.VideoWriter(str(out_path), cc, self.fps, self.frame_size)
        if not self.vw.isOpened():
            cc2 = cv2.VideoWriter_fourcc(*"avc1")
            self.vw = cv2.VideoWriter(str(out_path), cc2, self.fps, self.frame_size)
            if not self.vw.isOpened():
                raise RuntimeError("Failed to open VideoWriter (mp4v/avc1)")

    def write(self, frame_bgr: np.ndarray) -> None:
        self.vw.write(frame_bgr)

    def close(self) -> None:
        try:
            self.vw.release()
        except Exception:
            pass


# ----------------------------
# Confirmation buffer + clip meta
# ----------------------------

@dataclass
class RecentHit:
    t: float
    center: Tuple[float, float]
    cls_name: str
    conf: float
    box: Box


@dataclass
class ClipMeta:
    session: str
    device_id: str
    clip_path: str
    started_utc: str
    started_local: str
    ended_utc: Optional[str] = None
    ended_local: Optional[str] = None
    fps: float = 0.0
    size: Tuple[int, int] = (0, 0)  # w, h
    preroll_seconds: float = 0.0
    preroll_frames: int = 0
    clip_seconds: float = 0.0
    cooldown_seconds: float = 0.0
    allow_classes: List[str] = None  # type: ignore[assignment]
    roi: Optional[Polygon] = None
    trigger_target: Optional[Dict] = None
    frames_written: int = 0

    # detection stats during recording
    det_samples: int = 0
    det_conf_min: Optional[float] = None
    det_conf_max: Optional[float] = None
    det_conf_sum: float = 0.0
    det_class_counts: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.allow_classes is None:
            self.allow_classes = []
        if self.det_class_counts is None:
            self.det_class_counts = {}

    def update_det(self, det: Optional[Detection]) -> None:
        if det is None:
            return
        c = float(det.conf)
        self.det_samples += 1
        self.det_conf_sum += c
        self.det_conf_min = c if self.det_conf_min is None else min(self.det_conf_min, c)
        self.det_conf_max = c if self.det_conf_max is None else max(self.det_conf_max, c)
        self.det_class_counts[det.cls_name] = int(self.det_class_counts.get(det.cls_name, 0)) + 1

    def to_json_dict(self) -> Dict:
        mean = (self.det_conf_sum / self.det_samples) if self.det_samples > 0 else None
        return {
            "schema": "aicam_clip_meta_v1",
            "session": self.session,
            "device_id": self.device_id,
            "clip_path": self.clip_path,
            "started_utc": self.started_utc,
            "started_local": self.started_local,
            "ended_utc": self.ended_utc,
            "ended_local": self.ended_local,
            "fps": self.fps,
            "size": [int(self.size[0]), int(self.size[1])],
            "preroll_seconds": float(self.preroll_seconds),
            "preroll_frames": int(self.preroll_frames),
            "clip_seconds": float(self.clip_seconds),
            "cooldown_seconds": float(self.cooldown_seconds),
            "allow_classes": list(self.allow_classes),
            "roi": self.roi,
            "trigger_target": self.trigger_target,
            "frames_written": int(self.frames_written),
            "detection_samples": int(self.det_samples),
            "detection_conf_min": self.det_conf_min,
            "detection_conf_max": self.det_conf_max,
            "detection_conf_mean": mean,
            "detection_class_counts": dict(self.det_class_counts),
        }


def best_detection(dets: List[Detection]) -> Optional[Detection]:
    if not dets:
        return None
    return max(dets, key=lambda d: float(d.conf))


def update_target_from_detection(
    target: Optional[Target],
    det: Detection,
    now: float,
) -> Target:
    cx, cy = center_of(det.box)
    if target is None or target.classification != det.cls_name:
        return Target(
            id=f"{det.cls_name}",
            classification=det.cls_name,
            confidence=float(det.conf),
            position=(float(cx), float(cy), 0.0),
            velocity=(0.0, 0.0, 0.0),
            last_seen_mono=float(now),
            bbox=det.box,
        )

    dt = max(1e-6, float(now - target.last_seen_mono))
    vx = (float(cx) - float(target.position[0])) / dt
    vy = (float(cy) - float(target.position[1])) / dt
    return Target(
        id=target.id,
        classification=det.cls_name,
        confidence=float(det.conf),
        position=(float(cx), float(cy), 0.0),
        velocity=(vx, vy, 0.0),
        last_seen_mono=float(now),
        bbox=det.box,
    )


def decay_confidence(conf: float, dt: float, half_life_s: float) -> float:
    if half_life_s <= 1e-6:
        return 0.0
    return float(conf) * (0.5 ** (float(dt) / float(half_life_s)))


# ----------------------------
# Main
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AICAM Drone/Airborne recorder (Caffe SSD)")

    # Storage
    ap.add_argument("--mount", default=MOUNT, help="Preferred mount base (if present)")
    ap.add_argument("--base-subdir", default=BASE_SUBDIR, help="Subdirectory under mount/fallback")
    ap.add_argument("--fallback-dir", default=FALLBACK_DIR, help="Fallback base directory")
    ap.add_argument("--device-id", default=DEVICE_ID, help="Device/camera identifier for logs and metadata")
    # Source
    ap.add_argument("--source", default="picamera2", help="Source (required): picamera2 (Picamera2/libcamera). OpenCV VideoCapture sources are disabled in this build.")
    ap.add_argument("--preview", action="store_true", help="Show preview window")

    # Capture settings (Picamera2 only)
    ap.add_argument("--width", type=int, default=640, help="Requested capture width")
    ap.add_argument("--height", type=int, default=480, help="Requested capture height")
    ap.add_argument("--fps", type=float, default=15.0, help="Requested capture FPS")

    # Model files
    ap.add_argument("--model-dir", default=MODEL_DIR, help="Base directory for model files")
    ap.add_argument("--cfg", default=PROTOTXT, help="Caffe prototxt (deploy.prototxt)")
    ap.add_argument("--model", default=CAFFEMODEL, help="Caffe model (caffemodel)")
    ap.add_argument("--names", default="", help="Optional class names file (one per line)")

    # Detection controls
    ap.add_argument("--classes", default="aeroplane,bird", help="Allowed classes CSV (empty => all)")
    ap.add_argument("--conf", type=float, default=0.60, help="Detection confidence threshold")
    ap.add_argument("--max-det", type=int, default=25, help="Max detections per frame")
    ap.add_argument("--detect-every", type=int, default=1, help="Run detection every N frames (standby/track)")
    ap.add_argument("--detect-every-recording", type=int, default=3, help="Run detection every N frames while recording")

    # Inference pacing / load shedding
    # Default 0 => preserve legacy behavior unless explicitly enabled.
    ap.add_argument("--max-infer-fps", type=float, default=0.0, help="Hard cap on inference FPS (0 disables)")
    ap.add_argument("--adaptive-skip", action="store_true", help="Adaptively increase detect-every based on inference latency EMA")
    ap.add_argument("--max-skip-factor", type=int, default=6, help="Max multiplier applied to detect-every when using --adaptive-skip")
    ap.add_argument("--infer-ema-alpha", type=float, default=0.2, help="EMA alpha for inference latency (0-1)")

    # Motion gating (optional)
    ap.add_argument("--motion-threshold", type=float, default=0.0, help="Mean abs frame diff (0-255) above which detection is skipped (0 disables)")
    ap.add_argument("--motion-downscale", default="160x120", help="Downscale size for motion estimation, e.g. 160x120")
    ap.add_argument("--motion-roi-scale", type=float, default=0.0, help="Center ROI scale for motion metric (0 or >=1 = full frame; e.g. 0.5 uses center 50%)")
    ap.add_argument("--motion-threshold-exit", type=float, default=0.0, help="Exit threshold for motion gating (0 => auto from motion-threshold)")
    ap.add_argument("--motion-high-frames", type=int, default=2, help="Consecutive frames above enter threshold required to enable motion gating")
    ap.add_argument("--motion-low-frames", type=int, default=2, help="Consecutive frames below exit threshold required to disable motion gating")
    ap.add_argument("--motion-settle-frames", type=int, default=4, help="Extra frames to keep detection suppressed after motion calms")

    # ROI & size gating
    ap.add_argument("--roi", default="", help='ROI polygon: "x1,y1 x2,y2 x3,y3 ..."')
    ap.add_argument("--mask-outside-roi", action="store_true", help="Mask outside ROI before inference")
    ap.add_argument("--min-area", type=int, default=0, help="Min bbox area in pixels")
    ap.add_argument("--max-area", type=int, default=10_000_000, help="Max bbox area in pixels")

    # Confirmation / motion gating
    ap.add_argument("--confirm-hits", type=int, default=3, help="Hits needed within window to acquire")
    ap.add_argument("--confirm-window-s", type=float, default=1.5, help="Time window to accumulate hits")
    ap.add_argument("--min-speed", type=float, default=0.0, help="Min center speed in px/s (requires >=2 hits)")
    ap.add_argument("--max-speed", type=float, default=1e9, help="Max center speed in px/s")

    # Target/state behavior
    ap.add_argument("--lost-sec", type=float, default=2.0, help="Seconds since last seen before LOST_TARGET")
    ap.add_argument("--conf-half-life-s", type=float, default=1.0, help="Confidence decay half-life while unseen")
    ap.add_argument("--reacquire-sec", type=float, default=5.0, help="Max seconds in LOST_TARGET before STANDBY")

    # Recording
    ap.add_argument("--clip-seconds", type=float, default=10.0, help="Clip duration seconds")
    ap.add_argument("--min-clip-sec", type=float, default=2.0, help="Minimum clip seconds to keep; shorter clips are discarded")
    ap.add_argument("--preroll-seconds", type=float, default=2.0, help="Seconds saved before trigger")
    ap.add_argument("--cooldown-seconds", type=float, default=5.0, help="Minimum seconds between clip starts")
    ap.add_argument("--fourcc", default="mp4v", help="FourCC codec (mp4v recommended)")
    ap.add_argument("--log-level", default="INFO", help="Log level (DEBUG/INFO/WARNING/ERROR)")
    return ap



def discard_clip_files(clip_path: Path) -> None:
    """Remove clip mp4 and sibling json if present."""
    try:
        clip_path.unlink(missing_ok=True)
    except TypeError:
        if clip_path.exists():
            clip_path.unlink()
    json_path = clip_path.with_suffix(".json")
    try:
        json_path.unlink(missing_ok=True)
    except TypeError:
        if json_path.exists():
            json_path.unlink()


def clip_duration_seconds(meta: ClipMeta) -> float:
    # Best-effort duration based on frames and fps.
    if meta.fps <= 1e-6:
        return 0.0
    return float(meta.frames_written) / float(meta.fps)


def finalize_clip(meta: ClipMeta, min_clip_sec: float, tmp_dir: Path, logger: logging.Logger, events_text: TextEvents, events_jsonl: JsonlEvents) -> None:
    """
    Write per-clip JSON. If clip is shorter than min_clip_sec, discard files and log DISCARD.
    """
    dur_s = clip_duration_seconds(meta)
    clip_path = Path(meta.clip_path)

    # If too short, discard and log 
    if dur_s < float(min_clip_sec):
        discard_clip_files(clip_path)
        logger.info("Discarded short clip=%s dur=%.2f min=%.2f", clip_path.name, dur_s, float(min_clip_sec))
        events_text.info("DISCARD", session=meta.session, device_id=meta.device_id, clip=clip_path.name, dur=f"{dur_s:.2f}", reason="too_short", min=f"{float(min_clip_sec):.2f}")
        events_jsonl.emit({
            "type": "discard",
            "session": meta.session,
            "device_id": meta.device_id,
            "ts_utc": ts_utc_iso(),
            "ts_local": ts_local_iso(),
            "clip": str(clip_path),
            "dur_s": dur_s,
            "min_clip_sec": float(min_clip_sec),
            "reason": "too_short",
        })
        return

    # Otherwise keep and write JSON metadata
    try:
        write_clip_json(meta, tmp_dir)
    except Exception:
        logger.exception("Failed to write clip JSON")

def write_clip_json(meta: ClipMeta, tmp_dir: Path) -> None:
    clip_path = Path(meta.clip_path)
    json_path = clip_path.with_suffix(".json")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / (json_path.name + ".tmp")
    tmp_path.write_text(json.dumps(meta.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(json_path)


def main() -> int:
    args = build_arg_parser().parse_args()

    # Session identifier for cross-log correlation 
    session = ts_utc_compact()
    device_id = str(args.device_id)

    # Allow --model-dir to override default cfg/model locations
    try:
        md = Path(str(args.model_dir))
        if str(args.cfg) == PROTOTXT:
            args.cfg = str(md / Path(PROTOTXT).name)
        if str(args.model) == CAFFEMODEL:
            args.model = str(md / Path(CAFFEMODEL).name)
    except Exception:
        pass


    base = pick_base_dir(args.mount, args.base_subdir, args.fallback_dir)
    clips_dir = base / "clips"
    logs_dir = base / "logs"
    tmp_dir = base / TMP_SUBDIR

    logger = setup_rotating_logger("aicam_drone", logs_dir / "aicam_drone.log", args.log_level)
    events_jsonl = JsonlEvents(logs_dir / "events.jsonl")
    events_text = TextEvents(logs_dir / EVENT_LOG_NAME)

    # Load detector
    names = load_names(Path(args.names)) if str(args.names).strip() else None
    detector = SsdCaffe(prototxt_path=Path(args.cfg), caffemodel_path=Path(args.model), names=names)
    allow = parse_classes_csv(str(args.classes))

    # Open video source (Picamera2 only)
    src = str(args.source).strip().lower()
    if src not in {"picamera2", "picam", "libcamera"}:
        logger.error("Invalid --source=%s (this build only supports Picamera2)", args.source)
        events_jsonl.emit({"type": "error", "ts_utc": ts_utc_iso(), "ts_local": ts_local_iso(), "where": "open_source", "source": str(args.source), "hint": "Only picamera2 is supported"})
        events_text.error("ERROR", where="open_source", source=str(args.source), hint="Only picamera2 is supported")
        return 2

    picam2 = None

    def _read_frame() -> Tuple[bool, Optional[np.ndarray]]:
        """Picamera2 frame reader."""
        nonlocal picam2
        if picam2 is None:
            return False, None
        try:
            rgb = picam2.capture_array()  # RGB
            if rgb is None:
                return False, None
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return True, bgr
        except Exception:
            return False, None

    try:
        from picamera2 import Picamera2

        picam2 = Picamera2()
        req_w, req_h = int(max(16, args.width)), int(max(16, args.height))
        fps = float(args.fps) if float(args.fps) > 0 else 15.0
        config = picam2.create_video_configuration(main={"size": (req_w, req_h), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()

        # Best-effort FPS control (libcamera control in microseconds)
        try:
            frame_us = int(1_000_000 / fps)
            picam2.set_controls({"FrameDurationLimits": (frame_us, frame_us)})
        except Exception:
            pass

    except Exception as e:
        logger.exception("Failed to initialize Picamera2")
        events_jsonl.emit({"type": "error", "ts_utc": ts_utc_iso(), "ts_local": ts_local_iso(), "where": "open_source", "source": str(args.source), "error": str(e)})
        events_text.error("ERROR", where="open_source", source=str(args.source), error=str(e))
        return 2

    ok, frame0 = _read_frame()
    if not ok or frame0 is None:
        logger.error("Failed to read first frame")
        events_jsonl.emit({"type": "error", "ts_utc": ts_utc_iso(), "ts_local": ts_local_iso(), "where": "first_frame"})
        events_text.error("ERROR", where="first_frame")
        return 2

    h, w = frame0.shape[:2]
    roi_poly = parse_roi_polygon(str(args.roi)) if str(args.roi).strip() else None
    roi_mask = build_roi_mask(frame0.shape, roi_poly)

    # Buffers
    preroll_frames = max(0, int(round(float(args.preroll_seconds) * fps)))
    preroll: Deque[np.ndarray] = collections.deque(maxlen=preroll_frames)
    hits: Deque[RecentHit] = collections.deque()

    # State
    state = State.INIT
    target: Optional[Target] = None
    state_enter_mono = time.monotonic()

    # Clip state
    writer: Optional[ClipWriter] = None
    clip_end_t: float = 0.0
    last_trigger_t: float = -1e9
    last_seen_any_mono: float = -1e9
    active_meta: Optional[ClipMeta] = None

    stop_flag = {"stop": False}

    def set_state(new_state: State, reason: str) -> None:
        nonlocal state, state_enter_mono
        if new_state == state:
            return
        old = state
        state = new_state
        state_enter_mono = time.monotonic()
        logger.info("STATE %s -> %s (%s)", old.value, new_state.value, reason)
        events_jsonl.emit({
            "type": "state",
            "ts_utc": ts_utc_iso(),
            "ts_local": ts_local_iso(),
            "from": old.value,
            "to": new_state.value,
            "reason": reason,
            "target": (dataclasses.asdict(target) if target else None),
        })
        events_text.info("STATE", session=session, device_id=device_id, frm=old.value, to=new_state.value, reason=reason)

    def _sig_handler(_sig, _frame) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    set_state(State.STANDBY, "init")
    logger.info("Started source=%s fps=%.2f size=%dx%d preroll_frames=%d base=%s", src, fps, w, h, preroll_frames, str(base))
    events_jsonl.emit({
        "type": "start",
        "session": session,
        "device_id": device_id,
        "ts_utc": ts_utc_iso(),
        "ts_local": ts_local_iso(),
        "source": str(src),
        "fps": fps,
        "size": [w, h],
        "allow": allow,
        "roi": roi_poly,
    })
    events_text.info("START", session=session, device_id=device_id, source=str(src), fps=f"{fps:.2f}", size=f"{w}x{h}", base=str(base))

    frame_idx = 0
    frame = frame0

    # Motion estimation state (optional; cheap frame-to-frame mean abs diff)
    prev_motion_gray: Optional[np.ndarray] = None
    motion_value = 0.0
    motion_high = False
    motion_high_count = 0
    motion_low_count = 0
    motion_settle_remaining = 0

    # Inference pacing / load shedding state
    last_infer_mono = 0.0
    infer_ema_s: Optional[float] = None

    # Parse motion downscale argument once
    try:
        md = str(args.motion_downscale).lower().replace(" ", "")
        mw_s, mh_s = md.split("x", 1)
        motion_w, motion_h = int(mw_s), int(mh_s)
        motion_w = max(32, min(motion_w, w))
        motion_h = max(24, min(motion_h, h))
    except Exception:
        motion_w, motion_h = min(160, w), min(120, h)

    # One-time startup log for motion gate parameters (field tuning aid)
    try:
        enter_th0 = float(args.motion_threshold)
        exit_th0 = float(args.motion_threshold_exit)
        if enter_th0 > 0.0:
            if exit_th0 <= 0.0:
                exit_th0 = enter_th0 * 0.7
            if exit_th0 > enter_th0:
                exit_th0 = enter_th0
        else:
            exit_th0 = 0.0
        logger.info(
            "MotionGate cfg: enabled=%s enter=%.2f exit=%.2f high_frames=%d low_frames=%d settle_frames=%d roi_scale=%.2f downscale=%dx%d",
            "yes" if enter_th0 > 0.0 else "no",
            enter_th0,
            exit_th0,
            max(1, int(args.motion_high_frames)),
            max(1, int(args.motion_low_frames)),
            max(0, int(args.motion_settle_frames)),
            float(args.motion_roi_scale),
            int(motion_w),
            int(motion_h),
        )
    except Exception:
        pass

    try:
        while not stop_flag["stop"]:
            frame_idx += 1
            now = time.monotonic()

            # --- Motion gating (optional) ---
            # Computes mean abs pixel diff between consecutive downscaled grayscale frames.
            # Includes hysteresis, debounce (consecutive frames), and a settle period.
            motion_value = 0.0
            enter_th = float(args.motion_threshold)
            if enter_th > 0.0:
                exit_th = float(args.motion_threshold_exit)
                if exit_th <= 0.0:
                    exit_th = enter_th * 0.7
                if exit_th > enter_th:
                    exit_th = enter_th

                try:
                    gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if motion_w != w or motion_h != h:
                        gray_small = cv2.resize(gray_small, (motion_w, motion_h), interpolation=cv2.INTER_AREA)

                    # Optional center ROI crop to reduce edge-dominant motion and exposure flicker
                    roi_scale = float(args.motion_roi_scale)
                    if 0.0 < roi_scale < 1.0:
                        hh, ww = gray_small.shape[:2]
                        rw = max(1, int(ww * roi_scale))
                        rh = max(1, int(hh * roi_scale))
                        x0 = (ww - rw) // 2
                        y0 = (hh - rh) // 2
                        gray_small = gray_small[y0:y0 + rh, x0:x0 + rw]

                    if prev_motion_gray is None or prev_motion_gray.shape != gray_small.shape:
                        motion_value = 0.0
                    else:
                        motion_value = float(cv2.mean(cv2.absdiff(gray_small, prev_motion_gray))[0])
                    prev_motion_gray = gray_small
                except Exception:
                    motion_value = 0.0

                high_needed = max(1, int(args.motion_high_frames))
                low_needed = max(1, int(args.motion_low_frames))
                settle_frames = max(0, int(args.motion_settle_frames))

                if motion_high:
                    if motion_value < exit_th:
                        motion_low_count += 1
                    else:
                        motion_low_count = 0

                    if motion_low_count >= low_needed:
                        motion_high = False
                        motion_low_count = 0
                        motion_high_count = 0
                        motion_settle_remaining = settle_frames
                else:
                    # settle countdown after leaving high motion
                    if motion_settle_remaining > 0:
                        # If motion rises again, require debounce to re-enter.
                        if motion_value > enter_th:
                            motion_high_count += 1
                            if motion_high_count >= high_needed:
                                motion_high = True
                                motion_high_count = 0
                                motion_settle_remaining = 0
                        else:
                            motion_high_count = 0
                            motion_settle_remaining -= 1
                    else:
                        if motion_value > enter_th:
                            motion_high_count += 1
                        else:
                            motion_high_count = 0

                        if motion_high_count >= high_needed:
                            motion_high = True
                            motion_high_count = 0
                            motion_low_count = 0

            motion_gate_active = bool(motion_high) or (motion_settle_remaining > 0)
            if motion_gate_active:
                hits.clear()

            # preroll
            if preroll_frames > 0:
                preroll.append(frame.copy())

            # recording write
            if writer is not None:
                writer.write(frame)
                if active_meta is not None:
                    active_meta.frames_written += 1
                if now >= clip_end_t:
                    writer.close()
                    writer = None

                    # finalize clip meta + write JSON
                    if active_meta is not None:
                        active_meta.ended_utc = ts_utc_iso()
                        active_meta.ended_local = ts_local_iso()
                        finalize_clip(active_meta, float(args.min_clip_sec), tmp_dir, logger, events_text, events_jsonl)
                        events_text.info("CLIP_COMPLETE", session=active_meta.session, device_id=active_meta.device_id, clip=Path(active_meta.clip_path).name)
                        events_jsonl.emit({
                            "type": "clip_complete",
                            "session": session,
                            "device_id": device_id,
                            "ts_utc": ts_utc_iso(),
                            "ts_local": ts_local_iso(),
                            "clip": active_meta.clip_path,
                        })
                        active_meta = None

                    hits.clear()
                    preroll.clear()

            # detection cadence depends on recording state
            dets: List[Detection] = []
            det_best: Optional[Detection] = None

            base_detect_every = int(args.detect_every_recording if writer is not None else args.detect_every)
            if base_detect_every < 1:
                base_detect_every = 1

            # Optional adaptive skip based on inference latency EMA
            skip_factor = 1
            if bool(args.adaptive_skip) and infer_ema_s is not None:
                # Increase skipping when inference consumes too much of the allowed infer budget.
                max_infer_fps = float(args.max_infer_fps)
                if max_infer_fps > 0.0:
                    budget = max(1e-6, 1.0 / max_infer_fps)
                    util = float(infer_ema_s) / budget
                    if util > 0.8:
                        skip_factor = min(int(args.max_skip_factor), max(1, int(util / 0.8)))

            detect_every = int(base_detect_every) * int(skip_factor)
            if detect_every < 1:
                detect_every = 1

            # Inference pacing (hard cap)
            max_infer_fps = float(args.max_infer_fps)
            min_infer_period = (1.0 / max_infer_fps) if max_infer_fps > 0.0 else 0.0
            cadence_ok = (frame_idx % detect_every == 0)
            fps_ok = (min_infer_period <= 0.0) or ((now - last_infer_mono) >= min_infer_period)

            if (not motion_gate_active) and cadence_ok and fps_ok:
                inp = apply_roi_mask_bgr(frame, roi_mask) if args.mask_outside_roi else frame

                t_infer0 = time.monotonic()
                dets = detector.predict(
                    frame_bgr=inp,
                    allow=allow,
                    conf_thres=float(args.conf),
                    max_det=int(args.max_det),
                )
                t_infer1 = time.monotonic()

                last_infer_mono = float(now)

                infer_dt = max(0.0, float(t_infer1 - t_infer0))
                alpha = float(args.infer_ema_alpha)
                if alpha < 0.0:
                    alpha = 0.0
                if alpha > 1.0:
                    alpha = 1.0
                if infer_ema_s is None:
                    infer_ema_s = infer_dt
                else:
                    infer_ema_s = (alpha * infer_dt) + ((1.0 - alpha) * float(infer_ema_s))

                # Filter by ROI center, area
                filtered: List[Detection] = []
                for d in dets:
                    if not in_roi_center(d.box, roi_mask):
                        continue
                    a = area_of(d.box)
                    if a < int(args.min_area) or a > int(args.max_area):
                        continue
                    filtered.append(d)
                dets = filtered
                det_best = best_detection(dets)

                # Update clip stats if recording
                if writer is not None and active_meta is not None:
                    active_meta.update_det(det_best)

                # Update target hypothesis for gating
                if det_best is not None:
                    last_seen_any_mono = now
                    target = update_target_from_detection(target, det_best, now)
                else:
                    if target is not None:
                        dt_unseen = max(0.0, now - float(target.last_seen_mono))
                        target.confidence = decay_confidence(float(target.confidence), dt_unseen, float(args.conf_half_life_s))

            # State machine (gating only; recording does not change state)
            if state == State.STANDBY:
                target = None
                hits.clear()
                if det_best is not None:
                    set_state(State.ACQUIRE_TARGET, "first_detection")

            elif state == State.ACQUIRE_TARGET:
                # Drop old hits
                window_s = float(args.confirm_window_s)
                while hits and (now - hits[0].t) > window_s:
                    hits.popleft()

                if det_best is not None:
                    hits.append(RecentHit(t=now, center=center_of(det_best.box), cls_name=det_best.cls_name, conf=det_best.conf, box=det_best.box))

                if det_best is None and hits and (now - hits[-1].t) > window_s:
                    set_state(State.STANDBY, "acquire_timeout")

                if len(hits) >= int(args.confirm_hits):
                    speed_ok = True
                    min_speed = float(args.min_speed)
                    max_speed = float(args.max_speed)
                    if min_speed > 0.0 or max_speed < 1e9:
                        if len(hits) < 2:
                            speed_ok = False
                        else:
                            h1, h2 = hits[-2], hits[-1]
                            dt = max(1e-6, h2.t - h1.t)
                            dx = h2.center[0] - h1.center[0]
                            dy = h2.center[1] - h1.center[1]
                            spd = float((dx * dx + dy * dy) ** 0.5) / dt
                            speed_ok = (spd >= min_speed) and (spd <= max_speed)

                    if speed_ok:
                        set_state(State.TRACK, "confirmed")
                    else:
                        while len(hits) > int(args.confirm_hits):
                            hits.popleft()

            elif state == State.TRACK:
                if target is None or (now - float(target.last_seen_mono)) >= float(args.lost_sec):
                    set_state(State.LOST_TARGET, "lost_target")
                else:
                    # Trigger clip only if not already recording
                    if writer is None and (now - last_trigger_t) >= float(args.cooldown_seconds):
                        # Enforce minimum free space before triggering a clip.
                        free_gb = free_space_gb(clips_dir)
                        if free_gb < float(MIN_FREE_GB):
                            last_trigger_t = now  # throttle repeated checks/logs
                            logger.warning(
                                "Skip trigger: low disk free_gb=%.2f min_free_gb=%.2f",
                                free_gb,
                                float(MIN_FREE_GB),
                            )
                            events_jsonl.emit({
                                "type": "skip",
                                "reason": "low_disk",
                                "session": session,
                                "device_id": device_id,
                                "ts_utc": ts_utc_iso(),
                                "ts_local": ts_local_iso(),
                                "free_gb": round(float(free_gb), 3),
                                "min_free_gb": float(MIN_FREE_GB),
                            })
                            events_text.info(
                                "SKIP",
                                reason="low_disk",
                                free_gb=f"{free_gb:.2f}",
                                min_free_gb=f"{float(MIN_FREE_GB):.2f}",
                            )
                            # Stay in TRACK, do not open writer.
                        else:
                            last_trigger_t = now

                            clip_name = f"aicam_{ts_utc_compact()}_airborne.mp4"
                            out_path = clips_dir / clip_name

                            writer = ClipWriter(out_path, fps=fps, frame_size=(w, h), fourcc=str(args.fourcc))

                            # Per-clip metadata
                            active_meta = ClipMeta(
                                session=session,
                                device_id=device_id,
                                clip_path=str(out_path),
                                started_utc=ts_utc_iso(),
                                started_local=ts_local_iso(),
                                fps=float(fps),
                                size=(int(w), int(h)),
                                preroll_seconds=float(args.preroll_seconds),
                                preroll_frames=int(preroll_frames),
                                clip_seconds=float(args.clip_seconds),
                                cooldown_seconds=float(args.cooldown_seconds),
                                allow_classes=list(allow),
                                roi=roi_poly,
                                trigger_target=(dataclasses.asdict(target) if target else None),
                                frames_written=0,
                            )

                            # Write preroll
                            for fr in preroll:
                                writer.write(fr)
                                active_meta.frames_written += 1

                            clip_end_t = now + float(args.clip_seconds)

                            logger.info("Triggered clip=%s cls=%s conf=%.3f", out_path.name, (target.classification if target else None), (target.confidence if target else -1.0))
                            events_jsonl.emit({
                                "type": "trigger",
                                "session": session,
                                "device_id": device_id,
                                "ts_utc": ts_utc_iso(),
                                "ts_local": ts_local_iso(),
                                "clip": str(out_path),
                                "fps": fps,
                                "size": [w, h],
                                "roi": roi_poly,
                                "target": (dataclasses.asdict(target) if target else None),
                            })
                            events_text.info("CLIP", clip=out_path.name, cls=(target.classification if target else None), conf=(round(float(target.confidence), 3) if target else None))

            elif state == State.LOST_TARGET:
                if det_best is not None:
                    set_state(State.TRACK, "reacquired")
                else:
                    if (now - last_seen_any_mono) >= float(args.reacquire_sec):
                        set_state(State.STANDBY, "reacquire_expired")

            elif state == State.FAILSAFE:
                # Minimal, in-scope failsafe: stop recording and shutdown
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                    writer = None
                set_state(State.SHUTDOWN, "failsafe")

            # Preview (optional)
            if args.preview:
                disp = frame.copy()
                if det_best is not None:
                    x1, y1, x2, y2 = det_best.box
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        disp,
                        f"{det_best.cls_name} {det_best.conf:.2f} {state.value}",
                        (max(0, x1), max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if roi_poly:
                    cv2.polylines(disp, [np.array(roi_poly, dtype=np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)

                cv2.imshow("AICAM_Drone", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            ok, frame = _read_frame()
            if not ok or frame is None:
                logger.warning("Frame read failed; stopping")
                events_jsonl.emit({"type": "error", "ts_utc": ts_utc_iso(), "ts_local": ts_local_iso(), "where": "read_frame"})
                events_text.error("ERROR", where="read_frame")
                break

    except Exception:
        logger.exception("Unhandled exception; entering FAILSAFE")
        events_text.error("ERROR", where="exception")
        events_jsonl.emit({"type": "error", "ts_utc": ts_utc_iso(), "ts_local": ts_local_iso(), "where": "exception"})
        set_state(State.FAILSAFE, "exception")

    # Cleanup
    set_state(State.SHUTDOWN, "stop")

    try:
        if picam2 is not None:
            picam2.stop()
    except Exception:
        pass
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass
    if active_meta is not None:
        # best-effort meta write
        active_meta.ended_utc = ts_utc_iso()
        active_meta.ended_local = ts_local_iso()
        try:
            finalize_clip(active_meta, float(args.min_clip_sec), tmp_dir, logger, events_text, events_jsonl)
        except Exception:
            pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    logger.info("Stopped")
    events_jsonl.emit({"type": "stop", "session": session, "device_id": device_id, "ts_utc": ts_utc_iso(), "ts_local": ts_local_iso()})
    events_text.info("STOP", session=session, device_id=device_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
