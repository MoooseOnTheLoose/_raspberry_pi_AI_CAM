#!/usr/bin/env python3
"""
AICAM_Human.py (Raspberry Pi OS)

Human presence recorder:
- Runs MobileNet-SSD inference on the lores stream (Caffe model).
- Starts recording ONLY when target human class is detected with sufficient confidence.
- Stops after target absence hold time (post-roll).
- Writes MP4 clips + per-clip JSON metadata + a simple events.log.

Requirements:
  sudo apt install -y ffmpeg python3-picamera2 python3-opencv

Model files (MobileNet-SSD Caffe):
  deploy.prototxt
  mobilenet_iter_73000.caffemodel

Place model files in MODEL_DIR below.

Notes:
- This is designed for an offline, single-node setup.
- Pre-roll buffering uses Picamera2 CircularOutput (encoded ring buffer) for stability on Raspberry Pi OS.
"""

import os
import re
import sys
import json
import time
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, CircularOutput
from picamera2 import MappedArray


# -----------------------------
# Storage
# -----------------------------
MOUNT = "/media/usr/deskView"
BASE_SUBDIR = "humancam"
TMP_SUBDIR = "tmp"  # internal temp files (raw h264, concat lists)
FALLBACK_DIR = "/home/usr/humancam"

EVENT_LOG_NAME = "events.log"
MIN_FREE_GB = 10.0  # adjust to taste

# -----------------------------
# Recording parameters
# -----------------------------
FPS = 15
BITRATE = 6000000  # H.264 encoder bitrate (bits/sec)

MAIN_SIZE = (1920, 1080)
LORES_SIZE = (640, 360)

# Target detection / gating
HUMAN_MIN_CONF = 0.60
INFER_EVERY_N_FRAMES = 3  # inference cadence while running main loop
DETECT_INTERVAL_S = INFER_EVERY_N_FRAMES / float(FPS)  # seconds between inferences
START_FRAMES = 3          # target must be present this many inference ticks to start recording
STOP_FRAMES = 10          # target must be absent this many inference ticks to stop recording
COOLDOWN_SEC = 5.0        # minimum time between clip starts
PRE_ROLL_SEC = 3.0       # seconds of video to prepend before target trigger (requires circular buffer)
MIN_CLIP_SEC = 2.0        # discard clips shorter than this
MAX_CLIP_SEC = 60.0       # split if longer than this (keeps files manageable)

# Overlay
HEADER_H = 76
COLOR_TEXT = (235, 235, 235)
COLOR_DIM = (160, 160, 160)
COLOR_OK = (80, 220, 80)
COLOR_WARN = (40, 180, 255)

# -----------------------------
# AI model (Caffe MobileNet-SSD)
# -----------------------------
MODEL_DIR = "/home/usr/models"  # <-- change if needed
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")

CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

TARGET_CLASSES = {"person"}  # MobileNet-SSD label for humans

# -----------------------------
# ROI / exclusion zones (recommended)
# -----------------------------
# The ROI is evaluated on the LORES frame (LORES_SIZE). The trigger uses the center point of
# the best detection bounding box.
#
# - If ROI_INCLUDE_POLY_NORM is set, detections must fall inside it to count.
# - Any ROI_EXCLUDE_POLYS_NORM always veto (even if inside include).
#
# Coordinates are normalized (0.0-1.0) relative to LORES frame width/height.
#
# Example include polygon (a centered rectangle):
#   ROI_INCLUDE_POLY_NORM = [(0.10, 0.20), (0.90, 0.20), (0.90, 0.95), (0.10, 0.95)]
# Example exclude polygon (ignore a sidewalk strip):
#   ROI_EXCLUDE_POLYS_NORM = [[(0.0, 0.0), (1.0, 0.0), (1.0, 0.12), (0.0, 0.12)]]
ROI_ENABLE = True
ROI_INCLUDE_POLY_NORM = None  # set to a list of (x_norm,y_norm) to enable an include polygon
ROI_EXCLUDE_POLYS_NORM = []   # list of polygons, each polygon is list of (x_norm,y_norm)

# -----------------------------
# Helpers
# -----------------------------
def fail_if_missing(path: str) -> None:
    if not os.path.exists(path):
        print(f"[FATAL] Missing required file: {path}", file=sys.stderr)
        sys.exit(2)


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _norm_poly_to_px(poly_norm, w: int, h: int):
    if not poly_norm:
        return None
    pts=[]
    for x,y in poly_norm:
        x=int(round(float(x)* (w-1)))
        y=int(round(float(y)* (h-1)))
        pts.append((x,y))
    return pts


def _point_in_poly(pt, poly_pts) -> bool:
    """poly_pts: list[(x,y)]"""
    if not poly_pts or len(poly_pts) < 3:
        return False
    arr = np.array(poly_pts, dtype=np.int32)
    # cv2.pointPolygonTest expects contour shape (N,1,2)
    arr = arr.reshape((-1,1,2))
    return cv2.pointPolygonTest(arr, pt, False) >= 0


def _roi_accept_bbox_center(bbox_lores, lo_w: int, lo_h: int) -> bool:
    """Return True if bbox center passes include/exclude ROI rules."""
    if not ROI_ENABLE:
        return True
    if bbox_lores is None:
        return False
    x1,y1,x2,y2 = bbox_lores
    cx = int(round((x1+x2)/2.0))
    cy = int(round((y1+y2)/2.0))

    inc = _norm_poly_to_px(ROI_INCLUDE_POLY_NORM, lo_w, lo_h)
    if inc:
        if not _point_in_poly((cx,cy), inc):
            return False

    for poly_norm in (ROI_EXCLUDE_POLYS_NORM or []):
        exc = _norm_poly_to_px(poly_norm, lo_w, lo_h)
        if exc and _point_in_poly((cx,cy), exc):
            return False

    return True


def _roi_polys_for_main():
    """Precompute ROI polygons in MAIN pixel coords for overlay."""
    lo_w, lo_h = LORES_SIZE
    main_w, main_h = MAIN_SIZE
    sx = main_w / float(lo_w)
    sy = main_h / float(lo_h)

    def scale(poly_norm):
        pts = _norm_poly_to_px(poly_norm, lo_w, lo_h)
        if not pts:
            return None
        return [(int(round(x*sx)), int(round(y*sy))) for x,y in pts]

    inc = scale(ROI_INCLUDE_POLY_NORM) if ROI_INCLUDE_POLY_NORM else None
    excs=[]
    for poly_norm in (ROI_EXCLUDE_POLYS_NORM or []):
        s = scale(poly_norm)
        if s:
            excs.append(s)
    return inc, excs


def pick_out_dir() -> str:
    """
    Prefer removable mount if present; else fallback to local.
    Creates daily folder: <base>/clips/YYYY/MM/DD
    """
    base = None
    mount_base = os.path.join(MOUNT, BASE_SUBDIR)
    if os.path.isdir(MOUNT):
        base = mount_base
    else:
        base = FALLBACK_DIR

    day_dir = datetime.now().strftime("%Y/%m/%d")
    out_dir = os.path.join(base, "clips", day_dir)
    ensure_dir(out_dir)

    meta_dir = os.path.join(base, "meta", day_dir)
    ensure_dir(meta_dir)
    tmp_dir = os.path.join(base, TMP_SUBDIR)
    ensure_dir(tmp_dir)

    return base


def append_event(base: str, line: str) -> None:
    path = os.path.join(base, EVENT_LOG_NAME)
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def draw_label(img, x1, y1, text, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    y = max(HEADER_H + th + 8, y1)
    x = max(8, x1)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 14, y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def ffmpeg_remux_h264_to_mp4(h264_path: str, mp4_path: str, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-fflags", "+genpts",
        "-r", str(fps),
        "-i", h264_path,
        "-c", "copy",
        "-movflags", "+faststart",
        mp4_path,
    ]
    subprocess.run(cmd, check=True)


def ffmpeg_concat_mp4(parts: List[str], out_mp4: str, work_dir: str) -> None:
    lst_path = os.path.join(work_dir, f"concat_{int(time.time()*1000)}.txt")
    with open(lst_path, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", lst_path,
            "-c", "copy",
            "-movflags", "+faststart",
            out_mp4,
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(lst_path)
        except OSError:
            pass

@dataclass
class ClipMeta:
    # Required (non-default) fields must come before any default-valued fields
    clip_name: str
    clip_path: str
    meta_path: str
    start_local: str
    start_utc: str

    # Optional / default fields
    tmp_main_h264: Optional[str] = None
    tmp_pre_h264: Optional[str] = None
    pre_roll_s: float = 0.0
    end_local: Optional[str] = None
    end_utc: Optional[str] = None
    duration_s: Optional[float] = None
    fps: int = FPS
    main_size: Tuple[int, int] = MAIN_SIZE
    lores_size: Tuple[int, int] = LORES_SIZE
    model: str = "MobileNet-SSD (Caffe)"
    target_classes: List[str] = None
    trigger_class: Optional[str] = None
    min_conf: float = HUMAN_MIN_CONF
    conf_min: Optional[float] = None
    conf_mean: Optional[float] = None
    conf_max: Optional[float] = None
    detections: int = 0
    frames_inferred: int = 0
    notes: Optional[str] = None


def write_clip_meta(meta: ClipMeta) -> None:
    ensure_dir(os.path.dirname(meta.meta_path))
    with open(meta.meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)



# -----------------------------
# Target detection + recording helpers
# -----------------------------
def detect_targets_on_lores(net, lo_bgr: np.ndarray) -> Tuple[bool, Optional[str], float, Optional[Tuple[int,int,int,int]]]:
    """Run MobileNet-SSD on a lores BGR frame and return (seen, label, conf, bbox_lores) for the best target."""
    lo_h, lo_w = lo_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(lo_bgr, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.0
    best_bbox = None
    best_label = None

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        cls_id = int(detections[0, 0, i, 1])
        if cls_id < 0 or cls_id >= len(CLASSES):
            continue
        label = CLASSES[cls_id]
        if label not in TARGET_CLASSES:
            continue
        if conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([lo_w, lo_h, lo_w, lo_h])
            (x1, y1, x2, y2) = box.astype("int")
            x1 = max(0, min(lo_w - 1, x1))
            y1 = max(0, min(lo_h - 1, y1))
            x2 = max(0, min(lo_w - 1, x2))
            y2 = max(0, min(lo_h - 1, y2))
            cand_bbox = (x1, y1, x2, y2)
            # ROI gate on bbox center (lores coords)
            if not _roi_accept_bbox_center(cand_bbox, lo_w, lo_h):
                continue
            best_conf = conf
            best_bbox = cand_bbox
            best_label = label

    seen = best_conf >= HUMAN_MIN_CONF and best_bbox is not None and best_label is not None
    return seen, best_label, best_conf, best_bbox


def setup_camera_with_overlay(latest_det: Dict[str, Any]) -> Tuple[Picamera2, H264Encoder, CircularOutput]:
    """Configure Picamera2 with overlay; returns (picam2, encoder, circular_output)."""
    picam2 = Picamera2()
    video_cfg = picam2.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "RGB888"},
        controls={"FrameRate": FPS},
    )
    picam2.configure(video_cfg)

    encoder = H264Encoder(bitrate=BITRATE)

    # buffersize is in frames for Picamera2 CircularOutput; add slack for safety
    buffersize = int(max(1, round(PRE_ROLL_SEC * FPS))) + int(FPS * 2)
    circular = CircularOutput(buffersize=buffersize)

    def pre_callback(request):
        with MappedArray(request, "main") as m:
            img = m.array
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            h, w = img.shape[:2]
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, HEADER_H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

            now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(img, "HUMANCAM (area)", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)
            cv2.putText(img, now_s, (16, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_DIM, 2, cv2.LINE_AA)

            det_age = time.time() - float(latest_det.get("ts", 0))
            conf = float(latest_det.get("conf", 0.0))
            seen = bool(latest_det.get("seen", False)) and det_age < 2.0

            label = str(latest_det.get('label') or 'human')

            status = f"{label}: {'YES' if seen else 'no'}  conf={conf:.2f}"
            cv2.putText(img, status, (420, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                        (COLOR_OK if seen else COLOR_DIM), 2, cv2.LINE_AA)
            # ROI overlay (main coords)
            if ROI_ENABLE:
                inc_poly, exc_polys = _roi_polys_for_main()
                if inc_poly:
                    cv2.polylines(img, [np.array(inc_poly, dtype=np.int32)], True, (255, 255, 0), 2)
                    cv2.putText(img, "ROI", (inc_poly[0][0] + 6, inc_poly[0][1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                for ep in exc_polys:
                    cv2.polylines(img, [np.array(ep, dtype=np.int32)], True, (0, 0, 255), 2)
                    cv2.putText(img, "EXCLUDE", (ep[0][0] + 6, ep[0][1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


            if seen and latest_det.get("bbox") is not None:
                x1, y1, x2, y2 = latest_det["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OK, 2)
                det_label = str(latest_det.get('label') or 'human')
                draw_label(img, x1, y1, f"{det_label} {conf:.2f}", COLOR_OK)

            m.array[:] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    picam2.pre_callback = pre_callback
    return picam2, encoder, circular


def start_circular_recording(picam2: Picamera2, encoder: H264Encoder, circular: CircularOutput) -> None:
    try:
        picam2.stop_recording()
    except Exception:
        pass
    picam2.start_recording(encoder, circular)


def start_file_recording(picam2: Picamera2, encoder: H264Encoder, out_h264: str) -> None:
    try:
        picam2.stop_recording()
    except Exception:
        pass
    picam2.start_recording(encoder, FileOutput(out_h264))


def finalize_clip(base: str, meta: ClipMeta, conf_values: List[float], duration_s: float, note: str = "") -> None:
    """Finalize a clip: remux/concat, duration gate, write meta, quarantine on errors, cleanup temps."""
    final_mp4 = meta.clip_path
    tmp_dir = os.path.join(base, "tmp")
    ensure_dir(tmp_dir)

    tmp_pre_mp4 = os.path.join(tmp_dir, f"{meta.clip_name}_pre.mp4")
    tmp_main_mp4 = os.path.join(tmp_dir, f"{meta.clip_name}_main.mp4")

    try:
        # Stats
        if conf_values:
            meta.conf_min = float(min(conf_values))
            meta.conf_mean = float(sum(conf_values) / len(conf_values))
            meta.conf_max = float(max(conf_values))
            meta.detections = int(len(conf_values))

        # Remux main
        ffmpeg_remux_h264_to_mp4(meta.tmp_main_h264, tmp_main_mp4, meta.fps)

        # Optional pre-roll
        if meta.tmp_pre_h264 and os.path.exists(meta.tmp_pre_h264):
            ffmpeg_remux_h264_to_mp4(meta.tmp_pre_h264, tmp_pre_mp4, meta.fps)
            ffmpeg_concat_mp4([tmp_pre_mp4, tmp_main_mp4], final_mp4)
        else:
            shutil.move(tmp_main_mp4, final_mp4)

        meta.end_local = local_now_iso()
        meta.end_utc = utc_now_iso()
        meta.duration_s = float(duration_s)

        discard = duration_s < MIN_CLIP_SEC
        if discard:
            try:
                os.remove(final_mp4)
            except OSError:
                pass
            append_event(base, f"{local_now_iso()} | DISCARD | clip={meta.clip_name} | dur={duration_s:.2f}s {note}".rstrip())
        else:
            write_clip_meta(meta)
            append_event(base, f"{local_now_iso()} | HUMAN_EXIT | clip={meta.clip_name} | dur={duration_s:.2f}s {note}".rstrip())

    except Exception as e:
        qdir = os.path.join(base, "quarantine")
        ensure_dir(qdir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if os.path.exists(final_mp4):
                shutil.move(final_mp4, os.path.join(qdir, f"{stamp}_{os.path.basename(final_mp4)}"))
        except Exception:
            pass
        append_event(base, f"{local_now_iso()} | ERROR | finalize | clip={meta.clip_name} | {type(e).__name__}: {e}")
    finally:
        for p in [meta.tmp_main_h264, meta.tmp_pre_h264, tmp_pre_mp4, tmp_main_mp4]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
              
# -----------------------------
# Main
# -----------------------------
def main() -> None:
    fail_if_missing(PROTOTXT)
    fail_if_missing(CAFFEMODEL)

    base = pick_out_dir()
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    append_event(base, f"{local_now_iso()} | START | session={session_ts}")

    # Model
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    # Shared detection snapshot for overlay
    latest_det: Dict[str, Any] = {"seen": False, "label": None, "conf": 0.0, "bbox": None, "ts": 0.0}

    # Camera
    picam2, encoder, circular = setup_camera_with_overlay(latest_det)
    picam2.start()

    # Idle state: always keep the ring buffer running
    start_circular_recording(picam2, encoder, circular)

    # State machine
    recording = False
    cooldown_until = 0.0
    seen_streak = 0
    absent_streak = 0

    current_meta: Optional[ClipMeta] = None
    conf_values: List[float] = []
    clip_start_t = 0.0

    def new_paths(clip_name: str) -> Tuple[str, str, str]:
        # clips + meta are dated by YYYY/MM/DD
        day_path = datetime.now().strftime("%Y/%m/%d")
        clip_dir = os.path.join(base, "clips", day_path)
        meta_dir = os.path.join(base, "meta", day_path)
        tmp_dir = os.path.join(base, "tmp")
        ensure_dir(clip_dir)
        ensure_dir(meta_dir)
        ensure_dir(tmp_dir)
        final_mp4 = os.path.join(clip_dir, f"{clip_name}.mp4")
        meta_json = os.path.join(meta_dir, f"{clip_name}.json")
        tmp_main_h264 = os.path.join(tmp_dir, f"{clip_name}_main.h264")
        return final_mp4, meta_json, tmp_main_h264

    def start_clip(with_preroll: bool, trigger_label: str, note: str = "") -> None:
        nonlocal recording, current_meta, conf_values, clip_start_t

        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", trigger_label).lower()
        clip_name = "area_" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{safe_label}"
        final_mp4, meta_json, tmp_main_h264 = new_paths(clip_name)
        tmp_pre_h264 = os.path.join(base, "tmp", f"{clip_name}_pre.h264")

        current_meta = ClipMeta(
            clip_name=clip_name,
            clip_path=final_mp4,
            meta_path=meta_json,
            tmp_main_h264=tmp_main_h264,
            tmp_pre_h264=None,
            pre_roll_s=0.0,
            target_classes=sorted(list(TARGET_CLASSES)),
            trigger_class=trigger_label,
            start_local=local_now_iso(),
            start_utc=utc_now_iso(),
        )
        conf_values = []

        # Capture pre-roll BEFORE switching away from circular output
        if with_preroll and PRE_ROLL_SEC > 0:
            try:
                circular.copy_to(tmp_pre_h264, seconds=PRE_ROLL_SEC)
                current_meta.tmp_pre_h264 = tmp_pre_h264
                current_meta.pre_roll_s = float(PRE_ROLL_SEC)
            except Exception:
                current_meta.tmp_pre_h264 = None
                current_meta.pre_roll_s = 0.0

        start_file_recording(picam2, encoder, tmp_main_h264)
        recording = True
        clip_start_t = time.time()
        append_event(base, f"{local_now_iso()} | HUMAN_ENTER | clip={clip_name} {note}".rstrip())

    def stop_clip(note: str = "") -> None:
        nonlocal recording, current_meta, conf_values, clip_start_t
        if not recording or current_meta is None:
            return

        try:
            picam2.stop_recording()
        except Exception:
            pass

        duration_s = max(0.0, time.time() - clip_start_t)
        finalize_clip(base, current_meta, conf_values, duration_s, note=note)

        recording = False
        current_meta = None
        conf_values = []
        clip_start_t = 0.0

        # Resume circular buffer for next event
        start_circular_recording(picam2, encoder, circular)

    try:
        last_detect_t = 0.0

        while True:
            now = time.time()

            # Detection cadence
            if now - last_detect_t < DETECT_INTERVAL_S:
                time.sleep(0.01)
                continue
            last_detect_t = now

            lo_rgb = picam2.capture_array("lores")
            lo_bgr = cv2.cvtColor(lo_rgb, cv2.COLOR_RGB2BGR)

            seen, label, conf, bbox_lo = detect_targets_on_lores(net, lo_bgr)

            # Map bbox to main resolution for overlay display
            bbox_main = None
            if bbox_lo is not None:
                lo_h, lo_w = lo_bgr.shape[:2]
                main_w, main_h = MAIN_SIZE
                sx = main_w / float(lo_w)
                sy = main_h / float(lo_h)
                x1, y1, x2, y2 = bbox_lo
                bbox_main = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

            latest_det.update({"seen": seen, "label": label, "conf": float(conf), "bbox": bbox_main, "ts": now})

            # Update streaks
            if seen:
                seen_streak += 1
                absent_streak = 0
            else:
                absent_streak += 1
                seen_streak = 0

            # If recording, accumulate confidences and enforce max segment length
            if recording and current_meta is not None and seen:
                conf_values.append(float(conf))

            if recording and current_meta is not None:
                seg_len = now - clip_start_t
                if seg_len >= MAX_CLIP_SEC:
                    # Finalize this segment and immediately start another (no pre-roll for continuation)
                    keep_label = current_meta.trigger_class or "human"
                    stop_clip(note="segment_max")
                    start_clip(with_preroll=False, trigger_label=keep_label, note="segment_continue")
                    continue

            # Start logic
            if (not recording) and now >= cooldown_until and seen_streak >= START_FRAMES:
                start_clip(with_preroll=True, trigger_label=label or "human")
                continue

            # Stop logic
            if recording and absent_streak >= STOP_FRAMES:
                stop_clip(note="target_gone")
                cooldown_until = time.time() + COOLDOWN_SEC
                continue

    except KeyboardInterrupt:
        append_event(base, f"{local_now_iso()} | STOP | KeyboardInterrupt")
    finally:
        try:
            if recording and current_meta is not None:
                stop_clip(note="shutdown")
        except Exception:
            pass
        try:
            picam2.stop_recording()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
