# AICAM Drone YOLOv8n – ONNX Development Guide

This guide explains how to **train**, **export**, and **run** a YOLOv8n
single-class (`drone`) model as an **ONNX** file that works correctly with
`AICAM_Drone_YOLOv8n.py` using **OpenCV DNN**.

It is written to be accurate, reproducible, and beginner-friendly, while still
respecting the real technical constraints of YOLOv8 + ONNX on embedded systems.

---

## Table of contents

1. Purpose and scope  
2. Repository layout  
3. Python virtual environments (venv)  
4. Installing required tools  
5. Dataset preparation  
6. Training YOLOv8n  
7. Exporting ONNX correctly  
8. Class names and labeling  
9. Verifying the ONNX model  
10. Running AICAM_Drone_YOLOv8n  
11. Raspberry Pi performance notes  
12. Common mistakes and fixes  
13. Mental model: how detection really works  
14. Final checklist  

---

## 1. Purpose and scope

This guide exists to prevent the most common YOLOv8 ONNX mistakes:

- Exporting ONNX with embedded NMS
- Using dynamic shapes that break OpenCV
- Mismatched class names
- Incorrect image sizes
- Silent failures with “no detections”

The goal is **correctness first**, not convenience.

---

## 2. Recommended repository layout

```
.
├── AICAM_Drone_YOLOv8n.py
├── README.md
├── data/
│   └── synth_drone_yolo.zip
├── models/
│   ├── drone_yolov8n.onnx
│   └── drone.names
```

---

## 3. Python virtual environment (venv)

### Why use a virtual environment?

- Prevents dependency conflicts
- Keeps system Python clean
- Makes builds reproducible
- Strongly recommended for Linux and Raspberry Pi

### Create and activate

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 4. Install required tools

Always install packages using:

```bash
python -m pip install ...
```

### Training and export system

```bash
python -m pip install ultralytics onnx opencv-python
```

### Headless systems / Raspberry Pi

```bash
python -m pip install ultralytics onnx opencv-python-headless
```

---

## 5. Dataset preparation

Unzip the uploaded dataset:

```bash
mkdir -p data
unzip data/synth_drone_yolo.zip -d data/
```

Expected structure:

```
data/synth_drone_yolo/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

The dataset defines **one class only**:

```yaml
names:
  0: drone
```

Do not rename or add classes.

---

## 6. Training YOLOv8n

Train a lightweight detector:

```bash
yolo detect train   model=yolov8n.pt   data=data/synth_drone_yolo/data.yaml   imgsz=640   epochs=50   batch=16
```

Training output:

```
runs/detect/train/weights/best.pt
```

---

## 7. Exporting ONNX (CRITICAL)

This is the most important step.

```bash
yolo export   model=runs/detect/train/weights/best.pt   format=onnx   imgsz=640   opset=12   nms=False   dynamic=False   simplify=False
```

### Why these flags matter

| Setting | Reason |
|------|------|
| `nms=False` | Script performs NMS itself |
| `dynamic=False` | Prevents OpenCV DNN shape issues |
| `opset=12` | Stable across devices |
| `simplify=False` | Avoids broken graphs |

Copy the result:

```bash
mkdir -p models
cp runs/detect/train/weights/best.onnx models/drone_yolov8n.onnx
```

---

## 8. Class names file

Create a names file:

```bash
echo drone > models/drone.names
```

This must match:
```bash
--classes drone
```

Mismatch = no detections.

---

## 9. Verify ONNX with OpenCV

Before running the main script:

```bash
python - <<'PY'
import cv2
net = cv2.dnn.readNetFromONNX("models/drone_yolov8n.onnx")
print("ONNX loaded successfully")
PY
```

If this fails, re-export the ONNX.

---

## 10. Run AICAM_Drone_YOLOv8n

```bash
python3 AICAM_Drone_YOLOv8n.py   --yolo-model models/drone_yolov8n.onnx   --yolo-names models/drone.names   --yolo-imgsz 640   --conf 0.6
```

Keep `--yolo-imgsz` consistent with export size.

---

## 11. Raspberry Pi performance notes

Recommended adjustments:

- Use `--yolo-imgsz 512`
- Increase frame skipping:
  ```
  --detect-every 2
  ```
- Limit inference:
  ```
  --max-infer-fps 5
  ```

Always use `opencv-python-headless` on headless systems.

---

## 12. Common mistakes and fixes

| Issue | Cause |
|----|----|
| No detections | Wrong class name |
| ONNX won’t load | Wrong opset or dynamic shapes |
| Bad boxes | imgsz mismatch |
| Low FPS | imgsz too large |
| Crashes on Pi | GUI OpenCV installed |

---

## 13. Mental model (important)

YOLOv8 ONNX **does not perform detection**.

It outputs:
- Bounding box predictions
- Confidence scores
- Class logits

Your script:
1. Decodes predictions
2. Applies thresholds
3. Runs NMS
4. Triggers logic (recording, alerts, etc.)

Embedding NMS in ONNX breaks this pipeline.

---

## 14. Final checklist

- [ ] Single class: `drone`
- [ ] ONNX exported with `nms=False`
- [ ] `drone.names` matches
- [ ] imgsz consistent everywhere
- [ ] ONNX loads in OpenCV

---

End of guide.
