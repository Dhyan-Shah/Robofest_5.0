# 🚁 ARCNET — Aerial Robotics Minefield Navigation System
### GUJCOST Robofest 5.0 | National Finalists | Senior Category

> *"You never learn to swim by learning about water."*

G H Patel College of Engineering & Technology | CV/AIML Team | March 2026

---

## 📌 Project Overview

Team ARCNET built an autonomous drone system for the **Aerial Robotics Minefield Navigation Challenge** at GUJCOST Robofest 5.0 — India's biggest robotics competition.

**Mission:** Deploy a drone over a 20m × 100m minefield, detect 20–40 buried mines, generate a safe walking path, and physically mark each mine using a servo-triggered disc-dropping mechanism — all within a strict time window.

---

## 🛠️ Hardware Setup

| Component | Details |
|---|---|
| **Raspberry Pi 4** | Onboard compute — YOLO inference, GPS reading, MAVLink control |
| **Pixhawk Flight Controller** | Drone navigation, servo control via AUX1 (Channel 9) |
| **u-blox M8N GPS Module** | Connected to Pixhawk, read via MAVLink at 57600 baud |
| **Waveshare USB Camera** | Frame capture at 1280×720 for mine detection |
| **WiFi Router** | Inter-drone swarm communication |
| **Servo Motor** | Mine marker disc-drop mechanism on Pixhawk AUX1 |

---

## 🧠 AI/ML Models

### Mine Detection — YOLOv8
- Trained on a **custom annotated dataset** across varied lighting, angles, and terrain
- Achieves **~85% accuracy** on validation set
- Converted to **TFLite** for optimized real-time inference on Raspberry Pi 4
- Input size: 640×640

### Gesture Recognition — MobileNetV2
- Trained on a **custom gesture dataset** built from scratch
- Recognizes: **ARM**, **TAKEOFF**, **RTL** commands
- Deployed on Raspberry Pi for real-time drone control

> 📦 Model files are not included in this repository due to size.

---

## 🔄 Full Pipeline

```
capture.py  →  pipeline.py  →  hybrid.py  →  mark_mines.py
```

| Script | Purpose | Key Technologies |
|---|---|---|
| `capture.py` | Captures frames + stamps GPS on each frame | pymavlink, OpenCV, threading |
| `pipeline.py` | GPS-based orthomosaic stitching + YOLO mine detection | pyproj, YOLOv8 TFLite, numpy |
| `hybrid.py` | Safe path generation around detected mines | Dijkstra, NetworkX, occupancy grid |
| `mark_mines.py` | Real-time detection + autonomous disc marking | MAVLink, servo PWM, threading |
| `test_servo.py` | Servo mechanism test and calibration | MAVLink, PWM control |


---

## 📡 GPS Configuration

| Parameter | Value |
|---|---|
| Port | `/dev/ttyAMA0` |
| Baud Rate | `57600` |
| Fix Type | `4` (RTK Fix) |
| Satellites | `11` |
| Verified Coordinates | LAT 22.5604, LON 72.9199, ALT ~44m |

---

## 🗺️ Pipeline Details

### Frame Capture (`capture.py`)
- Camera: Waveshare USB via `cv2.VideoCapture(0)` at 1280×720
- GPS overlay drawn on each frame (LAT, LON, ALT, SATS, FIX)
- Output: `frames/frame_XXXX.jpg` + `frames/gps_log.json`

### Orthomosaic Stitching (`pipeline.py`)
- GPS coordinates projected to local meters using flat-earth formula
- Ground coverage calculated from altitude and camera FOV (62.2° × 48.8°)
- YOLO TFLite runs a single detection pass on the full orthomosaic
- Mine pixel positions converted back to real GPS coordinates

### Pathfinding (`hybrid.py`)
- Grid resolution: 0.5m per cell over 20m × 100m field
- Safe radius: 1.2m exclusion zone around each mine (1m + 0.2m buffer)
- Algorithm: Dijkstra on NetworkX graph with virtual FINISH_LINE node

### Autonomous Marking (`mark_mines.py`)
- YOLO inference on live camera frame every 0.3 seconds
- Mine GPS calculated from pixel position + drone GPS + FOV math
- Deduplication: mines within 1.5m of already-marked mines are skipped
- Servo: descend to 0.5m → PWM 1900 (release) → ascend

---

## 🎯 Servo Configuration

| Parameter | Value |
|---|---|
| Channel | Pixhawk AUX1 (Channel 9) |
| Hold PWM | `1100` (disc held) |
| Release PWM | `1900` (disc drops) |
| Drop Duration | 1.5 seconds |

---

## 📦 Requirements

```
pymavlink
opencv-python
numpy==1.23.5
pyproj
networkx
tflite-runtime
pyserial
future
```

---

## 📊 Expected Scoring

| Task | Points |
|---|---|
| Safe takeoff + swarm activation | 10 |
| Gesture / voice recognition | 10 |
| Mine detection + mapping | 30 |
| Safe path creation + marking | 20 |
| Person crosses safely | 20 |
| Time bonus (per minute saved) | +5 |
| **Expected Total** | **70–90+ / 95** |

---

## 👥 Team ARCNET

| Name | Role |
|---|---|
| **Dhyan** | CV/AIML — Model training, TFLite deployment, full pipeline |
| **Mohammed Ali** | Team Member |
| **Kathan Majithia** | Team Member |
| **Zinal Jain** | Team Member |

**Mentor:** Priyang Bhatt

**Institution:** G H Patel College of Engineering & Technology (GCET)

---

## 📄 License

This project is for educational and competition purposes.

---

*ARCNET | GUJCOST Robofest 5.0 | CV/AIML Team | March 2026*