# RoboKriti — Rover Pi Software Stack

> Autonomous rover vision & control system for the RoboKriti robotics competition.
> A Raspberry Pi streams live video from an ESP32-CAM, runs real-time ML inference
> (image classification, QR codes, face recognition, number-plate OCR), and
> communicates with an Arduino over USB-Serial for motor/sensor control.

---

## Architecture

```
  ┌─── LAPTOP / PHONE ────────────────────────────────────────────┐
  │  Browser → http://<Pi-IP>:8000   (live feed + dashboard)      │
  └───────────────────────────────────────────────────────────────-┘
                           │ Wi-Fi
  ┌─── RASPBERRY PI ──────────────────────────────────────────────┐
  │  app.py ── FastAPI + Uvicorn (port 8000)                      │
  │    ├── vision.py ── MJPEG stream + 4 ML detectors             │
  │    │     ├── classifier.py   (TFLite MobileNetV2)             │
  │    │     ├── qr_detector.py  (pyzbar / OpenCV)                │
  │    │     ├── face_detector.py(YuNet + SFace, pure OpenCV)     │
  │    │     └── number_plate.py (Morpho ANPR + RapidOCR)         │
  │    └── serial_comms.py ── Arduino USB-Serial 115200 baud      │
  └───────┬───────────────────────────────────┬───────────────────┘
          │ USB                               │ Wi-Fi
  ┌───────▼───────────┐           ┌───────────▼──────────────┐
  │  ARDUINO           │           │  ESP32-CAM (OV2640)      │
  │  Motors, IR×4,     │           │  MJPEG on :81/stream     │
  │  Ultrasonic, PID   │           │  Snapshot on /capture    │
  └────────────────────┘           └──────────────────────────┘
```

---

## Project Structure

```
rover_pi/
├── app.py                  # FastAPI web server (entry point)
├── vision.py               # ESP32-CAM capture + multi-detector pipeline
├── serial_comms.py          # Arduino USB-Serial communication
├── config.py               # All tuneable settings
├── requirements.txt         # Python dependencies
├── detectors/
│   ├── classifier.py        # TFLite image classifier (MobileNetV2)
│   ├── qr_detector.py       # QR code detection (pyzbar + OpenCV)
│   ├── face_detector.py     # Face recognition (YuNet + SFace, no dlib)
│   └── number_plate.py      # Number plate ANPR + OCR
├── models/
│   ├── classifier.tflite    # Trained TFLite model
│   ├── labels.txt           # Class labels (Category/Content per line)
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── known_faces/             # One JPEG per person (e.g. Roger_Federer.jpg)
├── train/
│   └── train_classifier.py  # MobileNetV2 fine-tuning → TFLite export
├── templates/index.html     # Web dashboard (sci-fi HUD theme)
└── static/
    ├── style.css
    └── script.js
```

---

## Quick Start

```bash
# 1. System dependencies
sudo apt update && sudo apt install -y cmake libboost-all-dev libzbar0 tesseract-ocr

# 2. Python packages
cd rover_pi
pip install -r requirements.txt

# 3. Place your assets
#    models/classifier.tflite + models/labels.txt  (from training)
#    known_faces/Person_Name.jpg                    (for face recognition)

# 4. Edit config.py — set ESP32_CAM_IP and SERIAL_PORT

# 5. Run
python app.py
# Open http://<Pi-IP>:8000 on your laptop
```

---

## Configuration (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ESP32_CAM_IP` | `192.168.105.245` | ESP32-CAM IP on Wi-Fi |
| `ESP32_MODE` | `"stream"` | `"stream"` (15-25 FPS) or `"snapshot"` (2-5 FPS) |
| `ESP32_FEED_QUALITY` | `12` | JPEG quality for live feed (0=best, 63=worst) |
| `ESP32_CAPTURE_QUALITY` | `5` | JPEG quality for ML capture |
| `ESP32_FRAMESIZE` | `5` | 5=QVGA 320×240, 8=VGA 640×480 |
| `SERIAL_PORT` | `/dev/ttyUSB0` | Arduino serial port |
| `SERIAL_BAUD` | `115200` | Must match Arduino firmware |
| `WEB_PORT` | `8000` | Dashboard HTTP port |
| `CLASSIFIER_CONFIDENCE_THRESHOLD` | `0.55` | Min softmax score to report |
| `MAX_RUN_DURATION` | `300` | Auto-stop after 5 minutes |

---

## How It Works

### Capture & Detect (Dashboard Button)

1. User clicks **INITIATE CAPTURE & DETECT**
2. Pi copies latest frame from memory (instant, no new HTTP request)
3. Resize to 480px → bilateral denoise → run all 4 detectors sequentially
4. Returns annotated image + JSON results to the browser

### Automatic Detection (Arduino-Triggered, Non-Blocking)

During a competition run, detection is **fully parallel** — the rover never
stops for ML inference:

1. Arduino line-follower reaches a card station → sends `IMAGE_READY`
2. Pi **instantly grabs** the current frame (memory copy, ~0ms)
3. Pi sends `DONE` to Arduino **immediately** → rover resumes driving
4. ML detection runs in a **background thread** on the Pi while the rover moves
5. Results are logged asynchronously in the dashboard

```
Arduino                              Pi
   |                                  |
   |── IMAGE_READY ─────────────────>|  (rover at card)
   |                                  |  grab frame (instant)
   |<──────── DONE ──────────────────|  (sent immediately)
   |                                  |
   | [Rover resumes driving]          |  [ML runs in background]
   |                                  |  TFLite + QR + Face + Plate
   |                                  |  Results logged async
```

---

## ML Detectors

### 1. Image Classifier (`detectors/classifier.py`)

### 2. QR Code Detector (`detectors/qr_detector.py`)

### 3. Face Recognition (`detectors/face_detector.py`)

### 4. Number Plate ANPR (`detectors/number_plate.py`)

---

## Arduino Serial Protocol

Simple newline-terminated text at 115200 baud.

**Arduino → Pi:**

| Message | Meaning |
|---------|---------|
| `IMAGE_READY` | Camera pointed at card — capture now |
| `OBSTACLE_DETECTED` | Ultrasonic triggered — bypass starting |
| `PI_TIMEOUT` | Pi didn't respond in time |
| `SENSORS:S1=x,S2=x,...` | Periodic sensor data |

**Pi → Arduino:**

| Command | Meaning |
|---------|---------|
| `START_RUN` | Begin line following |
| `STOP` | Emergency stop |
| `DONE` | Frame captured — resume driving |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | HTML dashboard |
| `GET` | `/video_feed` | MJPEG live stream |
| `GET` | `/api/frame` | Single JPEG frame (JS fallback) |
| `POST` | `/api/capture` | Run all detectors, return JSON + base64 image |
| `GET` | `/api/capture_image` | Last annotated capture as JPEG |
| `POST` | `/api/start` | Start timed run → sends `START_RUN` to Arduino |
| `POST` | `/api/stop` | Stop run → sends `STOP` to Arduino |
| `GET` | `/api/status` | `{ running, elapsed, log[], sensors{} }` |

---

## Training the Classifier

> Run on your PC (GPU recommended), not on the Pi.

**Dataset structure** (ImageFolder, two levels):
```
dataset/
├── Brand logo/
│   ├── Apple/   ← images
│   └── Tesla/
├── Vehicle/
│   ├── Car/
│   └── Bicycle/
└── ...
```

**Train:**
```bash
python train/train_classifier.py --data_dir ./dataset --epochs 15
# Optional: evaluate
python train/train_classifier.py --data_dir ./dataset --epochs 15 --test
```

**What it does:**
1. Scans two-level directory → builds `(path, "Category/Content")` pairs
2. Splits 85/15 train/val, computes class weights for imbalance
3. Fine-tunes MobileNetV2 (last 30 layers), augments with flip/brightness/contrast
4. Exports `models/classifier.tflite` (float16, ~3 MB) + `models/labels.txt`

**Transfer to Pi:**
```bash
scp models/classifier.tflite models/labels.txt pi@<Pi-IP>:~/rover_pi/models/
```

---

## ESP32-CAM Setup

Flash the standard **CameraWebServer** example from Arduino IDE. Set your
Wi-Fi credentials and upload. Endpoints:

| Endpoint | Description |
|----------|-------------|
| `http://<IP>/capture` | Single JPEG |
| `http://<IP>:81/stream` | MJPEG stream |
| `http://<IP>/control?var=X&val=Y` | Camera settings |

Update `ESP32_CAM_IP` in `config.py`.

---


## Troubleshooting

| Problem | Fix |
|---------|-----|
| No video feed | `ping <ESP32_IP>`, test `http://<IP>:81/stream` in browser |
| Low FPS | Use `ESP32_MODE="stream"`, `ESP32_FRAMESIZE=5` (QVGA) |
| Serial port error | `ls /dev/ttyUSB* /dev/ttyACM*`, update `SERIAL_PORT` in config |
| Permission denied on serial | `sudo usermod -aG dialout $USER` then reboot |
| TFLite model not found | Train model first, copy `.tflite` + `labels.txt` to `models/` |
| Face not recognized | Ensure clear front-facing JPEG in `known_faces/`, try lower thresholds |
| No QR detected | Ensure `libzbar0` installed: `sudo apt install libzbar0` |
| Dashboard not loading | Check `hostname -I`, try `curl http://localhost:8000/api/status` |

---

## Dependencies

**System:** `cmake`, `libboost-all-dev`, `libzbar0`, `tesseract-ocr`

**Python:** `fastapi`, `uvicorn`, `requests`, `opencv-python-headless`, `numpy`,
`pyserial`, `ai-edge-litert`, `pyzbar`, `rapidocr-onnxruntime`, `pytesseract`,
`imutils`, `scikit-image`

**Training only (PC):** `tensorflow` ≥ 2.13, `Pillow`

