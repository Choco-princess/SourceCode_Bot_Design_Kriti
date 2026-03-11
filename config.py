"""
=============================================================================
 ROVER PI — Central Configuration
=============================================================================
Edit the values below to match your hardware setup before deploying to the Pi.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# ESP32-CAM
# ---------------------------------------------------------------------------
# The ESP32-CAM runs its own web server.  Typical endpoints:
#   MJPEG stream : http://<IP>:81/stream
#   Single JPEG  : http://<IP>/capture
#
# >>> Change this IP to match your ESP32-CAM's address on the Wi-Fi network.
ESP32_CAM_IP = "192.168.105.245"
ESP32_STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"   # MJPEG stream
ESP32_CAPTURE_URL = f"http://{ESP32_CAM_IP}/capture"     # single JPEG snapshot
ESP32_CONTROL_URL = f"http://{ESP32_CAM_IP}/control"     # camera setting control
# How we fetch frames: "stream" = parse MJPEG stream, "snapshot" = poll /capture
# Stream mode is significantly faster (15-25 FPS vs 2-5 FPS) because it keeps
# one persistent connection open instead of a new HTTP request per frame.
ESP32_MODE = "stream"
# Timeout in seconds when connecting to / reading from the ESP32
# Lowered from 5s — ESP32 should respond in <1s on local Wi-Fi
ESP32_HTTP_TIMEOUT = 2

# JPEG quality sent to ESP32-CAM (0=best/largest .. 63=worst/smallest)
# Live Feed: Quality 10-15 offers high FPS while keeping decent visibility.
# User recommended: 10-15. We use 12 for a good balance.
ESP32_FEED_QUALITY = 12         # Optimal per stackexchange article
ESP32_CAPTURE_QUALITY = 5       # Sharp for ML

# ESP32-CAM framesize enum
#   5 = QVGA  320x240    (User Requested: Optimal for Live Feed FPS)
#   8 = VGA   640x480
# We use QVGA (5) for the fastest possible live feed.
ESP32_FRAMESIZE = 5             # QVGA 320x240

# Settings sent to ESP32-CAM automatically on startup via /control endpoint.
# Keys = ESP32 control variable names, values = integers.
# Reference: https://randomnerdtutorials.com/esp32-cam-ov2640-camera-settings/
# Do NOT include "framesize" here — changing it over HTTP causes the
# ESP32-CAM to reinitialise the camera buffer, which often kills the
# stream.  Set the resolution once via the ESP32 web UI instead.
ESP32_STARTUP_SETTINGS = {
    "brightness":  0,   # neutral (was -2 — way too dark)
    "contrast":    1,   # slight boost helps edge detection
    "gainceiling": 2,   # 2 = 8x  (helps in dim light)
    "ae_level":    1,   # slightly brighter auto-exposure
    "aec":         1,   # auto exposure ON
    "agc":         1,   # auto gain ON
    "awb":         1,   # auto white balance ON
}

# ---------------------------------------------------------------------------
# Arduino Serial
# ---------------------------------------------------------------------------
# On Raspberry Pi the Arduino usually shows up as /dev/ttyUSB0 or /dev/ttyACM0
# On Windows it will be something like "COM3"
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 1              # seconds

# ---------------------------------------------------------------------------
# Web Server
# ---------------------------------------------------------------------------
WEB_HOST = "0.0.0.0"           # Listen on all interfaces so laptops can connect
WEB_PORT = 8000

# ---------------------------------------------------------------------------
# ML Models — paths relative to this file
# ---------------------------------------------------------------------------
# TFLite classifier for categories: Brand logos, Vehicles, Furniture, Pets,
# Smart switches, Parcels.
# >>> Place your trained .tflite file here after running train/train_classifier.py
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier.tflite")
TFLITE_LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.txt")

# Known faces directory — put one clear JPEG per person, named as the person's
# name, e.g. known_faces/Roger_Federer.jpg
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

# ---------------------------------------------------------------------------
# Vision Pipeline
# ---------------------------------------------------------------------------
# Number of consecutive identical predictions required to confirm a detection.
PREDICTION_STABILITY_FRAMES = 3

# Confidence threshold for TFLite classifier (0.0 — 1.0)
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.55

# Maximum run duration in seconds
MAX_RUN_DURATION = 300          # 5 minutes

# ---------------------------------------------------------------------------
# Category → label mapping used by the unified classifier
# Each label in labels.txt should be formatted as "Category/Content",
# e.g. "Brand logo/Maybach", "Vehicle/Car", "Furniture/Chair"
# ---------------------------------------------------------------------------
CATEGORY_PREFIXES = [
    "Brand logo",
    "Vehicle",
    "Furniture",
    "Pets",
    "Smart switch",
    "Parcel",
]
