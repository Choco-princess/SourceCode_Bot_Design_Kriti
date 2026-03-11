"""
=============================================================================
 Vision Pipeline  —  vision.py
=============================================================================
Fetches frames from the ESP32-CAM's HTTP stream and runs all detectors
on demand when the user clicks "Capture" on the web dashboard.

ESP32-CAM Capture Modes
-----------------------
1. **"stream"** — connects to the MJPEG stream at ESP32_STREAM_URL and
   parses individual JPEG frames out of the multipart HTTP response.
   This is the fastest and most reliable mode for the live feed.

2. **"snapshot"** — repeatedly GETs a single JPEG from ESP32_CAPTURE_URL.
   Simpler but slower (~2-5 fps depending on network).

Thread Safety
-------------
The latest frame and latest detection results are stored behind a
threading.Lock so the FastAPI server can read them at any time.
=============================================================================
"""

import time
import threading

import cv2
import numpy as np
import requests

import config
from detectors.qr_detector import QRDetector
from detectors.number_plate import NumberPlateDetector
from detectors.face_detector import FaceDetector

# The TFLite classifier is optional — skip gracefully if model file is missing.
_classifier = None
try:
    from detectors.classifier import TFLiteClassifier
    _classifier = TFLiteClassifier(
        model_path=config.TFLITE_MODEL_PATH,
        labels_path=config.TFLITE_LABELS_PATH,
        confidence_threshold=config.CLASSIFIER_CONFIDENCE_THRESHOLD,
    )
    print("[VISION] TFLite classifier loaded.")
except FileNotFoundError as exc:
    print(f"[VISION] {exc}")
    print("[VISION] Classifier disabled — other detectors still active.")
except Exception as exc:
    print(f"[VISION] Could not load classifier: {exc}")


class VisionPipeline:
    """ESP32-CAM capture + multi-detector inference."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

        # Latest raw JPEG bytes from ESP32 — stored directly, no decode
        self._frame_jpeg = None
        # Latest capture result (set after user clicks Capture)
        self._last_capture_jpeg = None  # JPEG bytes of captured image
        self._last_results = None       # list of result dicts
        self._capture_busy = False

        # Lightweight detector — loads instantly
        self._qr = QRDetector()

        # Heavy detectors — loaded on startup to prevent delay during run
        print("[VISION] Loading heavy detectors on startup...")
        self._plate = NumberPlateDetector()
        self._face = FaceDetector(config.KNOWN_FACES_DIR)
        self._detectors_loaded = True
        print("[VISION] Detectors loaded successfully.")

        # Separate sessions: one for stream, one for control/capture
        # ESP32-CAM is single-core — mixing on one session chokes it
        self._stream_session = requests.Session()
        self._ctrl_session = requests.Session()

        self._running = False

    # ------------------------------------------------------------------
    # ESP32-CAM Remote Configuration
    # ------------------------------------------------------------------
    def _esp32_set(self, var: str, val: int):
        """Send a single control command to the ESP32-CAM."""
        try:
            self._ctrl_session.get(
                config.ESP32_CONTROL_URL,
                params={"var": var, "val": val},
                timeout=2,
            )
        except requests.RequestException:
            pass  # best-effort

    def _apply_esp32_settings(self):
        """Apply all startup settings from config to the ESP32-CAM."""
        settings = getattr(config, "ESP32_STARTUP_SETTINGS", {})
        if not settings:
            return
        print("[VISION] Applying ESP32-CAM settings...")
        for var, val in settings.items():
            self._esp32_set(var, val)
            time.sleep(0.2)   # ESP32 is slow — give it time between cmds
        # Let the camera settle after config changes
        time.sleep(0.5)
        print("[VISION] ESP32-CAM configured.")

    def _set_esp32_quality(self, quality: int):
        """Change JPEG quality on the ESP32-CAM (0=best, 63=worst)."""
        self._esp32_set("quality", quality)

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------
    def start(self):
        """Begin background frame fetch (live feed only — no inference)."""
        self._running = True
        # Apply ESP32-CAM settings (brightness, contrast, etc.)
        self._apply_esp32_settings()
        # Set resolution once (VGA) and feed quality
        fs = getattr(config, "ESP32_FRAMESIZE", 8)
        feed_q = getattr(config, "ESP32_FEED_QUALITY", 12)
        self._esp32_set("framesize", fs)
        time.sleep(0.5)  # let sensor reinit
        self._set_esp32_quality(feed_q)
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def get_frame_jpeg(self):
        """Return the latest raw JPEG bytes directly (zero re-encoding)."""
        with self._lock:
            return self._frame_jpeg

    def get_raw_frame(self):
        """Return a copy of the latest BGR frame (decodes on demand)."""
        with self._lock:
            if self._frame_jpeg is None:
                return None
            img_array = np.frombuffer(self._frame_jpeg, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def get_last_results(self):
        """Return the last capture+detect results."""
        with self._lock:
            return {
                "results": self._last_results,
                "capture_jpeg": self._last_capture_jpeg,
            }

    # ------------------------------------------------------------------
    # Lazy-load heavy detectors (first capture only)
    # ------------------------------------------------------------------
    def _ensure_detectors(self):
        """Load plate + face detectors on first use to save RAM at startup."""
        if self._detectors_loaded:
            return
        print("[VISION] Loading detectors (first capture)...")
        self._plate = NumberPlateDetector()
        self._face = FaceDetector(config.KNOWN_FACES_DIR)
        self._detectors_loaded = True
        print("[VISION] All detectors ready.")

    # ------------------------------------------------------------------
    # Capture + Detect  (called by app.py when user clicks Capture or auto-10s)
    # ------------------------------------------------------------------
    def capture_and_detect(self):
        """
        Grab the current frame, run ALL detectors, and return results.
        Used by the dashboard Capture button (synchronous — waits for results).
        """
        # Immediately copy the latest frame from the fast stream to avoid ESP32-CAM hang.
        with self._lock:
            jpeg_bytes = self._frame_jpeg
            
        if jpeg_bytes is None:
            # Fallback to fetching it directly if stream isn't active
            jpeg_bytes = self._fetch_frame_snapshot()
            
        if jpeg_bytes is None:
            return None

        return self._run_detectors_on_jpeg(jpeg_bytes)

    # ------------------------------------------------------------------
    # Detect from pre-captured JPEG bytes (used for async detection)
    # ------------------------------------------------------------------
    def detect_from_jpeg(self, jpeg_bytes):
        """
        Run all detectors on pre-captured JPEG bytes.
        Unlike capture_and_detect(), this does NOT fetch a new frame —
        it uses the already-captured bytes passed in.
        Called from a background thread so the rover can keep driving.
        """
        if jpeg_bytes is None:
            return None
        return self._run_detectors_on_jpeg(jpeg_bytes)

    # ------------------------------------------------------------------
    # Shared detection logic (used by both capture_and_detect and detect_from_jpeg)
    # ------------------------------------------------------------------
    def _run_detectors_on_jpeg(self, jpeg_bytes):
        """Core detection pipeline: decode → resize → denoise → run all detectors."""
        img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        # Resize once for ALL detectors — huge speed boost on Pi 3
        h, w = frame.shape[:2]
        max_w = 480
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        # Light denoise for ESP32-CAM
        frame = cv2.bilateralFilter(frame, 5, 50, 50)

        t0 = time.time()

        with self._inference_lock:
            classifier_results = []
            qr_results = []
            plate_results = []
            face_results = []

            if _classifier is not None:
                all_preds = _classifier.detect(frame)
                all_preds.sort(key=lambda d: d.get("confidence", 0), reverse=True)
                classifier_results = all_preds[:3]

            qr_results = self._qr.detect(frame)
            plate_results = self._plate.detect(frame)
            face_results = self._face.detect(frame)

            elapsed = time.time() - t0
            print(f"[VISION] Detection took {elapsed:.2f}s")

            all_results = classifier_results + qr_results + plate_results + face_results

            annotated = frame.copy()
            for det in all_results:
                bbox = det.get("bbox")
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{det['category']}: {det['content']}"
                    cv2.putText(annotated, label, (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            capture_jpeg = buf.tobytes()

            results = {
                "classifier": classifier_results,
                "qr": qr_results,
                "face": face_results,
                "plate": plate_results,
                "all": all_results,
                "timestamp": time.time(),
                "capture_jpeg": capture_jpeg,
            }

            with self._lock:
                self._last_capture = annotated
                self._last_capture_jpeg = capture_jpeg
                self._last_results = results

            return results

    # ------------------------------------------------------------------
    # ESP32-CAM Frame Fetching
    # ------------------------------------------------------------------
    def _fetch_frame_snapshot(self):
        """GET a single JPEG from the ESP32-CAM /capture endpoint."""
        try:
            resp = self._ctrl_session.get(
                config.ESP32_CAPTURE_URL,
                timeout=2,
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
        except requests.RequestException as exc:
            print(f"[VISION] Snapshot fetch error: {exc}")
        return None

    def _stream_frames(self):
        """Generator: connect to ESP32-CAM MJPEG stream and yield raw JPEG bytes.
        Uses bytearray for O(1) append instead of O(n) bytes concatenation."""
        url = config.ESP32_STREAM_URL
        print(f"[VISION] Connecting to MJPEG stream: {url}")

        SOI = b"\xff\xd8"  # JPEG Start Of Image
        EOI = b"\xff\xd9"  # JPEG End Of Image

        while self._running:
            try:
                resp = self._stream_session.get(
                    url, stream=True, timeout=(3, 10),
                    # (connect_timeout, read_timeout) — read=10s so
                    # slow frames don't cause a reconnect storm
                )
                if resp.status_code != 200:
                    print(f"[VISION] Stream HTTP {resp.status_code}")
                    time.sleep(1)
                    continue

                buf = bytearray()
                # 8192 bytes is a standard TCP buffer size, often more efficient for images
                for chunk in resp.iter_content(chunk_size=8192):
                    if not self._running:
                        resp.close()
                        return

                    buf.extend(chunk)

                    # Extract all complete JPEG frames from the buffer
                    while True:
                        si = buf.find(SOI)
                        if si == -1:
                            buf.clear()
                            break
                        ei = buf.find(EOI, si + 2)
                        if ei == -1:
                            # Trim junk before SOI to keep buffer small
                            if si > 0:
                                del buf[:si]
                            break
                        # Complete JPEG frame found
                        yield bytes(buf[si : ei + 2])
                        del buf[: ei + 2]

            except requests.RequestException as exc:
                if not self._running:
                    return
                print(f"[VISION] Stream error: {exc} — reconnecting...")
                time.sleep(0.5)

    # ------------------------------------------------------------------
    # Main capture loop — live feed only (no inference per frame)
    # ------------------------------------------------------------------
    def _capture_loop(self):
        print(f"[VISION] Starting live feed (mode={config.ESP32_MODE})")

        if config.ESP32_MODE == "stream":
            frame_source = self._stream_frames()
        else:
            def snapshot_gen():
                while self._running:
                    jpeg = self._fetch_frame_snapshot()
                    if jpeg is not None:
                        yield jpeg
                    else:
                        time.sleep(0.3)
            frame_source = snapshot_gen()

        for jpeg_bytes in frame_source:
            if not self._running:
                break

            # Store raw JPEG bytes directly — no decode/re-encode overhead
            with self._lock:
                self._frame_jpeg = jpeg_bytes

        print("[VISION] Capture loop ended.")
