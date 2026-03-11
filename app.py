"""
=============================================================================
 FastAPI Web Server  —  app.py
=============================================================================
Hosts the rover dashboard and exposes:
    GET  /                  — serves the HTML dashboard
    GET  /video_feed        — MJPEG live camera stream (re-streamed from ESP32)
    POST /api/capture       — capture a photo, run all ML detectors, return results
    GET  /api/capture_image — returns the last captured+annotated image as JPEG
    POST /api/start         — start a run (resets timer, tells Arduino)
    POST /api/stop          — stop the run
    GET  /api/status        — JSON snapshot of current state
=============================================================================
Run on the Pi:
    python app.py
Then open http://<Pi-IP>:8000 on your laptop.
=============================================================================
"""

import json
import time
import asyncio
import threading
import base64
import signal
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from vision import VisionPipeline
from serial_comms import ArduinoComm

# ── Initialise components ────────────────────────────────────────────────
app = FastAPI(title="RoboKriti Rover Dashboard")

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

vision = VisionPipeline()
arduino = ArduinoComm()

# Run state
_run_lock = threading.Lock()
_run_active = False
_run_start_time = 0.0
_detected_log = []  # history of all confirmed detections during this run


# ── Arduino Event Handler ───────────────────────────────────────────────
def _run_detection_async(frame_jpeg):
    """
    Run ML detection on a previously-captured frame in a background thread.
    The rover is already driving again by the time this runs.
    """
    global _detected_log

    results = vision.detect_from_jpeg(frame_jpeg)
    if results and results["all"]:
        for d in results["all"]:
            entry = f"{d['category']}: {d['content']}"
            with _run_lock:
                if entry not in _detected_log:
                    _detected_log.append(entry)
            print(f"[APP] Detected → {entry}")
    else:
        print("[APP] No confident detection on this card")


def _on_arduino_event(event: str):
    """
    Called by serial_comms when Arduino sends IMAGE_READY, OBSTACLE_DETECTED, etc.
    Runs in a separate thread — must be thread-safe.

    Design: capture frame instantly → send DONE → run ML in parallel.
    The rover never waits for ML inference to finish.
    """
    if event == "IMAGE_READY":
        print("[APP] IMAGE_READY received — grabbing frame...")

        # 1. Instant frame capture (memory copy from MJPEG buffer)
        frame_jpeg = vision.get_frame_jpeg()

        # 2. Tell Arduino to resume driving IMMEDIATELY
        arduino.done()
        print("[APP] Sent DONE to Arduino — rover resumes driving")

        # 3. Run ML detection in a background thread (non-blocking)
        if frame_jpeg is not None:
            threading.Thread(
                target=_run_detection_async,
                args=(frame_jpeg,),
                daemon=True,
            ).start()
        else:
            print("[APP] No frame available — skipping detection")

    elif event == "OBSTACLE_DETECTED":
        print("[APP] Obstacle detected — bypass starting...")

    elif event == "PI_TIMEOUT":
        print("[APP] Arduino timed out waiting for Pi")

arduino.set_event_callback(_on_arduino_event)


# ── Helpers ──────────────────────────────────────────────────────────────
def _elapsed():
    if not _run_active:
        return 0.0
    return time.time() - _run_start_time


# ── Dashboard HTML ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Single-frame endpoint (polled by JS) ────────────────────────────────
@app.get("/api/frame")
async def api_frame():
    """Return the latest JPEG frame.  JS polls this for the live feed."""
    jpeg = vision.get_frame_jpeg()
    if jpeg is None:
        return Response(status_code=204)          # No Content — JS retries
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ── MJPEG live stream (fastest — zero JS overhead) ─────────────────────
@app.get("/video_feed")
async def video_feed():
    """
    MJPEG stream: browser renders natively via <img src="/video_feed">.
    No JS fetch/blob/URL cycle — just raw JPEG passthrough.
    """
    async def generate():
        BOUNDARY = b"--frame\r\n"
        HEADER = b"Content-Type: image/jpeg\r\n\r\n"
        prev = None
        while True:
            jpeg = vision.get_frame_jpeg()
            if jpeg is not None and jpeg is not prev:
                # Optimized for max FPS: yield immediately
                yield BOUNDARY + HEADER + jpeg + b"\r\n"
                prev = jpeg
                # Minimal sleep to yield control but not throttle FPS
                # 0.03 (30ms) caps at ~33 FPS. 0.005 caps at ~200 FPS.
                await asyncio.sleep(0.005)
            else:
                # Wait briefly for new frame from vision thread
                await asyncio.sleep(0.005)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Capture + Detect API ────────────────────────────────────────────────
@app.post("/api/capture")
async def api_capture():
    """
    Capture a photo from ESP32-CAM, run all 4 detectors, and return results.
    Returns:
        classifier: top-2 TFLite predictions
        qr: QR code detections
        face: face recognition results
        plate: number plate OCR results
    """
    global _detected_log

    results = await asyncio.get_event_loop().run_in_executor(
        None, vision.capture_and_detect
    )

    if results is None:
        return JSONResponse(
            {"error": "No frame available — is the ESP32-CAM connected?"},
            status_code=503,
        )

    # Build response
    def _format(det_list):
        out = []
        for d in (det_list or []):
            entry = {
                "category": d.get("category", ""),
                "content": d.get("content", ""),
            }
            if "confidence" in d:
                entry["confidence"] = round(d["confidence"], 3)
            out.append(entry)
        return out

    # Add to detection log
    for d in results["all"]:
        entry = f"{d['category']}: {d['content']}"
        with _run_lock:
            if entry not in _detected_log:
                _detected_log.append(entry)

    # Encode capture image as base64 for displaying in browser
    capture_b64 = ""
    # Retrieve directly from results dict (vision now includes it) or fallback
    jpeg_data = results.get("capture_jpeg")
    if not jpeg_data:
        # Fallback to stored last results
        last_res = vision.get_last_results()
        if last_res:
            jpeg_data = last_res.get("capture_jpeg")

    if jpeg_data:
        capture_b64 = base64.b64encode(jpeg_data).decode("ascii")

    return {
        "classifier": _format(results["classifier"]),
        "qr": _format(results["qr"]),
        "face": _format(results["face"]),
        "plate": _format(results["plate"]),
        "capture_image": capture_b64,
        "timestamp": results["timestamp"],
    }


@app.get("/api/capture_image")
async def api_capture_image():
    """Return the last captured+annotated image as JPEG."""
    data = vision.get_last_results()
    if data and data["capture_jpeg"]:
        return Response(content=data["capture_jpeg"], media_type="image/jpeg")
    return JSONResponse({"error": "No capture available"}, status_code=404)


# ── REST API ─────────────────────────────────────────────────────────────
@app.post("/api/start")
async def api_start():
    global _run_active, _run_start_time, _detected_log
    with _run_lock:
        _run_active = True
        _run_start_time = time.time()
        _detected_log = []
    arduino.start_run()
    return {"status": "started"}


@app.post("/api/stop")
async def api_stop():
    global _run_active
    with _run_lock:
        _run_active = False
    arduino.stop()
    return {"status": "stopped"}


@app.get("/api/status")
async def api_status():
    with _run_lock:
        return {
            "running": _run_active,
            "elapsed": round(_elapsed(), 1),
            "log": _detected_log[-30:],
            "sensors": arduino.get_sensor_data(),
        }


# ── Background: auto-stop after 5 min ───────────────────────────────────
def _watchdog():
    global _run_active
    while True:
        time.sleep(0.5)
        with _run_lock:
            if _run_active and _elapsed() >= config.MAX_RUN_DURATION:
                _run_active = False
                arduino.stop()
                print("[APP] Run auto-stopped — 5 min limit reached.")


# ── Lifecycle Events ─────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    vision.start()

    arduino.connect()
    if arduino.is_connected:
        arduino.start_listener()

    threading.Thread(target=_watchdog, daemon=True).start()

    print(f"\n{'='*60}")
    print(f"  Rover Dashboard → http://0.0.0.0:{config.WEB_PORT}")
    print(f"  ESP32-CAM stream → {config.ESP32_STREAM_URL}")
    print(f"{'='*60}\n")


# ── Main ─────────────────────────────────────────────────────────────────
def _handle_exit(sig, frame):
    """Ctrl+C / SIGTERM handler — stop everything and exit immediately."""
    print("\n[APP] Shutting down...")
    vision.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        log_level="info",
    )
