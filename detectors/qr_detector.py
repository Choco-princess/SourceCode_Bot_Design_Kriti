"""
=============================================================================
 QR Code Detector
=============================================================================
Uses OpenCV's built-in QR detector (no extra dependencies) to find and decode
QR codes in a frame.  Falls back to pyzbar if available for better accuracy.
"""

import cv2
import numpy as np

# Try to import pyzbar for better QR decoding; fall back to OpenCV if missing.
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    _USE_PYZBAR = True
except (ImportError, FileNotFoundError, OSError):
    _USE_PYZBAR = False


class QRDetector:
    """Detect and decode QR codes in an image frame."""

    def __init__(self):
        if not _USE_PYZBAR:
            self._detector = cv2.QRCodeDetector()

    @staticmethod
    def _enhance_for_qr(gray):
        """Sharpen and threshold grayscale image to help QR detection
        on blurry / low-contrast ESP32-CAM frames."""
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        return enhanced

    def detect(self, frame):
        """
        Parameters
        ----------
        frame : numpy.ndarray  (BGR image from OpenCV)

        Returns
        -------
        list[dict]  — each dict has keys:
            "category" : "QR code"
            "content"  : decoded string
            "bbox"     : (x, y, w, h) or None
        """
        results = []

        if _USE_PYZBAR:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Try original first
            decoded_objects = pyzbar_decode(gray)
            # If nothing found, try enhanced version
            if not decoded_objects:
                enhanced = self._enhance_for_qr(gray)
                decoded_objects = pyzbar_decode(enhanced)
            # Also try adaptive threshold as last resort
            if not decoded_objects:
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 15, 8
                )
                decoded_objects = pyzbar_decode(thresh)
            for obj in decoded_objects:
                data = obj.data.decode("utf-8", errors="replace")
                x, y, w, h = obj.rect
                results.append({
                    "category": "QR code",
                    "content": data,
                    "bbox": (x, y, w, h),
                })
        else:
            # OpenCV detector — try original, then enhanced
            data, points, _ = self._detector.detectAndDecode(frame)
            if not data:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                enhanced = self._enhance_for_qr(gray)
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                data, points, _ = self._detector.detectAndDecode(enhanced_bgr)
            if data:
                bbox = None
                if points is not None and len(points) > 0:
                    pts = points[0]
                    x_min = int(pts[:, 0].min())
                    y_min = int(pts[:, 1].min())
                    x_max = int(pts[:, 0].max())
                    y_max = int(pts[:, 1].max())
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                results.append({
                    "category": "QR code",
                    "content": data,
                    "bbox": bbox,
                })

        return results
