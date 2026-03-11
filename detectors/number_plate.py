"""
=============================================================================
 Vehicle Number Plate Detector + OCR  (PyImageSearch ANPR approach)
=============================================================================
Based on Adrian Rosebrock's ANPR tutorial from PyImageSearch.
Uses morphological operations + Scharr gradient + contour analysis to
localize plates, then RapidOCR (PaddleOCR ONNX) for text recognition
with Tesseract as fallback.

Pipeline:
  1. Blackhat morphology  → reveal dark text on light plate background
  2. Light-region mask    → closing + Otsu to find bright plate area
  3. Scharr gradient (x)  → emphasize vertical edges of characters
  4. Blur + close + Otsu  → merge character edges into plate blob
  5. Erode / dilate       → clean noise
  6. AND with light mask  → isolate plate region
  7. Contour + AR filter  → pick the best rectangular plate contour
  8. clear_border + OCR   → clean ROI edges, then OCR

Install:
  sudo apt install tesseract-ocr
  pip install pytesseract imutils scikit-image rapidocr-onnxruntime
=============================================================================
"""

import re
import cv2
import numpy as np
import os
import imutils

try:
    from skimage.segmentation import clear_border
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

# Primary OCR engine: RapidOCR (PaddleOCR models via ONNX Runtime)
try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPID_OCR = True
except ImportError:
    _RAPID_OCR = False

# Fallback OCR engine: Tesseract
try:
    import pytesseract
    _TESSERACT = True
except ImportError:
    _TESSERACT = False

_AVAILABLE = _RAPID_OCR or _TESSERACT
if not _AVAILABLE:
    print("[WARN] No OCR engine installed — number plate detection disabled.")
    print("       Install: pip install rapidocr-onnxruntime  (recommended)")
    print("       Or:      pip install pytesseract + sudo apt install tesseract-ocr")


class NumberPlateDetector:
    """
    Automatic Number Plate Recognition (ANPR) detector.
    
    Based on the PyImageSearch approach by Adrian Rosebrock:
    morphological ops → gradient → contour filtering → OCR.
    """

    def __init__(self, min_ar=2, max_ar=7, debug=False):
        """
        Parameters
        ----------
        min_ar : float
            Minimum aspect ratio for a plate bounding box.
            Indian plates ~4.5, European ~5.0, but allow wider range.
        max_ar : float
            Maximum aspect ratio for a plate bounding box.
        debug : bool
            If True, show intermediate images (only works with GUI).
        """
        self.minAR = min_ar
        self.maxAR = max_ar
        self.debug = debug

        # RapidOCR engine (primary — much better than Tesseract)
        self._rapid = None
        if _RAPID_OCR:
            try:
                self._rapid = RapidOCR()
            except Exception as e:
                print(f"[WARN] RapidOCR init failed: {e}")

        # Also load Haar cascade as a backup strategy
        self.plate_cascade = None
        if hasattr(cv2, 'data'):
            path = os.path.join(cv2.data.haarcascades,
                                'haarcascade_russian_plate_number.xml')
            if os.path.exists(path):
                self.plate_cascade = cv2.CascadeClassifier(path)

        # Indian plate patterns:
        #   Standard: XX 00 XX 0000  (state-district-series-number) e.g. TS09EJ1115
        #   BH-series: 00 BH 0000 X  (Bharat series) e.g. 22BH6517A
        self._plate_patterns = [
            re.compile(r"\d{2}\s?BH\s?\d{4}\s?[A-Z]", re.IGNORECASE),       # BH-series
            re.compile(r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{2,4}", re.IGNORECASE),  # Standard
        ]

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------
    def _debug_imshow(self, title, image):
        if self.debug:
            try:
                cv2.imshow(title, image)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Step 1-6: Locate license plate candidate contours
    # ------------------------------------------------------------------
    def _locate_candidates(self, gray, keep=5):
        """
        Use morphological operations + Scharr gradient to find
        candidate contours that may contain a license plate.
        Returns up to `keep` contours sorted by area (largest first).
        """
        # 1. Blackhat: reveal dark characters on light plate background
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self._debug_imshow("Blackhat", blackhat)

        # 2. Find light regions (potential plate backgrounds)
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self._debug_imshow("Light Regions", light)

        # 3. Scharr gradient in x-direction on blackhat image
        #    This emphasizes the vertical edges of characters
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        if maxVal - minVal > 0:
            gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self._debug_imshow("Scharr", gradX)

        # 4. Blur + close + Otsu threshold to merge character edges
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self._debug_imshow("Grad Thresh", thresh)

        # 5. Erode + dilate to clean up
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self._debug_imshow("Erode/Dilate", thresh)

        # 6. Bitwise AND with light regions to isolate plate
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self._debug_imshow("Final", thresh)

        # Find and sort contours by area
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        return cnts

    # ------------------------------------------------------------------
    # Step 7: Pick the best plate contour by aspect ratio
    # ------------------------------------------------------------------
    def _locate_plate(self, gray, candidates, do_clear_border=True):
        """
        Loop over candidate contours, filter by aspect ratio,
        and return the license plate ROI + contour.

        Returns
        -------
        (roi, lpCnt, (x, y, w, h)) or (None, None, None)
        """
        lpCnt = None
        roi = None
        bbox = None

        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # Check aspect ratio is plate-like
            if self.minAR <= ar <= self.maxAR:
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # Clear foreground pixels touching the border (noise)
                if do_clear_border and _SKIMAGE:
                    roi = clear_border(roi)

                bbox = (x, y, w, h)
                self._debug_imshow("License Plate", licensePlate)
                self._debug_imshow("ROI", roi)
                break

        return (roi, lpCnt, bbox)

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fix_plate_chars(text):
        """
        Post-process plate text to fix common OCR letter↔digit
        confusions based on Indian plate format: XX 00 XX 0000.
        """
        clean = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if len(clean) < 4:
            return clean

        # Build a corrected version based on expected positions
        # Indian plates: 2 letters, 1-2 digits, 1-3 letters, 1-4 digits
        # Common confusions: O↔0, I↔1, S↔5, Z↔2, B↔8, G↔6
        letter_map = {'0': 'O', '1': 'I', '5': 'S', '2': 'Z', '8': 'B', '6': 'G'}
        digit_map  = {'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6'}

        result = list(clean)
        n = len(result)

        # First 2 chars should be letters (state code)
        for i in range(min(2, n)):
            if result[i] in letter_map:
                result[i] = letter_map[result[i]]

        # Next 1-2 chars should be digits (district code)
        for i in range(2, min(4, n)):
            if result[i] in digit_map:
                result[i] = digit_map[result[i]]

        # After digits, expect 1-3 letters (series)
        # Find where the letter series starts after digits
        i = 2
        while i < n and result[i].isdigit():
            i += 1
        letter_start = i
        while i < n and not result[i].isdigit():
            if result[i] in letter_map:
                result[i] = letter_map[result[i]]
            i += 1

        # Remaining should be digits (registration number)
        while i < n:
            if result[i] in digit_map:
                result[i] = digit_map[result[i]]
            i += 1

        return "".join(result)

    @staticmethod
    def _build_tesseract_options(psm=7):
        """Build Tesseract config string with alphanumeric whitelist."""
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = f"-c tessedit_char_whitelist={alphanumeric}"
        options += f" --psm {psm}"
        return options

    def _rapid_ocr(self, image, strict=False):
        """
        OCR using RapidOCR (PaddleOCR models via ONNX Runtime).
        Accepts BGR color or grayscale image.
        Tries multiple scales to get the best plate reading.

        Parameters
        ----------
        strict : bool
            If True, only return text matching a plate regex pattern.
            Use True for full-frame OCR to avoid false positives.
            Use False for ROI-based OCR (geometry already confirmed plate).
        
        Returns cleaned plate text or None.
        """
        if self._rapid is None:
            return None

        best_text = None
        best_len = 0

        # Try at 2x upscale (best for plates) — single scale for speed
        h, w = image.shape[:2]
        scales = [2]

        all_texts = []

        for scale in scales:
            if scale == 1:
                img = image
            else:
                img = cv2.resize(image, (w * scale, h * scale),
                                 interpolation=cv2.INTER_CUBIC)

            try:
                result, _ = self._rapid(img)
            except Exception:
                continue

            if not result:
                continue

            # Combine all detected text fragments
            for _box, text, score in result:
                clean = re.sub(r"[^A-Za-z0-9]", "", text).upper()
                try:
                    score = float(score)
                except ValueError:
                    score = 0.0
                if len(clean) >= 3 and score >= 0.4:
                    all_texts.append(clean)

        if not all_texts:
            return None

        # Deduplicate and sort by length (longest first)
        all_texts = list(dict.fromkeys(all_texts))  # preserve order, remove dupes
        all_texts.sort(key=len, reverse=True)

        # 1) Try RAW text first (before any char correction)
        #    This avoids corrupting already-correct OCR output
        for t in all_texts:
            for pat in self._plate_patterns:
                m = pat.search(t)
                if m:
                    candidate = re.sub(r"\s+", "", m.group(0)).upper()
                    if len(candidate) > best_len:
                        best_text = candidate
                        best_len = len(candidate)

        # 2) Try with character correction (fixes O↔0, I↔1, etc.)
        #    Keep result only if it's LONGER than raw match
        for t in all_texts:
            fixed = self._fix_plate_chars(t)
            for pat in self._plate_patterns:
                m = pat.search(fixed)
                if m:
                    candidate = re.sub(r"\s+", "", m.group(0)).upper()
                    if len(candidate) > best_len:
                        best_text = candidate
                        best_len = len(candidate)

        if best_text:
            return best_text

        # In lenient mode (ROI already confirmed as plate), accept
        # any text that has both letters + digits and >= 6 chars
        if not strict:
            for t in all_texts:
                has_letters = bool(re.search(r"[A-Z]", t))
                has_digits  = bool(re.search(r"\d", t))
                if len(t) >= 6 and has_letters and has_digits:
                    return t

        return None

    def _ocr_roi(self, roi, psm=7):
        """OCR a thresholded plate ROI. Returns cleaned text or None.
        Tries RapidOCR first, falls back to Tesseract.
        """
        if not _AVAILABLE or roi is None:
            return None

        # ── RapidOCR (primary) ──────────────────────────────────────
        text = self._rapid_ocr(roi, strict=False)
        if text:
            return text

        # ── Tesseract (fallback) ────────────────────────────────────
        if not _TESSERACT:
            return None
        
        # Resize if too small for Tesseract
        h, w = roi.shape[:2]
        if h < 40:
            scale = 60 / h
            roi = cv2.resize(roi, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        options = self._build_tesseract_options(psm=psm)
        try:
            text = pytesseract.image_to_string(roi, config=options)
        except Exception:
            return None

        text = text.strip()
        clean = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        return clean if len(clean) >= 4 else None

    # ------------------------------------------------------------------
    # Haar Cascade fallback
    # ------------------------------------------------------------------
    def _haar_detect(self, frame, gray):
        """Use Haar cascade to find plate regions as a backup."""
        results = []
        if self.plate_cascade is None:
            return results

        rects = self.plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 10)
        )
        for (x, y, w, h) in rects:
            # Add padding
            pad_w, pad_h = int(w * 0.1), int(h * 0.15)
            px = max(0, x - pad_w)
            py = max(0, y - pad_h)
            pw = min(frame.shape[1] - px, w + 2 * pad_w)
            ph = min(frame.shape[0] - py, h + 2 * pad_h)

            plate_gray = gray[py:py + ph, px:px + pw]
            roi = cv2.threshold(plate_gray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            if _SKIMAGE:
                roi = clear_border(roi)

            results.append((roi, (x, y, w, h)))

        return results

    # ------------------------------------------------------------------
    # Full-frame OCR fallback
    # ------------------------------------------------------------------
    def _ocr_full_frame(self, frame):
        """Last resort: OCR the full frame for plate-like text.
        Uses RapidOCR on the color frame first, then Tesseract on gray.
        """
        if not _AVAILABLE:
            return None

        # ── RapidOCR on full frame (primary) ──────────────────────
        text = self._rapid_ocr(frame, strict=True)
        if text:
            return text

        # If RapidOCR is available, trust its result — don't fall back
        # to Tesseract which generates false positives from random textures
        if self._rapid is not None:
            return None

        # ── Tesseract on grayscale (only if RapidOCR unavailable) ───
        if not _TESSERACT:
            return None

        gray = frame
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best = None

        for psm in [7, 8, 6]:
            options = self._build_tesseract_options(psm=psm)
            try:
                text = pytesseract.image_to_string(gray, config=options).strip()
            except Exception:
                continue

            clean = re.sub(r"[^A-Z0-9]", "", text.upper())

            # Must match a real plate pattern (strict check)
            for pat in self._plate_patterns:
                m = pat.search(clean)
                if m:
                    best = re.sub(r"\s+", "", m.group(0)).upper()
                    break
            if best:
                break

        return best

    # ------------------------------------------------------------------
    # Main detect  (public API — called by vision.py / test_image.py)
    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        Detect and OCR license plates in `frame`.

        Returns
        -------
        list[dict]  with keys ``category``, ``content``, ``bbox``
        """
        results = []

        # Resize for consistency (smaller = faster on Pi 3)
        frame_resized = imutils.resize(frame, width=480)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # ── Strategy 1: PyImageSearch morphological pipeline ─────────
        candidates = self._locate_candidates(gray)
        (roi, lpCnt, bbox) = self._locate_plate(gray, candidates)

        if roi is not None:
            # Try OCR on the detected plate ROI
            text = self._ocr_roi(roi)
            if not text:
                # Also try on the original color crop (RapidOCR prefers color)
                x, y, w, h = bbox
                color_roi = frame_resized[y:y + h, x:x + w]
                text = self._rapid_ocr(color_roi, strict=False)
            if text:
                results.append({
                    "category": "Vehicle number plate",
                    "content": text,
                    "bbox": bbox,
                })

        # ── Strategy 2: Haar Cascade (if morphological pipeline missed) ──
        if not results:
            haar_hits = self._haar_detect(frame_resized, gray)
            for (roi, bbox) in haar_hits:
                text = self._ocr_roi(roi)
                if not text:
                    # Try color ROI
                    x, y, w, h = bbox
                    color_roi = frame_resized[y:y + h, x:x + w]
                    text = self._rapid_ocr(color_roi, strict=False)
                if text:
                    results.append({
                        "category": "Vehicle number plate",
                        "content": text,
                        "bbox": bbox,
                    })
                    break

        # ── Strategy 3: Full-frame OCR fallback (skip on Pi for speed) ──
        # Only runs if the frame is small enough to be a close-up of a plate
        if not results and frame.shape[0] < 300 and frame.shape[1] < 500:
            text = self._ocr_full_frame(frame)
            if text:
                h, w = gray.shape[:2]
                results.append({
                    "category": "Vehicle number plate",
                    "content": text,
                    "bbox": (0, 0, w, h),
                })

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            if r["content"] not in seen:
                seen.add(r["content"])
                unique.append(r)

        return unique
