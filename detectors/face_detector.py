"""
=============================================================================
 Face Recognition Detector  —  Pure OpenCV (no dlib)
=============================================================================
Uses OpenCV's built-in YuNet face detector + SFace recogniser.
Zero extra pip dependencies — works with opencv-python-headless >= 4.8.

Models (auto-downloaded on first run, ~37 MB total):
  models/face_detection_yunet_2023mar.onnx   (~230 KB)
  models/face_recognition_sface_2021dec.onnx (~37 MB)

Setup:
  Put one clear JPEG per person into the known_faces/ directory.
  Name the file as the person's name with underscores replacing spaces:
      known_faces/Roger_Federer.jpg
      known_faces/Keanu_Reeves.jpg
      known_faces/Henry_Cavill.jpg
=============================================================================
"""

import os
import cv2
import numpy as np
import urllib.request

# ── Model URLs (OpenCV Zoo on GitHub) ────────────────────────────────────
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


def _ensure_model(url: str, save_path: str) -> str:
    """Download model file if it doesn't exist yet."""
    if os.path.isfile(save_path):
        return save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[FACE] Downloading {os.path.basename(save_path)} ...")
    urllib.request.urlretrieve(url, save_path)
    print(f"[FACE] Saved to {save_path}")
    return save_path


class FaceDetector:
    """Identify known faces using OpenCV YuNet + SFace."""

    # Cosine-similarity threshold — higher = more lenient
    # SFace recommended: 0.363 (strict) to 0.40 (lenient)
    COSINE_THRESHOLD = 0.35
    # L2-norm threshold — lower = stricter
    # SFace recommended: 1.128 (strict) to 1.30 (lenient)
    L2_THRESHOLD = 1.20

    def __init__(self, known_faces_dir: str):
        self._known_features = []   # list of 128-d feature vectors
        self._known_names = []
        self._detector = None
        self._recogniser = None

        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

        # --- Load YuNet face detector ---
        try:
            det_path = _ensure_model(
                _YUNET_URL,
                os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"),
            )
            self._detector = cv2.FaceDetectorYN.create(
                det_path, "",
                (320, 320),
                score_threshold=0.5,    # lower = detect more faces (ESP32-CAM quality)
                nms_threshold=0.3,
                top_k=10,
            )
        except Exception as exc:
            print(f"[FACE] Could not load YuNet detector: {exc}")
            return

        # --- Load SFace recogniser ---
        try:
            rec_path = _ensure_model(
                _SFACE_URL,
                os.path.join(models_dir, "face_recognition_sface_2021dec.onnx"),
            )
            self._recogniser = cv2.FaceRecognizerSF.create(rec_path, "")
        except Exception as exc:
            print(f"[FACE] Could not load SFace recogniser: {exc}")
            return

        # --- Encode known faces ---
        if not os.path.isdir(known_faces_dir):
            print(f"[WARN] Known faces directory not found: {known_faces_dir}")
            return

        for fname in sorted(os.listdir(known_faces_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(known_faces_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            features = self._extract_features(img)
            if features:
                name = os.path.splitext(fname)[0].replace("_", " ")
                for feat in features:
                    self._known_features.append(feat)
                    self._known_names.append(name)
                print(f"[FACE] Loaded known face: {name} ({len(features)} encoding(s))")

        print(f"[FACE] {len(set(self._known_names))} known person(s) loaded "
              f"({len(self._known_features)} total encodings).")

    # ------------------------------------------------------------------
    def _preprocess(self, img):
        """Enhance image for better face detection on low-quality cameras."""
        # CLAHE on luminance channel for better contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    def _detect_faces(self, img):
        """Detect faces in image, returns faces array or None."""
        h, w = img.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(img)
        return faces

    # ------------------------------------------------------------------
    def _extract_features(self, img):
        """Extract feature vectors for ALL faces in img.
        Tries original + enhanced version for robustness."""
        features = []
        seen = set()

        for proc_img in [img, self._preprocess(img)]:
            faces = self._detect_faces(proc_img)
            if faces is None:
                continue
            for face in faces:
                aligned = self._recogniser.alignCrop(proc_img, face)
                feat = self._recogniser.feature(aligned)
                # Deduplicate by rounding
                key = tuple(np.round(feat.flatten(), 2))
                if key not in seen:
                    seen.add(key)
                    features.append(feat)

        return features

    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        Parameters
        ----------
        frame : np.ndarray — BGR image.

        Returns
        -------
        list[dict] with keys "category", "content", "confidence", "bbox"
        """
        if self._detector is None or self._recogniser is None:
            return []
        if not self._known_features:
            return []

        # Try both original and enhanced frames
        results = []
        found_names = set()

        for img in [frame, self._preprocess(frame)]:
            faces = self._detect_faces(img)
            if faces is None:
                continue

            for face in faces:
                aligned = self._recogniser.alignCrop(img, face)
                feature = self._recogniser.feature(aligned)

                # Compare against all known faces using BOTH metrics
                best_cosine = -1.0
                best_l2 = 999.0
                best_name = "Unknown"

                for known_feat, known_name in zip(self._known_features, self._known_names):
                    cos_score = self._recogniser.match(
                        feature, known_feat, cv2.FaceRecognizerSF_FR_COSINE
                    )
                    l2_score = self._recogniser.match(
                        feature, known_feat, cv2.FaceRecognizerSF_FR_NORM_L2
                    )

                    # Accept if EITHER metric passes threshold
                    cosine_ok = cos_score >= self.COSINE_THRESHOLD
                    l2_ok = l2_score <= self.L2_THRESHOLD

                    if (cosine_ok or l2_ok) and cos_score > best_cosine:
                        best_cosine = cos_score
                        best_l2 = l2_score
                        best_name = known_name

                if best_name != "Unknown" and best_name not in found_names:
                    found_names.add(best_name)
                    x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    results.append({
                        "category": "Face recognition",
                        "content": best_name,
                        "confidence": round(best_cosine, 3),
                        "bbox": (x, y, fw, fh),
                    })

            # If we found faces in original, skip enhanced to save time
            if results:
                break

        return results
