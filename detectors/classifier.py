"""
=============================================================================
 TFLite Image Classifier
=============================================================================
Loads a TensorFlow Lite model trained to classify the competition categories:
  Brand logo, Vehicle, Furniture, Pets, Smart switch, Parcel.

The labels.txt file should have one label per line in the format:
    Category/Content
e.g.
    Brand logo/Maybach
    Vehicle/Car
    Furniture/Chair
    Pets/Cat
    Smart switch/Smart Console
    Parcel/Parcel
=============================================================================
"""

import os
import numpy as np
import cv2

# TFLite runtime — try ai-edge-litert (new), then tflite-runtime (legacy),
# then full TensorFlow as last resort.
try:
    from ai_edge_litert import interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite


class TFLiteClassifier:
    """Run inference on a single frame and return (category, content, confidence)."""

    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.55):
        """
        Parameters
        ----------
        model_path : str
            Path to the .tflite model file.
        labels_path : str
            Path to labels.txt (one "Category/Content" per line).
        confidence_threshold : float
            Minimum softmax score to report a detection.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TFLite model not found at {model_path}. "
                "Train your model first — see train/train_classifier.py"
            )

        self._threshold = confidence_threshold

        # Load labels
        with open(labels_path, "r", encoding="utf-8") as f:
            self._labels = [line.strip() for line in f if line.strip()]

        # Load interpreter
        self._interpreter = tflite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Expected input shape [1, H, W, C]
        self._input_shape = self._input_details[0]["shape"]  # e.g. [1, 224, 224, 3]
        self._input_h = self._input_shape[1]
        self._input_w = self._input_shape[2]
        self._is_float = self._input_details[0]["dtype"] == np.float32

    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_low_quality(frame):
        """Enhance low-quality ESP32-CAM images before classification."""
        # CLAHE on L channel for contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        # Light denoise (fast bilateral)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        return enhanced

    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        Parameters
        ----------
        frame : np.ndarray — BGR image from OpenCV.

        Returns
        -------
        list[dict] — each dict has keys "category", "content", "confidence".
                     Multiple results are possible if several classes exceed
                     the threshold (for "Multiple categories" challenge).
        """
        # Pre-process: enhance low-quality image, then convert
        enhanced = self._preprocess_low_quality(frame)
        img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._input_w, self._input_h))

        if self._is_float:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.uint8)

        input_data = np.expand_dims(img, axis=0)

        # Run inference
        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()
        output_data = self._interpreter.get_tensor(self._output_details[0]["index"])[0]

        # Softmax scores
        if output_data.dtype != np.float32:
            output_data = output_data.astype(np.float32) / 255.0

        results = []
        for idx, score in enumerate(output_data):
            if idx < len(self._labels):
                label = self._labels[idx]
                # Parse "Category/Content"
                if "/" in label:
                    category, content = label.split("/", 1)
                else:
                    category, content = label, label
                results.append({
                    "category": category.strip(),
                    "content": content.strip(),
                    "confidence": float(score),
                    "bbox": None,
                })

        # Sort by confidence descending — vision.py picks top-N
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results
