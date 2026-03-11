"""
=============================================================================
 Training Script  —  train/train_classifier.py
=============================================================================
Trains a lightweight image classifier and exports it to TensorFlow Lite
format (.tflite) for on-device inference on the Raspberry Pi.

Usage:
    python train/train_classifier.py --data_dir ./merged_dataset --epochs 15
    python train/train_classifier.py --data_dir ./merged_dataset --epochs 15 --test

Dataset directory structure (ImageFolder style):
    merged_dataset/
    ├── Brand logo/
    │   ├── Apple/  …
    │   └── Tesla/  …
    ├── Vehicles/
    │   ├── bicycle/ …
    │   └── cars/    …
    └── …

Each sub-subfolder name becomes the label: "Category/Content"
  e.g.  Brand logo/Apple,  Vehicles/cars,  Pets/cats

The script:
  1. Scans the two-level directory and builds (path, label) pairs.
  2. Splits 85 / 15 train / val, computes class weights for balance.
  3. Uses tf.data pipelines (no RAM explosion for large datasets).
  4. Fine-tunes MobileNetV2 (very lightweight, ~3 MB) with a custom head.
  5. Converts to TFLite with float16 quantisation.
  6. Writes  models/classifier.tflite  and  models/labels.txt.
  7. (--test) Evaluates the TFLite model on the full dataset and prints
     per-class precision / recall / accuracy.
=============================================================================
"""

import os
import sys
import random
import argparse
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ── Config ───────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".avif", ".tif", ".tiff",
)


# =====================================================================
# 1.  Dataset scanning
# =====================================================================
def build_label_map(data_dir):
    """
    Walk two levels: data_dir / Category / Content / *.{img}
    Return a flat mapping  "Category/Content" → list of image paths.
    """
    label_to_paths = {}
    for category in sorted(os.listdir(data_dir)):
        cat_path = os.path.join(data_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for content in sorted(os.listdir(cat_path)):
            content_path = os.path.join(cat_path, content)
            if not os.path.isdir(content_path):
                continue
            label = f"{category}/{content}"
            imgs = [
                os.path.join(content_path, f)
                for f in os.listdir(content_path)
                if f.lower().endswith(IMG_EXTENSIONS)
            ]
            if imgs:
                label_to_paths[label] = imgs
    return label_to_paths


# =====================================================================
# 2.  tf.data pipeline  (streams from disk — no RAM explosion)
# =====================================================================
def _load_image_pillow(path_bytes, img_size=IMG_SIZE):
    """Load any image format via Pillow — much more robust than tf.image."""
    from PIL import Image
    path_str = path_bytes.numpy().decode("utf-8")
    try:
        img = Image.open(path_str).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
    except Exception:
        # Return a blank image if anything goes wrong
        arr = np.zeros((img_size, img_size, 3), dtype=np.float32)
    return arr


def _parse_image(path, label):
    """Wrapper that calls Pillow loader via tf.py_function."""
    img = tf.py_function(
        _load_image_pillow, [path], tf.float32
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, label


def _augment(img, label):
    """On-the-fly augmentation for training split."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_datasets(label_to_paths, val_split=0.15, seed=42):
    """Return (train_ds, val_ds, label_names, class_weights)."""
    label_names = sorted(label_to_paths.keys())
    label_idx = {name: i for i, name in enumerate(label_names)}

    all_paths, all_labels = [], []
    for label, paths in label_to_paths.items():
        idx = label_idx[label]
        for p in paths:
            all_paths.append(p)
            all_labels.append(idx)

    # Shuffle deterministically
    combined = list(zip(all_paths, all_labels))
    random.seed(seed)
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)

    n = len(all_paths)
    n_val = int(n * val_split)
    n_train = n - n_val

    train_paths, train_labels = list(all_paths[:n_train]), list(all_labels[:n_train])
    val_paths, val_labels = list(all_paths[n_train:]), list(all_labels[n_train:])

    # Class weights to handle imbalance (e.g. bicycle=6070 vs Apple=4)
    counts = Counter(train_labels)
    total = sum(counts.values())
    num_classes = len(label_names)
    class_weights = {
        cls: total / (num_classes * cnt) for cls, cnt in counts.items()
    }

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(min(n_train, 10000), seed=seed)
        .map(_parse_image, num_parallel_calls=AUTOTUNE)
        .map(_augment, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(_parse_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, label_names, class_weights


# =====================================================================
# 3.  Model
# =====================================================================
def build_model(num_classes):
    """MobileNetV2 with a custom classification head."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze most layers; fine-tune last 30
    for layer in base.layers[:-30]:
        layer.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


# =====================================================================
# 4.  Training
# =====================================================================
def train(data_dir, epochs, output_dir, run_test=False):
    print(f"\n[TRAIN] Scanning dataset at: {data_dir}")
    label_to_paths = build_label_map(data_dir)
    if not label_to_paths:
        print("[TRAIN] ERROR — no images found.  Check your directory structure.")
        sys.exit(1)

    for label, paths in label_to_paths.items():
        print(f"  {label:40s}  →  {len(paths)} images")

    train_ds, val_ds, label_names, class_weights = make_datasets(label_to_paths)
    num_classes = len(label_names)
    total_images = sum(len(p) for p in label_to_paths.values())
    print(f"\n[TRAIN] {total_images} images, {num_classes} classes\n")

    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        ],
    )

    # ── Export to TFLite ─────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    model_path = os.path.join(output_dir, "classifier.tflite")
    labels_path = os.path.join(output_dir, "labels.txt")

    with open(model_path, "wb") as f:
        f.write(tflite_model)

    with open(labels_path, "w", encoding="utf-8") as f:
        for name in label_names:
            f.write(name + "\n")

    print(f"\n[TRAIN] Model saved  → {model_path} ({len(tflite_model)/1024:.0f} KB)")
    print(f"[TRAIN] Labels saved → {labels_path}  ({num_classes} classes)")
    print("[TRAIN] Done ✓\n")

    # ── Evaluate on full dataset if requested ────────────────────
    if run_test:
        test_tflite_model(model_path, labels_path, data_dir)


# =====================================================================
# 5.  Testing / Evaluation  (uses the exported TFLite model)
# =====================================================================
def test_tflite_model(model_path, labels_path, data_dir):
    """Load the TFLite model and evaluate on every image in data_dir."""
    import cv2

    print("\n" + "=" * 70)
    print("  TESTING — Evaluating TFLite model on dataset")
    print("=" * 70)

    # Load labels
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_h = input_details[0]["shape"][1]
    input_w = input_details[0]["shape"][2]
    is_float = input_details[0]["dtype"] == np.float32

    # Build label map from dataset
    label_to_paths = build_label_map(data_dir)

    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    per_class_predicted = Counter()
    misclassified_examples = []  # store up to 5 per class

    for true_label, paths in sorted(label_to_paths.items()):
        for p in paths:
            try:
                img = cv2.imread(p)
                if img is None:
                    # Try with Pillow for unsupported formats
                    from PIL import Image
                    pil_img = Image.open(p).convert("RGB")
                    img = np.array(pil_img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if img is None:
                    continue

                # Pre-process
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (input_w, input_h))

                if is_float:
                    img_input = img_resized.astype(np.float32) / 255.0
                else:
                    img_input = img_resized.astype(np.uint8)

                input_data = np.expand_dims(img_input, axis=0)

                # Run inference
                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]["index"])[0]

                predicted_idx = int(np.argmax(output_data))
                confidence = float(output_data[predicted_idx])
                predicted_label = labels[predicted_idx]

                per_class_total[true_label] += 1
                per_class_predicted[predicted_label] += 1
                total += 1

                if predicted_label == true_label:
                    correct += 1
                    per_class_correct[true_label] += 1
                else:
                    if len([m for m in misclassified_examples if m[0] == true_label]) < 3:
                        misclassified_examples.append(
                            (true_label, predicted_label, confidence, os.path.basename(p))
                        )
            except Exception as e:
                print(f"  [WARN] Could not process {p}: {e}")

    # ── Print results ────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  {'Class':<40s}  {'Acc':>6s}  {'Correct':>8s}  {'Total':>6s}  {'Prec':>6s}")
    print(f"{'─'*78}")
    for label in sorted(per_class_total.keys()):
        t = per_class_total[label]
        c = per_class_correct.get(label, 0)
        acc = c / t * 100 if t > 0 else 0
        # Precision: of all images predicted as this class, how many were correct?
        pred_total = per_class_predicted.get(label, 0)
        prec = c / pred_total * 100 if pred_total > 0 else 0
        print(f"  {label:<40s}  {acc:5.1f}%  {c:>8d}  {t:>6d}  {prec:5.1f}%")
    print(f"{'─'*78}")
    overall_acc = correct / total * 100 if total > 0 else 0
    print(f"  {'OVERALL':<40s}  {overall_acc:5.1f}%  {correct:>8d}  {total:>6d}")
    print(f"{'─'*78}")

    # Show some misclassified examples
    if misclassified_examples:
        print(f"\n  Sample misclassifications (up to 3 per class):")
        print(f"  {'True Label':<30s}  {'Predicted':<30s}  {'Conf':>5s}  File")
        print(f"  {'─'*90}")
        for true, pred, conf, fname in misclassified_examples[:30]:
            print(f"  {true:<30s}  {pred:<30s}  {conf:4.1%}  {fname}")

    print(f"\n[TEST] Evaluation complete ✓\n")
    return overall_acc


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier for rover")
    parser.add_argument("--data_dir", required=True, help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--test", action="store_true",
                        help="Evaluate TFLite model on dataset after training")
    parser.add_argument("--test_only", action="store_true",
                        help="Only run evaluation (skip training)")
    parser.add_argument("--output_dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    ))
    args = parser.parse_args()

    if args.test_only:
        model_path = os.path.join(args.output_dir, "classifier.tflite")
        labels_path = os.path.join(args.output_dir, "labels.txt")
        test_tflite_model(model_path, labels_path, args.data_dir)
    else:
        train(args.data_dir, args.epochs, args.output_dir, run_test=args.test)
