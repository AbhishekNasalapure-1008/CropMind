"""
CropMind — CNN Training Script (MobileNetV2 Transfer Learning)
──────────────────────────────────────────────────────────────────────────────
DESIGNED FOR: Starting with as few as 10 images per class
SELF-LEARNING: Model automatically retrains when new confirmed images are added

USAGE:
    # First time training
    python model/train_model.py

    # After adding new images (self-learning retrain)
    python model/train_model.py --retrain

    # Check dataset status
    python model/train_model.py --status

DATASET FORMAT:
    dataset/
    ├── nitrogen/       ← nitrogen deficiency images
    ├── phosphorus/     ← phosphorus deficiency images
    ├── potassium/      ← potassium deficiency images
    └── healthy/        ← healthy leaf images

    confirmed_learning/     ← auto-created, stores user-confirmed predictions
    ├── nitrogen/
    ├── phosphorus/
    ├── potassium/
    └── healthy/

SELF-LEARNING FLOW:
    1. User uploads image → model predicts
    2. User confirms or corrects the prediction in UI
    3. Image + confirmed label saved to confirmed_learning/
    4. When confirmed_learning/ has 5+ new images → auto retrain triggers
    5. Model improves over time with real-world data
"""

import os
import sys
import json
import shutil
import random
import argparse
import numpy as np
import cv2
from collections import defaultdict
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DIR       = os.path.join(_BASE_DIR, "model", "synthetic_data")
REAL_DATA_DIR       = os.path.join(_BASE_DIR, "dataset")
CONFIRMED_DATA_DIR  = os.path.join(_BASE_DIR, "confirmed_learning")
MODEL_PATH          = os.path.join(_BASE_DIR, "model", "npk_model.h5")
CLASSES_PATH        = os.path.join(_BASE_DIR, "model", "label_classes.npy")
CHECKPOINT_DIR      = os.path.join(_BASE_DIR, "model", "checkpoints")
BEST_CKPT_PATH      = os.path.join(CHECKPOINT_DIR, "best_model.h5")
HISTORY_PATH        = os.path.join(_BASE_DIR, "model", "training_history.png")
TRAINING_LOG_PATH   = os.path.join(_BASE_DIR, "model", "training_log.json")
RETRAIN_TRIGGER     = os.path.join(_BASE_DIR, "model", ".retrain_needed")

# Self-learning config
MIN_IMAGES_TO_RETRAIN = 5       # retrain after this many new confirmed images
CONFIDENCE_THRESHOLD  = 0.75    # only auto-learn from predictions above this

CLASS_LABEL_MAP = {
    "nitrogen":   [1, 0, 0],
    "phosphorus": [0, 1, 0],
    "potassium":  [0, 0, 1],
    "healthy":    [0, 0, 0],
}
IMG_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    print(f"[INFO] Seed → {seed}")


# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA  (fallback when no real data exists)
# ──────────────────────────────────────────────────────────────────────────────

def _base_leaf(size: int = IMG_SIZE) -> np.ndarray:
    img    = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = [80, 90, 60]
    center = (size // 2, size // 2)
    axes   = (size // 2 - 10, int(size * 0.45))
    angle  = np.random.randint(-15, 15)
    cv2.ellipse(img, center, axes, angle, 0, 360, (55, 140, 55), -1)
    mid_x  = size // 2 + np.random.randint(-5, 5)
    cv2.line(img, (mid_x, 15), (mid_x, size - 15), (40, 100, 40), 2)
    for i in range(4, 8):
        y      = int(size * i / 10)
        spread = int((size // 2) * 0.7)
        cv2.line(img, (mid_x, y), (mid_x - spread, y - spread // 2), (45, 110, 45), 1)
        cv2.line(img, (mid_x, y), (mid_x + spread, y - spread // 2), (45, 110, 45), 1)
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _apply_nitrogen_deficiency(img):
    out  = img.copy().astype(np.float32)
    leaf = (img[:,:,1] > img[:,:,0]) & (img[:,:,1] > img[:,:,2])
    s    = np.random.uniform(0.5, 0.9)
    out[leaf, 0] = np.clip(out[leaf, 0] + 80*s, 0, 255)
    out[leaf, 1] = np.clip(out[leaf, 1] * (1 - 0.4*s), 0, 255)
    out[leaf, 2] = np.clip(out[leaf, 2] * (1 - 0.2*s), 0, 255)
    return out.astype(np.uint8)


def _apply_phosphorus_deficiency(img):
    out  = img.copy().astype(np.float32)
    leaf = (img[:,:,1] > img[:,:,0]) & (img[:,:,1] > img[:,:,2])
    s    = np.random.uniform(0.4, 0.85)
    out[leaf, 0] = np.clip(out[leaf, 0] + 60*s, 0, 255)
    out[leaf, 1] = np.clip(out[leaf, 1] * (1 - 0.35*s), 0, 255)
    out[leaf, 2] = np.clip(out[leaf, 2] + 50*s, 0, 255)
    return out.astype(np.uint8)


def _apply_potassium_deficiency(img):
    out      = img.copy().astype(np.float32)
    size     = img.shape[0]
    s        = np.random.uniform(0.4, 0.9)
    cx, cy   = size//2, size//2
    Y, X     = np.ogrid[:size, :size]
    dist     = np.sqrt((X-cx)**2 + (Y-cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    w        = np.clip((dist/max_dist - 0.4)/0.6, 0, 1) * s
    out[:,:,0] = np.clip(out[:,:,0] + 70*w, 0, 255)
    out[:,:,1] = np.clip(out[:,:,1] - 40*w, 0, 255)
    out[:,:,2] = np.clip(out[:,:,2] - 30*w, 0, 255)
    return out.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# HEAVY AUGMENTATION  (turns 10 real images → ~200+ effective training samples)
# ──────────────────────────────────────────────────────────────────────────────

def heavy_augment_image(img: np.ndarray, n_versions: int = 50) -> list:
    """
    Generate n_versions augmented variants of a single image.
    This is the KEY function that makes 10 images trainable.

    Augmentations:
    - Rotation, flip, brightness/contrast
    - Gaussian blur (camera shake simulation)
    - Random crop + resize (zoom simulation)
    - Hue/saturation shift (lighting variation — critical for leaf color)
    - Salt & pepper noise
    - Perspective warp (angle variation)
    """
    results = []
    size    = img.shape[0]

    for _ in range(n_versions):
        out = img.copy()

        # 1. Rotation
        angle = np.random.uniform(-40, 40)
        M     = cv2.getRotationMatrix2D((size/2, size/2), angle, 1.0)
        out   = cv2.warpAffine(out, M, (size, size), borderMode=cv2.BORDER_REFLECT)

        # 2. Flips
        if np.random.rand() > 0.5:
            out = cv2.flip(out, 1)
        if np.random.rand() > 0.6:
            out = cv2.flip(out, 0)

        # 3. Brightness + contrast
        alpha = np.random.uniform(0.7, 1.4)
        beta  = np.random.uniform(-40, 40)
        out   = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 4. Gaussian blur
        if np.random.rand() > 0.6:
            k   = np.random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)

        # 5. Random crop + resize
        zoom   = np.random.uniform(0.75, 1.0)
        crop   = int(size * zoom)
        startx = np.random.randint(0, size - crop + 1)
        starty = np.random.randint(0, size - crop + 1)
        out    = out[starty:starty+crop, startx:startx+crop]
        out    = cv2.resize(out, (size, size))

        # 6. Hue + saturation shift (critical for leaf color learning)
        if np.random.rand() > 0.4:
            hsv        = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,0] = (hsv[:,:,0] + np.random.uniform(-15, 15)) % 180
            hsv[:,:,1] = np.clip(hsv[:,:,1] * np.random.uniform(0.8, 1.2), 0, 255)
            out        = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 7. Salt & pepper noise
        if np.random.rand() > 0.7:
            noise_mask             = np.random.rand(size, size)
            out[noise_mask < 0.01] = 0
            out[noise_mask > 0.99] = 255

        # 8. Perspective warp
        if np.random.rand() > 0.6:
            margin = int(size * 0.1)
            pts1   = np.float32([[0,0],[size,0],[0,size],[size,size]])
            pts2   = np.float32([
                [np.random.randint(0, margin),         np.random.randint(0, margin)],
                [np.random.randint(size-margin, size), np.random.randint(0, margin)],
                [np.random.randint(0, margin),         np.random.randint(size-margin, size)],
                [np.random.randint(size-margin, size), np.random.randint(size-margin, size)],
            ])
            M2  = cv2.getPerspectiveTransform(pts1, pts2)
            out = cv2.warpPerspective(out, M2, (size, size), borderMode=cv2.BORDER_REFLECT)

        results.append(out)

    return results


def expand_small_dataset(image_paths: list, image_labels: list,
                          target_per_class: int = 200) -> tuple:
    """
    Expand any class below target_per_class using heavy augmentation.

    With 10 real images per class → generates 190 augmented versions.
    All augmented images are saved to disk (not held in RAM).
    """
    print(f"\n[INFO] Expanding dataset → target {target_per_class} images/class...")

    class_groups = defaultdict(list)
    for path, label in zip(image_paths, image_labels):
        class_groups[tuple(label)].append((path, label))

    expanded_paths  = list(image_paths)
    expanded_labels = list(image_labels)

    for label_key, items in class_groups.items():
        current_count = len(items)
        cls_name      = _label_to_class(list(label_key))

        if current_count >= target_per_class:
            print(f"  {cls_name:<12} {current_count:>4} images — no expansion needed ✅")
            continue

        needed      = target_per_class - current_count
        augment_dir = os.path.join(SYNTHETIC_DIR, f"aug_{cls_name}")
        os.makedirs(augment_dir, exist_ok=True)

        generated    = 0
        max_attempts = needed * 10          # prevents infinite loop on unreadable images
        attempts     = 0
        while generated < needed and attempts < max_attempts:
            src_path, src_label = random.choice(items)
            img = cv2.imread(src_path)
            attempts += 1
            if img is None:
                print(f"[WARNING] Could not read {src_path} during augmentation — skipping.")
                continue
            img   = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            batch = min(50, needed - generated)
            for aug_img in heavy_augment_image(img, n_versions=batch):
                if generated >= needed:
                    break
                save_path = os.path.join(augment_dir, f"aug_{cls_name}_{generated:05d}.jpg")
                cv2.imwrite(save_path, aug_img)
                expanded_paths.append(save_path)
                expanded_labels.append(list(label_key))
                generated += 1
        if attempts >= max_attempts and generated < needed:
            print(f"[WARNING] Max augmentation attempts reached for '{cls_name}'. "
                  f"Generated {generated}/{needed} images.")

        print(f"  {cls_name:<12} {current_count:>4} real → +{generated} augmented "
              f"= {current_count + generated} total ✅")

    print(f"\n[INFO] Total after expansion: {len(expanded_paths)} images")
    return expanded_paths, expanded_labels


def _label_to_class(label: list) -> str:
    for cls, vec in CLASS_LABEL_MAP.items():
        if vec == label:
            return cls
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _detect_real_dataset() -> bool:
    if not os.path.isdir(REAL_DATA_DIR):
        return False
    for cls in CLASS_LABEL_MAP:
        folder = os.path.join(REAL_DATA_DIR, cls)
        if os.path.isdir(folder) and any(
            os.path.splitext(f)[-1].lower() in IMG_EXTS
            for f in os.listdir(folder)
        ):
            return True
    return False


def load_all_images(include_confirmed: bool = True) -> tuple:
    """
    Load images from dataset/ and optionally confirmed_learning/.
    Validates each image is readable. Logs per-class counts.
    """
    image_paths, image_labels = [], []
    class_counts = defaultdict(int)

    sources = [REAL_DATA_DIR]
    if include_confirmed and os.path.isdir(CONFIRMED_DATA_DIR):
        sources.append(CONFIRMED_DATA_DIR)

    for source_dir in sources:
        source_name = os.path.basename(source_dir)
        for cls, vec in CLASS_LABEL_MAP.items():
            folder = os.path.join(source_dir, cls)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if os.path.splitext(fname)[-1].lower() not in IMG_EXTS:
                    continue
                full_path = os.path.join(folder, fname)
                try:
                    test = cv2.imread(full_path)
                    if test is None:
                        print(f"[WARNING] Skipping unreadable: {fname}")
                        continue
                except Exception:
                    continue
                image_paths.append(full_path)
                image_labels.append(vec)
                class_counts[f"{source_name}/{cls}"] += 1

    print(f"\n[INFO] ── Dataset Breakdown ──────────────────────────")
    for key, count in sorted(class_counts.items()):
        bar = "█" * min(count, 30)
        print(f"  {key:<35} {count:>4}  {bar}")
    print(f"  {'TOTAL':<35} {len(image_paths):>4}")
    print(f"[INFO] ──────────────────────────────────────────────\n")

    return image_paths, image_labels


# ──────────────────────────────────────────────────────────────────────────────
# SELF-LEARNING: SAVE CONFIRMED PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def save_confirmed_image(image_path: str, confirmed_class: str) -> bool:
    """
    Called by app.py when user confirms or corrects a prediction.

    Saves image to confirmed_learning/<class>/ and checks
    if enough new images exist to trigger retraining.

    Returns True if retraining should be triggered.
    """
    if confirmed_class not in CLASS_LABEL_MAP:
        print(f"[ERROR] Unknown class: {confirmed_class}")
        return False

    dest_dir  = os.path.join(CONFIRMED_DATA_DIR, confirmed_class)
    os.makedirs(dest_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest_path = os.path.join(dest_dir, f"confirmed_{timestamp}.jpg")

    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(dest_path, img)
        print(f"[LEARNING] ✅ Saved → confirmed_learning/{confirmed_class}/")
    except Exception as e:
        print(f"[ERROR] Could not save confirmed image: {e}")
        return False

    new_count = _count_new_confirmed_images()
    print(f"[LEARNING] Confirmed images since last retrain: {new_count}/{MIN_IMAGES_TO_RETRAIN}")

    if new_count >= MIN_IMAGES_TO_RETRAIN:
        with open(RETRAIN_TRIGGER, "w") as f:
            f.write(str(new_count))
        print(f"[LEARNING] 🔄 Threshold reached! Run: python model/train_model.py --retrain")
        return True

    return False


def _count_new_confirmed_images() -> int:
    if not os.path.isdir(CONFIRMED_DATA_DIR):
        return 0
    count = 0
    for cls in CLASS_LABEL_MAP:
        folder = os.path.join(CONFIRMED_DATA_DIR, cls)
        if not os.path.isdir(folder):
            continue
        count += sum(
            1 for f in os.listdir(folder)
            if os.path.splitext(f)[-1].lower() in IMG_EXTS
        )
    return count


def check_retrain_needed() -> bool:
    return os.path.exists(RETRAIN_TRIGGER)


def clear_retrain_flag():
    if os.path.exists(RETRAIN_TRIGGER):
        os.remove(RETRAIN_TRIGGER)


# ──────────────────────────────────────────────────────────────────────────────
# STRATIFIED SPLIT  (safe for small datasets)
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(image_paths: list, image_labels: list,
                     val_ratio: float = 0.2) -> tuple:
    """
    Stratified split — each class proportionally in both train and val.
    Safe for 1-image classes (puts them in train only).
    """
    class_indices: defaultdict[tuple, list[int]] = defaultdict(list)
    for i, label in enumerate(image_labels):
        class_indices[tuple(label)].append(i)

    train_idx, val_idx = [], []
    for label_key, indices in class_indices.items():
        idx_list: list[int] = list(indices)
        random.shuffle(idx_list)
        if len(idx_list) <= 1:
            train_idx.extend(idx_list)
            continue
        split = max(1, int(len(idx_list) * (1 - val_ratio)))
        train_idx.extend(idx_list[:split])
        val_idx.extend(idx_list[split:])

    random.shuffle(train_idx)
    random.shuffle(val_idx)

    # Guard: if val set is empty (e.g. every class has only 1 image),
    # fall back to a small slice from train so tf.data doesn't crash.
    if not val_idx:
        print("[WARNING] Not enough images for a validation split — using 10% of train as val.")
        fallback_n = max(1, len(train_idx) // 10)
        val_idx    = train_idx[-fallback_n:]
        train_idx  = train_idx[:-fallback_n]

    return (
        [image_paths[i] for i in train_idx],
        [image_labels[i] for i in train_idx],
        [image_paths[i] for i in val_idx],
        [image_labels[i] for i in val_idx],
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(image_labels: list) -> dict:
    # Guard: if no labels exist, return balanced weights
    if not image_labels:
        print("[WARNING] No training labels found — using default class weights (1.0).")
        return {0: 1.0, 1: 1.0, 2: 1.0}

    labels_array = np.array(image_labels)  # shape: (N, 3)
    # Ensure 2-D even when N==1
    if labels_array.ndim == 1:
        labels_array = labels_array.reshape(1, -1)

    n_total = len(labels_array)
    weights = {}
    for i, cls in enumerate(["nitrogen", "phosphorus", "potassium"]):
        pos        = labels_array[:, i].sum()
        weights[i] = n_total / (2.0 * pos) if pos > 0 else 1.0
    print("[INFO] Class weights: " +
          " | ".join(f"{cls}={weights[i]:.2f}"
                     for i, cls in enumerate(["N", "P", "K"])))
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# tf.data PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def build_tf_dataset(image_paths: list, image_labels: list,
                     batch_size: int = 8,
                     augment: bool = False,
                     shuffle: bool = True):
    """
    Memory-efficient tf.data pipeline.
    batch_size=8 default (appropriate for small datasets).
    FIX: tf.ensure_shape after decode_image prevents static shape crash.
    FIX: MobileNetV2 preprocessing (-1 to 1 scaling).
    """
    import tensorflow as tf

    paths_tensor  = tf.constant(image_paths,  dtype=tf.string)
    labels_tensor = tf.constant(image_labels, dtype=tf.float32)

    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.ensure_shape(img, [None, None, 3])      # FIX: force known rank
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - 0.5) * 2.0                          # MobileNetV2 scale: -1 to 1
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.clip_by_value(img, -1.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 500),
                        reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# MODEL  (optimized for small dataset)
# ──────────────────────────────────────────────────────────────────────────────

def build_model(freeze_base: bool = True):
    """
    MobileNetV2 transfer learning for small datasets.

    Small dataset design choices:
    - L2 regularization on Dense layers → prevent overfitting
    - Higher Dropout (0.5/0.4) → prevent overfitting
    - BatchNormalization → stabilize small-batch training
    - Base FULLY FROZEN when dataset < 200 images
      (ImageNet leaf features are good enough; fine-tuning on 10 imgs = overfit)
    - Fine-tuning ONLY unlocked at 200+ real images
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models, regularizers

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="leaf_input")
    x      = base(inputs, training=False)
    x      = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5, name="drop_256")(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), name="dense_128")(x)
    x = layers.BatchNormalization(name="bn_128")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4, name="drop_128")(x)

    outputs = layers.Dense(3, activation="sigmoid", name="npk_output")(x)

    model = models.Model(inputs, outputs, name="CropMind_NPK")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING HISTORY PLOT  (important for hackathon presentation)
# ──────────────────────────────────────────────────────────────────────────────

def save_training_plot(history_data: dict, save_path: str = HISTORY_PATH):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0A0F0A")
        for ax in axes:
            ax.set_facecolor("#111811")

        fig.suptitle("CropMind — Training History", fontsize=14,
                     fontweight="bold", color="#4ADE80")

        axes[0].plot(history_data.get("loss", []),     label="Train", color="#4ADE80", linewidth=2)
        axes[0].plot(history_data.get("val_loss", []), label="Val",   color="#FCD34D", linewidth=2, linestyle="--")
        axes[0].set_title("Loss", color="#F0FDF4")
        axes[0].set_xlabel("Epoch", color="#6B7280")
        axes[0].tick_params(colors="#6B7280")
        axes[0].legend(facecolor="#161E16", labelcolor="#F0FDF4")
        axes[0].grid(alpha=0.2, color="#1F2B1F")

        axes[1].plot(history_data.get("accuracy", []),     label="Train", color="#4ADE80", linewidth=2)
        axes[1].plot(history_data.get("val_accuracy", []), label="Val",   color="#FCD34D", linewidth=2, linestyle="--")
        axes[1].set_title("Accuracy", color="#F0FDF4")
        axes[1].set_xlabel("Epoch", color="#6B7280")
        axes[1].tick_params(colors="#6B7280")
        axes[1].legend(facecolor="#161E16", labelcolor="#F0FDF4")
        axes[1].grid(alpha=0.2, color="#1F2B1F")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0A0F0A")
        plt.close()
        print(f"[INFO] Training plot → {save_path}")
    except Exception as e:
        print(f"[WARNING] Could not save training plot: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOG
# ──────────────────────────────────────────────────────────────────────────────

def save_training_log(metrics: dict, dataset_size: int, mode: str = "initial"):
    log = []
    if os.path.exists(TRAINING_LOG_PATH):
        try:
            with open(TRAINING_LOG_PATH) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append({
        "timestamp":    datetime.now().isoformat(),
        "mode":         mode,
        "dataset_size": dataset_size,
        "metrics":      metrics,
    })
    os.makedirs(os.path.dirname(TRAINING_LOG_PATH), exist_ok=True)
    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Training log → {TRAINING_LOG_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# STATUS CHECK
# ──────────────────────────────────────────────────────────────────────────────

def print_status():
    print("\n" + "="*55)
    print("  CropMind — Dataset & Model Status")
    print("="*55)
    for cls in CLASS_LABEL_MAP:
        folder = os.path.join(REAL_DATA_DIR, cls)
        count  = 0
        if os.path.isdir(folder):
            count = sum(1 for f in os.listdir(folder)
                        if os.path.splitext(f)[-1].lower() in IMG_EXTS)
        status = "✅" if count >= 10 else ("⚠️ " if count > 0 else "❌")
        note   = " (good to train)" if count >= 10 else (" (add more)" if count > 0 else " (missing)")
        print(f"  dataset/{cls:<12} {status}  {count:>3} images{note}")

    print()
    total_confirmed = _count_new_confirmed_images()
    print(f"  confirmed_learning/    {total_confirmed} new images")
    print(f"  Retrain threshold:     {MIN_IMAGES_TO_RETRAIN} images")
    print(f"  Retrain needed:        {'YES — run --retrain ⚠️' if check_retrain_needed() else 'No'}")
    print()
    model_exists = os.path.exists(MODEL_PATH)
    print(f"  npk_model.h5:          {'✅ Exists' if model_exists else '❌ Not trained yet'}")
    if os.path.exists(TRAINING_LOG_PATH):
        try:
            with open(TRAINING_LOG_PATH) as f:
                log = json.load(f)
            last = log[-1]
            print(f"  Last trained:          {last['timestamp'][:19]}")
            print(f"  Training runs total:   {len(log)}")
            m = last.get("metrics", {})
            print(f"  Last val accuracy:     {m.get('val_accuracy', 'N/A')}")
        except Exception:
            pass
    print("="*55 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def train(mode: str = "initial", seed: int = 42):
    import tensorflow as tf

    set_seeds(seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SYNTHETIC_DIR,  exist_ok=True)

    print("\n" + "="*55)
    print(f"  CropMind — NPK Model Training  [{mode.upper()}]")
    print("="*55)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if _detect_real_dataset():
        image_paths, image_labels = load_all_images(include_confirmed=True)
    else:
        print("[INFO] No real dataset — using synthetic fallback.")
        image_paths, image_labels = _generate_synthetic_fallback()

    total_real = len(image_paths)

    # ── 2. Expand small dataset ───────────────────────────────────────────────
    # 10 real images → 200 effective training samples per class
    image_paths, image_labels = expand_small_dataset(
        image_paths, image_labels, target_per_class=200
    )

    # ── 3. Stratified split ───────────────────────────────────────────────────
    train_paths, train_labels, val_paths, val_labels = stratified_split(
        image_paths, image_labels, val_ratio=0.2
    )

    # ── 4. Class weights ──────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_labels)

    # ── 5. tf.data pipelines ──────────────────────────────────────────────────
    batch_size = 8   # small batch for small dataset
    train_ds   = build_tf_dataset(train_paths, train_labels,
                                   batch_size=batch_size, augment=True,  shuffle=True)
    val_ds     = build_tf_dataset(val_paths,   val_labels,
                                   batch_size=batch_size, augment=False, shuffle=False)

    # ── 6. Build or load model ────────────────────────────────────────────────
    if mode == "retrain" and os.path.exists(MODEL_PATH):
        print("[INFO] Loading existing model for incremental retrain...")
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # lower LR
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        # Freeze base if dataset < 200 real images (prevents overfit)
        should_freeze = total_real < 200
        print(f"[INFO] Real images: {total_real} → base {'FROZEN' if should_freeze else 'TRAINABLE'}")
        model = build_model(freeze_base=should_freeze)

    model.summary(line_length=65)

    # ── 7. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, verbose=1, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_CKPT_PATH,
            monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    # ── 8. Train ──────────────────────────────────────────────────────────────
    epochs = 60 if mode == "initial" else 30
    print(f"\n[Training] epochs={epochs} | batch={batch_size} | "
          f"train={len(train_paths)} | val={len(val_paths)}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 9. Fine-tune phase (only when dataset grows to 200+ real images) ──────
    if total_real >= 200 and mode == "initial":
        print("\n[Phase 2] Dataset large enough — fine-tuning top layers...")
        base_model = next(
            (l for l in model.layers if "mobilenetv2" in l.name.lower()), None
        )
        if base_model:
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss="binary_crossentropy",
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.AUC(name="auc"),
                ],
            )
            model.fit(
                train_ds, validation_data=val_ds,
                epochs=20, callbacks=callbacks, verbose=1,
            )

    # ── 10. Save ──────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    np.save(CLASSES_PATH, np.array(["nitrogen", "phosphorus", "potassium"]))
    print(f"\n[✅] Model   → {MODEL_PATH}")
    print(f"[✅] Classes → {CLASSES_PATH}")

    # ── 11. Plot + log ────────────────────────────────────────────────────────
    save_training_plot(history.history)
    results      = model.evaluate(val_ds, verbose=0)
    metrics_dict = dict(zip(["val_loss", "val_accuracy", "val_auc"], results))
    print("\n[RESULTS] " + " | ".join(f"{k}={v:.4f}" for k,v in metrics_dict.items()))
    save_training_log(metrics_dict, dataset_size=total_real, mode=mode)

    # ── 12. Clear retrain flag ─────────────────────────────────────────────────
    if mode == "retrain":
        clear_retrain_flag()
        print("[LEARNING] ✅ Retrain complete.")

    print("\n" + "="*55)
    print("  NEXT STEPS:")
    print(f"  • Add more real images to dataset/ folders")
    print(f"  • Run app and let users confirm predictions")
    print(f"  • After {MIN_IMAGES_TO_RETRAIN}+ confirmations:")
    print(f"    python model/train_model.py --retrain")
    print(f"  • Model improves with every retrain cycle")
    print("="*55 + "\n")


def _generate_synthetic_fallback():
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    fns    = [_apply_nitrogen_deficiency, _apply_phosphorus_deficiency, _apply_potassium_deficiency]
    paths  = []
    labels = []
    for idx, fn in enumerate(fns):
        for i in range(50):
            base = _base_leaf(IMG_SIZE)
            img  = fn(base)
            p    = os.path.join(SYNTHETIC_DIR, f"syn_{idx}_{i:04d}.jpg")
            cv2.imwrite(p, img)
            lbl       = [0, 0, 0]
            lbl[idx]  = 1
            paths.append(p)
            labels.append(lbl)
    for i in range(25):
        base = _base_leaf(IMG_SIZE)
        p    = os.path.join(SYNTHETIC_DIR, f"syn_healthy_{i:04d}.jpg")
        cv2.imwrite(p, base)
        paths.append(p)
        labels.append([0, 0, 0])
    return paths, labels


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CropMind Training Script")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain with newly confirmed images")
    parser.add_argument("--status",  action="store_true",
                        help="Show dataset and model status")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.retrain:
        train(mode="retrain", seed=args.seed)
    else:
        train(mode="initial", seed=args.seed)