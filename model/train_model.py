"""
CropMind — CNN Training Script (MobileNetV2 Transfer Learning)
Generates synthetic training data if no real dataset is available and trains the model.

USAGE:
  python model/train_model.py

This will:
  1. Generate synthetic NPK deficiency images (300 per class)
  2. Train a MobileNetV2-based model
  3. Save the trained model as model/npk_model.h5
"""

import os
import sys
import numpy as np
import cv2

# Path setup
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DIR = os.path.join(_BASE_DIR, "model", "synthetic_data")
MODEL_PATH    = os.path.join(_BASE_DIR, "model", "npk_model.h5")
CLASSES_PATH  = os.path.join(_BASE_DIR, "model", "label_classes.npy")

# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def _base_leaf(size=224) -> np.ndarray:
    """Generate a realistic-looking green leaf base using OpenCV drawing."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Background (soil/field)
    img[:] = [80, 90, 60]

    # Leaf shape (ellipse)
    center = (size // 2, size // 2)
    axes = (size // 2 - 10, int(size * 0.45))
    angle = np.random.randint(-15, 15)
    cv2.ellipse(img, center, axes, angle, 0, 360, (55, 140, 55), -1)

    # Midrib
    mid_x = size // 2 + np.random.randint(-5, 5)
    cv2.line(img, (mid_x, 15), (mid_x, size - 15), (40, 100, 40), 2)

    # Veins
    for i in range(4, 8):
        y = int(size * i / 10)
        spread = int((size // 2) * 0.7)
        cv2.line(img, (mid_x, y), (mid_x - spread, y - spread // 2), (45, 110, 45), 1)
        cv2.line(img, (mid_x, y), (mid_x + spread, y - spread // 2), (45, 110, 45), 1)

    # Noise texture
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _apply_nitrogen_deficiency(img: np.ndarray) -> np.ndarray:
    """Yellow tint — reduce green channel, boost red channel."""
    out = img.copy().astype(np.float32)
    # Only affect leaf pixels (green dominant)
    leaf_mask = (img[:, :, 1] > img[:, :, 0]) & (img[:, :, 1] > img[:, :, 2])
    strength = np.random.uniform(0.5, 0.9)
    out[leaf_mask, 0] = np.clip(out[leaf_mask, 0] + 80 * strength, 0, 255)   # boost R
    out[leaf_mask, 1] = np.clip(out[leaf_mask, 1] * (1 - 0.4 * strength), 0, 255)  # reduce G
    out[leaf_mask, 2] = np.clip(out[leaf_mask, 2] * (1 - 0.2 * strength), 0, 255)  # reduce B
    return out.astype(np.uint8)


def _apply_phosphorus_deficiency(img: np.ndarray) -> np.ndarray:
    """Purple/reddish tint — boost red and blue channels."""
    out = img.copy().astype(np.float32)
    leaf_mask = (img[:, :, 1] > img[:, :, 0]) & (img[:, :, 1] > img[:, :, 2])
    strength = np.random.uniform(0.4, 0.85)
    out[leaf_mask, 0] = np.clip(out[leaf_mask, 0] + 60 * strength, 0, 255)   # boost R
    out[leaf_mask, 1] = np.clip(out[leaf_mask, 1] * (1 - 0.35 * strength), 0, 255)  # reduce G
    out[leaf_mask, 2] = np.clip(out[leaf_mask, 2] + 50 * strength, 0, 255)   # boost B
    return out.astype(np.uint8)


def _apply_potassium_deficiency(img: np.ndarray) -> np.ndarray:
    """Brown edge scorch effect — darken edges with brown mask."""
    out = img.copy().astype(np.float32)
    size = img.shape[0]
    strength = np.random.uniform(0.4, 0.9)

    # Distance map from image center
    cx, cy = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    edge_weight = np.clip((dist / max_dist - 0.4) / 0.6, 0, 1) * strength

    # Apply brown tint scaled by edge distance
    out[:, :, 0] = np.clip(out[:, :, 0] + 70 * edge_weight, 0, 255)  # R
    out[:, :, 1] = np.clip(out[:, :, 1] - 40 * edge_weight, 0, 255)  # G
    out[:, :, 2] = np.clip(out[:, :, 2] - 30 * edge_weight, 0, 255)  # B
    return out.astype(np.uint8)


def _apply_augmentation(img: np.ndarray) -> np.ndarray:
    """Random rotation, flip, brightness, zoom augmentation."""
    size = img.shape[0]

    # Random rotation
    angle = np.random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REFLECT)

    # Random flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    if np.random.rand() > 0.7:
        img = cv2.flip(img, 0)

    # Brightness
    beta = np.random.uniform(-30, 30)
    img = np.clip(img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Random zoom (crop + resize)
    zoom_factor = np.random.uniform(0.85, 1.0)
    crop_size = int(size * zoom_factor)
    start = (size - crop_size) // 2
    img = img[start:start + crop_size, start:start + crop_size]
    img = cv2.resize(img, (size, size))

    return img


def generate_synthetic_data(n_per_class: int = 300):
    """Generate synthetic NPK leaf images and labels."""
    print(f"[INFO] Generating {n_per_class} samples per class...")
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)

    deficiency_fns = {
        "nitrogen":   _apply_nitrogen_deficiency,
        "phosphorus": _apply_phosphorus_deficiency,
        "potassium":  _apply_potassium_deficiency,
    }
    # Multi-label: each sample can have combination deficiencies
    # We'll also generate "healthy" and mixed samples
    all_images = []
    all_labels = []  # shape (n, 3) — [N, P, K] binary

    nutrient_list = ["nitrogen", "phosphorus", "potassium"]

    for idx, (nutrient, fn) in enumerate(deficiency_fns.items()):
        for i in range(n_per_class):
            base = _base_leaf(224)
            img = fn(base)
            img = _apply_augmentation(img)
            img_path = os.path.join(SYNTHETIC_DIR, f"{nutrient}_{i:04d}.jpg")
            cv2.imwrite(img_path, img)

            label = [0, 0, 0]
            label[idx] = 1
            # Occasionally add secondary deficiency (multi-label)
            if np.random.rand() < 0.15:
                secondary = np.random.randint(0, 3)
                label[secondary] = 1

            all_images.append(img / 255.0)
            all_labels.append(label)

    # Healthy samples (no deficiency)
    n_healthy = n_per_class // 2
    for i in range(n_healthy):
        base = _base_leaf(224)
        img = _apply_augmentation(base)
        img_path = os.path.join(SYNTHETIC_DIR, f"healthy_{i:04d}.jpg")
        cv2.imwrite(img_path, img)
        all_images.append(img / 255.0)
        all_labels.append([0, 0, 0])

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    print(f"[INFO] Generated {len(X)} total samples.")
    return X, y


# ─────────────────────────────────────────────
# MODEL BUILDING
# ─────────────────────────────────────────────

def build_model():
    """Build MobileNetV2-based multi-label NPK classifier."""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models

    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # Freeze base layers initially

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation="sigmoid", name="npk_output")(x)

    model = models.Model(inputs, outputs, name="CropMind_NPK")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────

def train():
    import tensorflow as tf

    print("=" * 60)
    print("  CropMind — NPK Model Training")
    print("=" * 60)

    # Generate synthetic data
    X, y = generate_synthetic_data(n_per_class=300)

    # Train / val split (80/20)
    n = len(X)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")

    # Data augmentation layers (applied only during training)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Build model
    model = build_model()
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-6
        ),
    ]

    # Phase 1: Train head only (base frozen)
    print("\n[Phase 1] Training classification head (base frozen)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune top layers of base
    print("\n[Phase 2] Fine-tuning top 30 layers of MobileNetV2...")
    base_model = model.layers[1]  # MobileNetV2
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model.save(MODEL_PATH)
    np.save(CLASSES_PATH, np.array(["nitrogen", "phosphorus", "potassium"]))
    print(f"\n[SUCCESS] Model saved to {MODEL_PATH}")
    print(f"[SUCCESS] Label classes saved to {CLASSES_PATH}")

    # Final eval
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[RESULTS] Val Loss={val_loss:.4f} | Acc={val_acc:.4f} | AUC={val_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
