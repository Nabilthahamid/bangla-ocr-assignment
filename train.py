import argparse
import json
import os
from pathlib import Path

import cv2
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# BanglaLekha-Isolated 84-class mapping (Folder "1" -> Index 0 ... Folder "84" -> Index 83)
# Verified against sample images: folders 1-50 are letters, 51-60 digits, 61-84 compounds.
_VOWELS = list("অআইঈউঊঋএঐওঔ")  # folders 1-11
_CONSONANTS = ["ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "ণ",
               "ত", "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
               "স", "হ", "ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ"]  # folders 12-50
_DIGITS = list("০১২৩৪৫৬৭৮৯")  # folders 51-60
# Folders 61-84: compound characters. Order is a best guess (handwriting is hard to
# identify) and is NOT verified; the app's lexicon matching only uses single characters.
_COMPOUNDS = ["ক্ষ", "ক্ত", "ক্র", "ক্ল", "ক্ষ্ম", "গ্ধ", "ঙ্ক", "ঙ্গ", "চ্ছ", "জ্ঞ",
              "ঞ্চ", "ঞ্জ", "ট্ট", "ণ্ড", "ত্ত", "ত্থ", "ত্র", "দ্দ", "দ্ধ", "দ্ব",
              "দ্র", "ন্ত", "ন্দ", "ন্ধ"]
BANGLA_LABELS = dict(enumerate(_VOWELS + _CONSONANTS + _DIGITS + _COMPOUNDS))
assert len(BANGLA_LABELS) == 84


def parse_args():
    parser = argparse.ArgumentParser(description="Train BanglaLekha-Isolated CNN")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("DATA_DIR", ""),
        help="Path to BanglaLekha-Isolated Images folder containing subfolders 1..84",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument(
        "--arch",
        type=str,
        default="baseline_cnn",
        choices=["baseline_cnn", "deep_bn_cnn"],
        help="Model architecture variant",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Optional cap on images per class for faster CPU training",
    )
    return parser.parse_args()


def preprocess_char_image(img_gray: np.ndarray, img_size: int) -> np.ndarray:
    """Ensure white ink on black background, pad to square, and resize."""
    # Invert if background is bright (mean > 127)
    if np.mean(img_gray) > 127:
        img_gray = 255 - img_gray

    h, w = img_gray.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    padded[y_off : y_off + h, x_off : x_off + w] = img_gray

    resized = cv2.resize(padded, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[..., np.newaxis]


def load_dataset(data_dir: str, img_size: int, max_per_class=None):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {root}. Pass --data_dir or set DATA_DIR."
        )

    X, y = [], []
    for class_idx in range(84):
        folder = root / str(class_idx + 1)
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")

        files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        if max_per_class:
            files = files[:max_per_class]

        for img_path in files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(preprocess_char_image(img, img_size))
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(arch: str, img_size: int, num_classes: int = 84) -> tf.keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 1))

    if arch == "baseline_cnn":
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
    else:  # deep_bn_cnn
        x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=arch)


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics(
                {
                    "train_loss": float(logs["loss"]),
                    "train_accuracy": float(logs["accuracy"]),
                    "val_loss": float(logs["val_loss"]),
                    "val_accuracy": float(logs["val_accuracy"]),
                },
                step=epoch + 1,
            )


def main():
    args = parse_args()
    print(f"Loading dataset from: {args.data_dir}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        args.data_dir, args.img_size, args.max_per_class
    )
    print(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts/mlflow", exist_ok=True)

    # Save index-to-character mapping
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in BANGLA_LABELS.items()}, f, ensure_ascii=False, indent=2)

    mlflow.set_tracking_uri("file:./artifacts/mlflow")
    mlflow.set_experiment("bangla-handwritten-ocr")

    with mlflow.start_run(run_name=f"{args.arch}_ep{args.epochs}_lr{args.lr}"):
        mlflow.log_params(
            {
                "architecture": args.arch,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "img_size": args.img_size,
                "train_samples": len(X_train),
            }
        )

        model = build_model(args.arch, args.img_size, num_classes=84)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[MLflowEpochLogger()],
            verbose=1,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Final Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
        mlflow.log_metrics({"test_accuracy": float(test_acc), "test_loss": float(test_loss)})

        model_path = "models/model.keras"
        model.save(model_path)
        mlflow.keras.log_model(model, artifact_path="model")
        print(f"Saved trained model to {model_path} and logged to MLflow.")


if __name__ == "__main__":
    main()