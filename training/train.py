
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Dataset folders A, B, C, ... Z.
DATASET_DIR = Path(
    r"C:\Users\amabh\Downloads\archive\asl_alphabet_train\asl_alphabet_train"
)

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Save to separate files so the existing models remain safe.
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_MODEL_PATH = OUTPUT_DIR / "asl_alphabet_v3.keras"
NEW_LABELS_PATH = OUTPUT_DIR / "labels_v3.json"

tf.keras.utils.set_random_seed(42)


def main():
    image_paths = []
    numeric_labels = []

    print("Loading image paths...")

    for label, letter in enumerate(LETTERS):
        letter_dir = DATASET_DIR / letter

        if not letter_dir.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder for {letter}: {letter_dir}"
            )

        files = []
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(letter_dir.glob(extension))

        if not files:
            raise RuntimeError(f"No images found in {letter_dir}")

        image_paths.extend(str(path) for path in files)
        numeric_labels.extend([label] * len(files))

        print(f"{letter}: {len(files)} images")

    paths = np.array(image_paths)
    labels = np.array(numeric_labels, dtype=np.int32)

    paths_train, paths_val, y_train, y_val = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])

        # Match the API's RGB conversion and 64 x 64 input size.
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    def make_dataset(paths, labels, training=False):
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        if training:
            dataset = dataset.shuffle(
                buffer_size=min(len(paths), 5000),
                seed=42,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.map(
            load_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(paths_train, y_train, training=True)
    val_ds = make_dataset(paths_val, y_val)

    # Mild augmentation helps the model handle natural variation.
    # No horizontal flipping: mirroring may change sign interpretation.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomTranslation(
                height_factor=0.06,
                width_factor=0.06,
            ),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
            ),
            augmentation,
            tf.keras.layers.Conv2D(
                32, 3, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                64, 3, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                128, 3, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(
                len(LETTERS),
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(NEW_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
    ]

    print("\nStarting training...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=False,
    )

    best_model = tf.keras.models.load_model(NEW_MODEL_PATH)
    _, accuracy = best_model.evaluate(val_ds, verbose=1)

    with open(NEW_LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(LETTERS, file)

    print(f"\nValidation accuracy: {accuracy:.2%}")
    print(f"Model saved to: {NEW_MODEL_PATH}")
    print(f"Labels saved to: {NEW_LABELS_PATH}")
    print("Your previous models have not been overwritten.")


if __name__ == "__main__":
    main()
