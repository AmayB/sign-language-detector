
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

DATASET_DIR = Path("data/my_hand")
MODEL_PATH = Path("models/my_hand_landmark_classifier.joblib")
LABELS_PATH = Path("models/my_hand_landmark_labels.json")
HAND_MODEL_PATH = Path("models/hand_landmarker.task")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# MediaPipe detects the hand and returns 21 landmarks.
# Each landmark has x, y, and z coordinates: 21 * 3 = 63 features.


# --------------------------------------------------
# Extract hand landmarks from an image
# --------------------------------------------------

def extract_landmarks(image_path, landmarker):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    # OpenCV reads images as BGR; MediaPipe expects RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # Use the first detected hand.
    hand = result.hand_landmarks[0]

    # Make coordinates relative to the wrist so that the
    # classifier focuses more on hand shape than image position.
    wrist = hand[0]

    features = []
    for landmark in hand:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    return np.array(features, dtype=np.float32)


# --------------------------------------------------
# Load dataset and extract features
# --------------------------------------------------

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand model not found: {HAND_MODEL_PATH}"
        )

    print("Loading MediaPipe hand-landmark model...")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH.resolve())
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )

    features = []
    labels = []
    failed_images = []
    label_names = sorted(
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not label_names:
        raise ValueError("No letter folders were found.")

    print(f"Found {len(label_names)} letter folders.")
    print("Extracting landmarks from images...\n")

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:
        for label in label_names:
            folder = DATASET_DIR / label
            image_paths = [
                p for p in folder.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]

            successful = 0

            for image_path in image_paths:
                landmarks = extract_landmarks(
                    image_path, landmarker
                )

                if landmarks is None:
                    failed_images.append(str(image_path))
                    continue

                features.append(landmarks)
                labels.append(label)
                successful += 1

            print(
                f"{label}: {successful}/{len(image_paths)} "
                "images had detectable hands"
            )

    if len(features) < 26:
        raise ValueError(
            "Too few images had detectable hands. "
            "Check the dataset and image framing."
        )

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    # Remove labels that have too few usable examples.
    unique, counts = np.unique(y, return_counts=True)
    usable_labels = [
        label for label, count in zip(unique, counts)
        if count >= 2
    ]
    keep = np.isin(y, usable_labels)
    X = X[keep]
    y = y[keep]

    print("\nDataset summary")
    print("----------------")
    print(f"Usable images: {len(X)}")
    print(f"Failed detections: {len(failed_images)}")
    print(f"Features per image: {X.shape[1]}")
    print(f"Usable letters: {len(np.unique(y))}")

    if len(np.unique(y)) < 2:
        raise ValueError(
            "At least two letters need usable images."
        )

    # Split into training and validation data.
    # This is a preliminary evaluation; near-duplicate images
    # can still make random-split accuracy look too optimistic.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Standardize the 63 coordinates, then train a small
    # multilayer perceptron classifier.
    classifier = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=600,
            early_stopping=True,
            random_state=42,
        ),
    )

    print("\nTraining classifier...")
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nValidation accuracy: {accuracy:.2%}")
    print("\nPer-letter report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=sorted(np.unique(y)),
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)

    with LABELS_PATH.open("w", encoding="utf-8") as file:
        json.dump(sorted(np.unique(y).tolist()), file, indent=2)

    failed_path = Path("models/failed_landmark_images.txt")
    failed_path.write_text(
        "\n".join(failed_images),
        encoding="utf-8",
    )

    print("\nSaved classifier:", MODEL_PATH)
    print("Saved labels:", LABELS_PATH)
    print("Failed image list:", failed_path)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
