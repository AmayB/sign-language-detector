
import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


# ==================================================
# CONFIGURATION
# ==================================================

DATASET_DIR = Path("data/my_hand")
HAND_MODEL_PATH = Path("models/hand_landmarker.task")

CLASSIFIER_PATH = Path("models/my_hand_landmark_classifier.joblib")
LABELS_PATH = Path("models/my_hand_landmark_labels.json")
FAILED_PATH = Path("models/failed_landmark_images.txt")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==================================================
# EXTRACT 63 HAND LANDMARK FEATURES
# ==================================================

def extract_landmarks(image_path, landmarker):
    """
    Detect a hand in an image and extract its 21 landmarks.

    Each landmark has x, y, and z coordinates.
    21 landmarks * 3 coordinates = 63 features.

    Coordinates are made relative to the wrist.
    Returns None if no hand is detected.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    # Convert OpenCV's BGR image into RGB for MediaPipe.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe image object.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    # Detect the hand.
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # Use the first detected hand.
    hand = result.hand_landmarks[0]

    # Use the wrist as the coordinate origin.
    wrist = hand[0]

    features = []

    for landmark in hand:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    return np.array(features, dtype=np.float32)


# ==================================================
# MAIN TRAINING FUNCTION
# ==================================================

def main():

    # Check that the required folders and files exist.
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {HAND_MODEL_PATH}"
        )

    # Find all letter folders, such as A, B, C, ..., Z.
    label_names = sorted(
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not label_names:
        raise ValueError("No letter folders found in the dataset.")

    print("Loading MediaPipe hand detector...")

    # Configure MediaPipe to process one image at a time.
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH.resolve())
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )

    all_features = []
    all_labels = []
    failed_images = []

    print(f"Found {len(label_names)} letter folders.")
    print("Extracting hand landmarks from images...\n")

    # Create the hand detector.
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        # Process every letter folder.
        for label in label_names:

            folder = DATASET_DIR / label

            image_paths = sorted(
                path
                for path in folder.rglob("*")
                if path.suffix.lower() in IMAGE_EXTENSIONS
            )

            successful = 0

            for image_path in image_paths:

                features = extract_landmarks(
                    image_path,
                    landmarker,
                )

                # Record images where no hand was detected.
                if features is None:
                    failed_images.append(str(image_path))
                    continue

                all_features.append(features)
                all_labels.append(label)
                successful += 1

            print(
                f"{label}: {successful}/{len(image_paths)} "
                "images processed successfully"
            )

    # Make sure we have enough usable examples.
    if not all_features:
        raise ValueError(
            "No hands were detected. Check your images and "
            "the MediaPipe model."
        )

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)

    # Keep only classes with at least two usable images,
    # because each needs training and validation examples.
    unique_labels, counts = np.unique(y, return_counts=True)

    usable_labels = [
        label
        for label, count in zip(unique_labels, counts)
        if count >= 2
    ]

    keep = np.isin(y, usable_labels)

    X = X[keep]
    y = y[keep]

    print("\nDataset summary")
    print("---------------------------")
    print(f"Usable images: {len(X)}")
    print(f"Failed detections: {len(failed_images)}")
    print(f"Features per image: {X.shape[1]}")
    print(f"Letters with enough data: {len(np.unique(y))}")

    if len(np.unique(y)) < 2:
        raise ValueError(
            "At least two letters need two or more detected images."
        )

    # Split the data into training and validation sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Standardize the features, then train a neural network.
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

    print("\nTraining the landmark classifier...")

    classifier.fit(X_train, y_train)

    # Evaluate the model on the validation split.
    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nValidation accuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=sorted(np.unique(y)),
            zero_division=0,
        )
    )

    # Save the trained classifier separately from your old model.
    CLASSIFIER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(classifier, CLASSIFIER_PATH)

    # Save the class names for use by the API.
    with LABELS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            sorted(np.unique(y).tolist()),
            file,
            indent=2,
        )

    # Save a list of images where no hand was detected.
    FAILED_PATH.write_text(
        "\n".join(failed_images),
        encoding="utf-8",
    )

    print("\nTraining complete!")
    print(f"Classifier saved to: {CLASSIFIER_PATH}")
    print(f"Labels saved to: {LABELS_PATH}")
    print(f"Failed image list saved to: {FAILED_PATH}")


if __name__ == "__main__":
    main()
