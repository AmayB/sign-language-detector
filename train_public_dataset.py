
from pathlib import Path
import json
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_DIR = Path(
    r"C:\Users\amabh\Downloads\ASL_Alphabet_Dataset\asl_alphabet_train"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Keep these separate from your existing models.
MODEL_PATH = MODEL_DIR / "my_hand_landmark_classifier_gpu.pth"
LABELS_PATH = MODEL_DIR / "my_hand_landmark_labels_gpu.json"

# Each processed image is saved to this file.
# If training is interrupted, rerun this script to resume extraction.
CACHE_PATH = MODEL_DIR / "landmark_cache.jsonl"
FAILED_PATH = MODEL_DIR / "failed_images_gpu.txt"

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BATCH_SIZE = 2048
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Training device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_DIR / "hand_landmarker.task")
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
)


def extract_features(image_path, landmarker):
    """Extract 63 wrist-relative XYZ hand landmark values."""

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    points = result.hand_landmarks[0]
    wrist = points[0]

    features = []

    for point in points:
        features.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z,
        ])

    return np.asarray(features, dtype=np.float32)


# --------------------------------------------------
# RESUMABLE LANDMARK EXTRACTION
# --------------------------------------------------

def load_cache():
    """Load previously processed images from the JSONL cache."""

    features = []
    labels = []
    processed_paths = set()
    failed_images = []

    if not CACHE_PATH.exists():
        return features, labels, processed_paths, failed_images

    print(f"Loading existing cache: {CACHE_PATH}")

    with CACHE_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                path = record["path"]
                processed_paths.add(path)

                if record["features"] is not None:
                    features.append(
                        np.asarray(record["features"], dtype=np.float32)
                    )
                    labels.append(record["label"])
                else:
                    failed_images.append(
                        f'{record["label"]}\t{path}\t'
                        f'{record.get("error", "No hand detected")}'
                    )

            except (json.JSONDecodeError, KeyError, TypeError) as error:
                print(
                    f"Skipping invalid cache line {line_number}: {error}"
                )

    print(f"Cached images: {len(processed_paths):,}")
    print(f"Usable cached samples: {len(labels):,}")

    return features, labels, processed_paths, failed_images


def extract_dataset():
    """Extract only images not already recorded in the cache."""

    features, labels, processed_paths, failed_images = load_cache()

    start_time = time.time()
    newly_processed = 0

    with HandLandmarker.create_from_options(
        landmarker_options
    ) as landmarker:

        # Append instead of replacing the cache.
        with CACHE_PATH.open("a", encoding="utf-8") as cache_file:

            for letter in LETTERS:
                folder = DATASET_DIR / letter

                if not folder.is_dir():
                    raise FileNotFoundError(
                        f"Missing letter folder: {folder}"
                    )

                image_paths = sorted(
                    path for path in folder.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                )

                remaining = [
                    path for path in image_paths
                    if str(path.resolve()) not in processed_paths
                ]

                print(
                    f"\n{letter}: {len(image_paths):,} total images; "
                    f"{len(remaining):,} remaining"
                )

                for index, image_path in enumerate(remaining, start=1):
                    path_key = str(image_path.resolve())

                    try:
                        result = extract_features(image_path, landmarker)

                        if result is None:
                            record = {
                                "path": path_key,
                                "label": letter,
                                "features": None,
                                "error": "No hand detected or unreadable",
                            }
                            failed_images.append(
                                f"{letter}\t{path_key}\t"
                                "No hand detected or unreadable"
                            )
                        else:
                            record = {
                                "path": path_key,
                                "label": letter,
                                "features": result.tolist(),
                            }
                            features.append(result)
                            labels.append(letter)

                    except Exception as error:
                        record = {
                            "path": path_key,
                            "label": letter,
                            "features": None,
                            "error": str(error),
                        }
                        failed_images.append(
                            f"{letter}\t{path_key}\t{error}"
                        )

                    # Record every attempted image so a restart can skip it.
                    cache_file.write(json.dumps(record) + "\n")
                    processed_paths.add(path_key)
                    newly_processed += 1

                    # Flush regularly so progress survives most interruptions.
                    if newly_processed % 25 == 0:
                        cache_file.flush()

                    if index % 500 == 0:
                        cache_file.flush()
                        elapsed = (time.time() - start_time) / 60
                        print(
                            f"  {index:,}/{len(remaining):,} new images; "
                            f"{len(processed_paths):,} total cached; "
                            f"{elapsed:.1f} minutes this run"
                        )

                cache_file.flush()

    FAILED_PATH.write_text(
        "\n".join(failed_images),
        encoding="utf-8",
    )

    if not features:
        raise RuntimeError("No usable hand landmarks were extracted.")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels)

    print(f"\nCached images: {len(processed_paths):,}")
    print(f"Usable samples: {len(y):,}")
    print(f"Feature shape: {X.shape}")
    print(f"Skipped/failed images: {len(failed_images):,}")

    return X, y


# --------------------------------------------------
# PYTORCH NEURAL NETWORK
# --------------------------------------------------

class ASLClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def train_classifier(X, y):
    """Train the classifier on the GPU when CUDA is available."""

    # Encode letters as integer class IDs.
    label_to_id = {letter: i for i, letter in enumerate(LETTERS)}
    y_ids = np.asarray([label_to_id[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_ids,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_ids,
    )

    # Normalize using training data only to avoid test-data leakage.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Reserve part of the training split for early stopping.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    X_val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(DEVICE)
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64)).to(DEVICE)

    model = ASLClassifier(
        input_size=X.shape[1],
        num_classes=len(LETTERS),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print(f"\nTraining samples: {len(y_train):,}")
    print(f"Validation samples: {len(y_val):,}")
    print(f"Test samples: {len(y_test):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training on: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(
                DEVICE, non_blocking=(DEVICE.type == "cuda")
            )
            batch_y = batch_y.to(
                DEVICE, non_blocking=(DEVICE.type == "cuda")
            )

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = loss_function(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(batch_y)

        average_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = loss_function(
                val_logits, y_val_tensor
            ).item()

            val_accuracy = (
                (val_logits.argmax(dim=1) == y_val_tensor)
                .float()
                .mean()
                .item()
            )

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"train loss: {average_train_loss:.4f} | "
            f"validation loss: {val_loss:.4f} | "
            f"validation accuracy: {val_accuracy:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation loss stopped improving.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate only on the held-out test set.
    X_test_tensor = torch.from_numpy(
        X_test.astype(np.float32)
    ).to(DEVICE)

    with torch.no_grad():
        predictions = (
            model(X_test_tensor)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nHeld-out test accuracy: {accuracy:.4%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(LETTERS))),
            target_names=LETTERS,
            zero_division=0,
        )
    )

    # Save weights and preprocessing information together.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": X.shape[1],
            "num_classes": len(LETTERS),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "labels": LETTERS,
        },
        MODEL_PATH,
    )

    LABELS_PATH.write_text(
        json.dumps(LETTERS, indent=2),
        encoding="utf-8",
    )

    print(f"GPU model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABELS_PATH}")


def main():
    start = time.time()

    if DEVICE.type != "cuda":
        print(
            "WARNING: CUDA is unavailable. "
            "Training will run on the CPU."
        )

    for letter in LETTERS:
        if not (DATASET_DIR / letter).is_dir():
            raise FileNotFoundError(
                f"Missing letter folder: {DATASET_DIR / letter}"
            )

    X, y = extract_dataset()
    train_classifier(X, y)

    print(f"\nTotal elapsed time: {(time.time() - start) / 60:.1f} minutes")
    print("Existing models were not overwritten.")
    print("Landmark cache retained so extraction can be resumed.")


if __name__ == "__main__":
    main()
