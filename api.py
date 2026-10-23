
import json
import uuid
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

CHECKPOINT_PATH = MODEL_DIR / "my_hand_landmark_classifier_gpu.pth"
LABELS_PATH = MODEL_DIR / "my_hand_landmark_labels_gpu.json"
LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"

COLLECTION_DIR = BASE_DIR / "data" / "my_hand"

ALLOWED_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

DEVICE = torch.device("cpu")


# --------------------------------------------------
# PyTorch model architecture
# --------------------------------------------------

class ASLClassifier(nn.Module):
    def __init__(self, input_size=63, num_classes=26):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# Load model and normalization values
# --------------------------------------------------

for path in (CHECKPOINT_PATH, LABELS_PATH, LANDMARKER_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    labels = json.load(file)

if not isinstance(labels, list) or len(labels) != 26:
    raise ValueError("The labels file must contain 26 letter classes.")

if not set(labels).issubset(ALLOWED_LETTERS):
    raise ValueError("The labels file contains unsupported classes.")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

if checkpoint["input_size"] != 63:
    raise ValueError("Expected 63 input landmark features.")

if checkpoint["num_classes"] != len(labels):
    raise ValueError("Checkpoint and label counts do not match.")

checkpoint_labels = checkpoint.get("labels")
if checkpoint_labels is not None and checkpoint_labels != labels:
    raise ValueError("Checkpoint labels do not match the labels JSON file.")

classifier = ASLClassifier(
    input_size=checkpoint["input_size"],
    num_classes=checkpoint["num_classes"],
).to(DEVICE)

classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.eval()

feature_mean = torch.as_tensor(
    checkpoint["mean"],
    dtype=torch.float32,
    device=DEVICE,
).reshape(1, -1)

feature_std = torch.as_tensor(
    checkpoint["std"],
    dtype=torch.float32,
    device=DEVICE,
).reshape(1, -1)

if feature_mean.shape[1] != 63 or feature_std.shape[1] != 63:
    raise ValueError("Invalid feature normalization dimensions.")

feature_std = torch.where(
    feature_std < 1e-8,
    torch.ones_like(feature_std),
    feature_std,
)


# --------------------------------------------------
# Initialize MediaPipe hand detector
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(LANDMARKER_PATH.resolve())
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)

hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
    landmarker_options
)

detector_lock = threading.Lock()


# --------------------------------------------------
# FastAPI application and CORS
# --------------------------------------------------

app = FastAPI(
    title="ASL Sign Language Detector",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sign-language-detector-4y1j.onrender.com",
        "https://asl.amaybhardwaj.com",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Extract 63 wrist-relative landmark features
# --------------------------------------------------

def extract_landmarks(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    with detector_lock:
        result = hand_detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]
    wrist = hand[0]
    features = []

    for landmark in hand:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    return np.asarray(features, dtype=np.float32).reshape(1, 63)


# --------------------------------------------------
# Read and validate uploaded images
# --------------------------------------------------

async def read_uploaded_image(file: UploadFile):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image.",
        )

    return image


# --------------------------------------------------
# API routes
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "ASL Sign Language Detector API is running.",
        "model": CHECKPOINT_PATH.name,
        "device": str(DEVICE),
        "classes": labels,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(DEVICE),
        "classes_count": len(labels),
    }


@app.post("/debug-detect")
async def debug_detect(file: UploadFile = File(...)):
    image = await read_uploaded_image(file)

    height, width = image.shape[:2]
    features = extract_landmarks(image)

    return {
        "image_width": width,
        "image_height": height,
        "image_channels": image.shape[2] if len(image.shape) == 3 else 1,
        "landmarker_file_exists": LANDMARKER_PATH.is_file(),
        "landmarker_file_size": LANDMARKER_PATH.stat().st_size,
        "hand_detected": features is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await read_uploaded_image(file)
    features = extract_landmarks(image)

    if features is None:
        raise HTTPException(
            status_code=422,
            detail="No hand detected. Show one clearly visible hand in the image.",
        )

    try:
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=DEVICE,
        )

        feature_tensor = (feature_tensor - feature_mean) / feature_std

        with torch.inference_mode():
            logits = classifier(feature_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            best_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[best_index].item())

        predicted_letter = labels[best_index]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    if predicted_letter not in ALLOWED_LETTERS:
        raise HTTPException(
            status_code=500,
            detail="The model returned an unknown letter.",
        )

    return {
        "letter": predicted_letter,
        "confidence": confidence,
    }


@app.post("/collect")
async def collect(letter: str, file: UploadFile = File(...)):
    letter = letter.strip().upper()

    if letter not in ALLOWED_LETTERS:
        raise HTTPException(
            status_code=400,
            detail="Letter must be A-Z.",
        )

    image = await read_uploaded_image(file)

    letter_dir = COLLECTION_DIR / letter
    letter_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = letter_dir / filename

    if not cv2.imwrite(str(save_path), image):
        raise HTTPException(
            status_code=500,
            detail="Could not save the collected image.",
        )

    return {
        "message": "Image collected successfully.",
        "letter": letter,
        "filename": filename,
        "count": len(list(letter_dir.glob("*.jpg"))),
    }


@app.get("/collection/counts")
def collection_counts():
    counts = {}

    for letter in sorted(ALLOWED_LETTERS):
        letter_dir = COLLECTION_DIR / letter
        counts[letter] = (
            len(list(letter_dir.glob("*.jpg")))
            if letter_dir.exists()
            else 0
        )

    return {"counts": counts}
