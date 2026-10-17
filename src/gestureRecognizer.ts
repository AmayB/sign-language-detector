
import {
  FilesetResolver,
  GestureRecognizer,
} from "@mediapipe/tasks-vision";

let recognizer: GestureRecognizer | null = null;

export async function initializeGestureRecognizer() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
  );

  recognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  return recognizer;
}

export function getGestureRecognizer() {
  return recognizer;
}

export function closeGestureRecognizer() {
  recognizer?.close();
  recognizer = null;
}
