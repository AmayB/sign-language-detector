
import { useEffect, useRef, useState } from "react";
import {
  closeHandTracker,
  getHandTracker,
  HAND_CONNECTIONS,
  initializeHandTracker,
} from "./handTracker";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

type Prediction = {
  letter: string;
  confidence: number;
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const landmarkFrameRef = useRef<number | null>(null);
  const predictingRef = useRef(false);

  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictionError, setPredictionError] = useState("");
  const [referenceOpen, setReferenceOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (landmarkFrameRef.current !== null) {
        window.cancelAnimationFrame(landmarkFrameRef.current);
      }
      closeHandTracker();
    };
  }, []);

  function clearLandmarkOverlay() {
    const canvas = overlayCanvasRef.current;
    const context = canvas?.getContext("2d");

    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  function drawLandmarks(timestamp: number) {
    const video = videoRef.current;
    const canvas = overlayCanvasRef.current;
    const tracker = getHandTracker();

    if (!video || !canvas || !tracker || video.videoWidth === 0) {
      return;
    }

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    const result = tracker.detectForVideo(video, timestamp);

    for (const landmarks of result.landmarks) {
      context.strokeStyle = "#5eead4";
      context.lineWidth = Math.max(2, canvas.width / 320);
      context.lineCap = "round";
      context.lineJoin = "round";

      for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
        const start = landmarks[startIndex];
        const end = landmarks[endIndex];

        context.beginPath();
        context.moveTo(start.x * canvas.width, start.y * canvas.height);
        context.lineTo(end.x * canvas.width, end.y * canvas.height);
        context.stroke();
      }

      context.fillStyle = "#f8fafc";
      for (const landmark of landmarks) {
        context.beginPath();
        context.arc(
          landmark.x * canvas.width,
          landmark.y * canvas.height,
          Math.max(3, canvas.width / 170),
          0,
          Math.PI * 2,
        );
        context.fill();
      }
    }

    landmarkFrameRef.current = window.requestAnimationFrame(drawLandmarks);
  }

  async function startLandmarkOverlay(video: HTMLVideoElement) {
    try {
      await initializeHandTracker();

      if (streamRef.current && videoRef.current === video) {
        landmarkFrameRef.current = window.requestAnimationFrame(drawLandmarks);
      }
    } catch {
      clearLandmarkOverlay();
    }
  }

  async function startCamera() {
    setCameraError("");
    setPredictionError("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        void startLandmarkOverlay(videoRef.current);
      }

      setCameraReady(true);
      setPrediction(null);

      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
      }

      timerRef.current = window.setInterval(() => {
        void predictFrame();
      }, 1000);
    } catch (error) {
      setCameraError(
        error instanceof Error
          ? error.message
          : "Could not access the camera. Check your browser permissions.",
      );
    }
  }

  function stopCamera() {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (landmarkFrameRef.current !== null) {
      window.cancelAnimationFrame(landmarkFrameRef.current);
      landmarkFrameRef.current = null;
    }
    clearLandmarkOverlay();
    closeHandTracker();

    setCameraReady(false);
    setPrediction(null);
    setPredictionError("");
    setLoading(false);
  }

  async function predictFrame() {
    const video = videoRef.current;

    if (
      !video ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.videoWidth === 0 ||
      predictingRef.current
    ) {
      return;
    }

    predictingRef.current = true;
    setLoading(true);

    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");

      if (!context) {
        throw new Error("Could not capture a camera frame.");
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (result) => {
            if (result) resolve(result);
            else reject(new Error("Could not create an image."));
          },
          "image/jpeg",
          0.85,
        );
      });

      const formData = new FormData();
      formData.append("file", blob, "webcam.jpg");

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setPrediction({
        letter: data.letter,
        confidence: data.confidence,
      });

      setPredictionError("");
    } catch (error) {
      setPredictionError(
        error instanceof Error
          ? error.message
          : "Could not get a prediction.",
      );
    } finally {
      predictingRef.current = false;
      setLoading(false);
    }
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setReferenceOpen(false);
      }
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  return (
    <main className="app">
      <header className="topbar">
        <h1>Sign Language Detector</h1>

        <a
          className="portfolio-link"
          href="https://amaybhardwaj.com"
          target="_blank"
          rel="noreferrer"
        >
          amaybhardwaj.com ↗
        </a>
      </header>

      <section className="main-content">
        <div className="intro">
          <h2>Sign Language Detector</h2>
          <p>
            
          </p>
        </div>

        <div className="action-row">
          <button
            className="button button-primary"
            onClick={cameraReady ? stopCamera : startCamera}
          >
            {cameraReady ? "Stop Camera" : "Start Camera"}
          </button>

          <button
            className="button button-secondary"
            onClick={() => setReferenceOpen(true)}
          >
            Sign Language Reference
          </button>
        </div>

        {cameraError && (
          <p className="error-message">{cameraError}</p>
        )}

        <div className="detector-layout">
          <section className="camera-panel">
            <div className="panel-heading">
              <h3>Camera</h3>
              <span className={cameraReady ? "status active" : "status"}>
                {cameraReady ? "Active" : "Inactive"}
              </span>
            </div>

            <div className="video-container">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={cameraReady ? "video visible" : "video"}
              />
              <canvas
                ref={overlayCanvasRef}
                className={cameraReady ? "video-overlay visible" : "video-overlay"}
                aria-hidden="true"
              />

              {!cameraReady && (
                <div className="camera-placeholder">
                  <p>Camera will appear here.</p>
                </div>
              )}
            </div>

            <p className="help-text">
              Use good lighting and keep your hand visible in the frame
            </p>
          </section>

          <section className="prediction-panel">
            <h3>Prediction</h3>

            {prediction ? (
              <>
                <div className="predicted-letter">{prediction.letter}</div>

                <p className="confidence">
                  Confidence:{" "}
                  {(prediction.confidence * 100).toFixed(1)}%
                </p>

                <div className="confidence-track">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(0, prediction.confidence * 100),
                      )}%`,
                    }}
                  />
                </div>
              </>
            ) : (
              <div className="prediction-placeholder">
                {loading
                  ? "Analyzing hand..."
                  : "Your predicted letter will appear here."}
              </div>
            )}

            {predictionError && (
              <p className="error-message">{predictionError}</p>
            )}

            <p className="help-text">
              Predictions update approximately once per second.
            </p>
          </section>
        </div>
      </section>

      {referenceOpen && (
        <div
          className="modal-overlay"
          onClick={() => setReferenceOpen(false)}
        >
          <section
            className="reference-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="reference-title"
            onClick={(event) => event.stopPropagation()}
          >
            <header className="modal-header">
              <div>
                <h2 id="reference-title"></h2>
                <p>Reference images for ASL alphabet</p>
              </div>

              <button
                className="close-button"
                onClick={() => setReferenceOpen(false)}
                aria-label="Close reference"
              >
                ×
              </button>
            </header>

            <div className="reference-grid">
              {letters.map((letter) => (
                <article className="reference-card" key={letter}>
                  <img
                    src={`/asl-reference/${letter}_test.jpg`}
                    alt={`ASL letter ${letter}`}
                    loading="lazy"
                    onError={(event) => {
                      event.currentTarget.style.display = "none";
                      event.currentTarget.parentElement?.classList.add(
                        "missing-image",
                      );
                    }}
                  />
                  <div className="reference-letter">{letter}</div>
                </article>
              ))}
            </div>

            <p className="reference-note">
              J and Z in ASL involve movement, so a reference image cannot
              show the actual sign.
            </p>
          </section>
        </div>
      )}
    </main>
  );
}
