"""
Inference script that downloads the model and mapping from Hugging Face model repo and runs webcam-based inference using MediaPipe + TensorFlow.

Usage:
    python inference.py --repo a-01a/hand-gesture-recognition

The script will download `hand_gesture_lstm_model.h5` and `gesture_mapping.json` from the specified repo and run a simple webcam loop that collects 30 frames and prints the predicted gesture and confidence.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from huggingface_hub import hf_hub_download


SEQUENCE_LENGTH = 30
LOCAL_DIR = Path(".cache_hf_model")
LOCAL_DIR.mkdir(exist_ok=True)


def download_file(repo_id: str, filename: str, token: str = None) -> Path:
    """Download a file from the HF hub model repo and return local path."""
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        return Path(local_path)
    except Exception as e:
        # Try to download into local cache folder by specifying repo_id and repo_type implicitly
        print(f"Primary hf_hub_download failed for {filename}: {e}")
        raise


def load_model_and_mapping(repo_id: str, token: str = None):
    print("Downloading model and mapping from Hugging Face Hub...")
    model_path = download_file(repo_id, "hand_gesture_lstm_model.h5", token)
    mapping_path = download_file(repo_id, "gesture_mapping.json", token)

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    with open(mapping_path, "r", encoding="utf-8") as f:
        gesture_mapping = json.load(f)

    id_to_gesture = {v: k for k, v in gesture_mapping.items()}
    return model, id_to_gesture


class MediaPipeExtractor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        return (np.array(landmarks) if landmarks else np.zeros(63), bool(results.multi_hand_landmarks))


def run_webcam(model, id_to_gesture):
    extractor = MediaPipeExtractor()
    sequence = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting webcam. Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks, has_hand = extractor.extract(frame)

        if has_hand:
            sequence.append(landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            seq_arr = np.array([sequence])
            preds = model.predict(seq_arr, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            gesture = id_to_gesture.get(idx, f"class_{idx}")
            print(f"Prediction: {gesture} (confidence: {conf:.2%})")
            sequence = []

        # Simple overlay
        h, w = frame.shape[:2]
        status = "Hand detected" if has_hand else "No hand"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if has_hand else (0, 0, 255), 2)

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Hand Gesture Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. a-01a/hand-gesture-recognition")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN_HGR"), help="Hugging Face token or set HF_TOKEN_HGR in .env")
    args = parser.parse_args()

    model, id_to_gesture = load_model_and_mapping(args.repo, args.token)
    run_webcam(model, id_to_gesture)


if __name__ == "__main__":
    main()
