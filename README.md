---
title: Hand Gesture Recognition
emoji: 🖐️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - computer-vision
  - gesture-recognition
  - lstm
  - mediapipe
  - hand-tracking
  - video-classification
---


# Hand Gesture Recognition using LSTM

A real-time hand gesture recognition system using MediaPipe for hand pose extraction and LSTM neural networks for temporal sequence classification.

## Project Overview

This project implements a complete pipeline for recognizing hand gestures from video sequences using:
- **MediaPipe Hands** for extracting 21 3D hand landmarks
- **LSTM Neural Networks** for learning temporal patterns in hand movements
- **Data Augmentation** to improve model generalization
- **Real-time Recognition** capability via webcam

## Dataset

The project uses the **LeapGestRecog** dataset from Kaggle:
- **Source**: `gti-upm/leapgestrecog`
- **Structure**: 10 subjects × 10 gestures × multiple video sequences
- **Format**: 100 frames per gesture sequence (PNG images)
- The dataset is automatically downloaded to the current directory and cleaned up after training

## Features

1. **Automatic Dataset Management**
   - Downloads dataset to current directory
   - Organizes and preprocesses data
   - Automatic cleanup after training to save space

2. **Hand Pose Extraction**
   - Uses MediaPipe to extract 21 landmarks (63 features: x, y, z coordinates)
   - Processes entire video sequences
   - Visualization support

3. **Data Augmentation**
   - Random noise injection
   - Random occlusion
   - Random scaling and translation
   - Increases dataset size by 3× (original + 2× augmented)

4. **Deep Learning Model**
   - 3-layer LSTM architecture
   - Batch normalization and dropout for regularization
   - Trained on 30-frame sequences
   - Achieves high accuracy on test set

5. **Real-time Recognition**
   - Webcam-based gesture recognition
   - Live predictions with confidence scores
   - Visual feedback with hand landmark overlay

## Installation

This project uses **uv** for fast and reliable package management.

### Quick Install

```bash
# Install uv (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Navigate to project directory
cd hand_gesture_recognition

# Install all dependencies
uv pip install -e .
```

### Required Packages:
- `kagglehub` - Dataset download
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `mediapipe` - Hand pose estimation
- `scikit-learn` - ML utilities
- `tensorflow` - Deep learning framework
- `tqdm` - Progress bars

All dependencies are automatically installed via `uv pip install -e .`

## Real-time Recognition

### Using Inference Script (Recommended)

The easiest way to test the model is using the `inference.py` script, which downloads the model from Hugging Face and runs webcam inference:

```bash
python inference.py --repo a-01a/hand-gesture-recognition
```

Features:
- ✅ Automatically downloads model from Hugging Face Hub
- ✅ Live gesture prediction with confidence scores
- ✅ Real-time hand landmark detection using MediaPipe
- ✅ FPS counter and hand detection status
- ✅ No manual model download required

Press 'q' to quit the webcam window.

### Using the Notebook

Alternatively, you can run the webcam demo in the notebook after training:

```python
recognizer = RealTimeGestureRecognizer('hand_gesture_lstm_model.h5', gesture_mapping)
recognizer.run_webcam_demo()
```

## Model Architecture

```
Input: (30, 63) - 30 frames × 63 features

LSTM Layer 1: 128 units (return sequences)
    ↓ BatchNormalization + Dropout(0.3)

LSTM Layer 2: 128 units (return sequences)
    ↓ BatchNormalization + Dropout(0.3)

LSTM Layer 3: 64 units
    ↓ BatchNormalization + Dropout(0.3)

Dense Layer 1: 256 units (ReLU)
    ↓ BatchNormalization + Dropout(0.3)

Dense Layer 2: 128 units (ReLU)
    ↓ BatchNormalization + Dropout(0.3)

Output Layer: 10 units (Softmax)
```

## Performance

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance visualization
- **Classification Report**: Precision, recall, F1-score per gesture
- **Gesture-wise Analysis**: Individual gesture accuracy

## Gestures Recognized

The model recognizes 10 different hand gestures from the LeapGestRecog dataset. Each gesture has unique characteristics captured through the temporal sequence of hand landmarks.

## Hyperparameters

- **Sequence Length**: 30 frames
- **LSTM Units**: 128 → 128 → 64
- **Dropout Rate**: 0.3
- **Batch Size**: 32
- **Learning Rate**: 0.001 (with ReduceLROnPlateau)
- **Epochs**: 100 (with EarlyStopping)
- **Train/Val/Test Split**: 64%/16%/20%

## Project Structure

```
hand_gesture_recognition/
├── hand_gesture_recognition.ipynb  # Main training notebook
├── inference.py                    # Webcam inference with model download from HF
├── upload_to_huggingface.py       # Upload model to Hugging Face
├── README.md                       # This file
├── TECHNICAL_REPORT.md            # Detailed mathematical concepts
├── LICENSE.md                      # License
├── pyproject.toml                 # Project configuration (uv)
├── hand_gesture_lstm_model.h5     # Saved model (generated)
├── gesture_mapping.json           # Gesture labels (generated)
└── datasets/                       # Dataset (auto-downloaded & auto-deleted)
```

## Cleanup

The notebook automatically deletes the downloaded dataset after training to save disk space. The trained model and gesture mappings are saved locally and can be uploaded to Hugging Face for easy sharing and deployment.

## For More Details

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for a comprehensive explanation of all mathematical concepts, algorithms, and methodologies used in this project.

## Citation

If you use this model in your research or application, please cite:

```bibtex
@misc{hand_gesture_lstm_2025,
  title={Hand Gesture Recognition using LSTM and MediaPipe},
  author={Abdul Ahad},
  year={2025},
  howpublished={\url{https://huggingface.co/spaces/a-01a/hand-gesture-recognition}},
  note={Real-time hand gesture recognition system using MediaPipe and LSTM networks}
}
```

## 📄 License

MIT License - See [LICENSE.md](LICENSE.md) for details.

---