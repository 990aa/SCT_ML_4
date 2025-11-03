# Hand Gesture Recognition using LSTM

A real-time hand gesture recognition system using MediaPipe for hand pose extraction and LSTM neural networks for temporal sequence classification.

## 🎯 Project Overview

This project implements a complete pipeline for recognizing hand gestures from video sequences using:
- **MediaPipe Hands** for extracting 21 3D hand landmarks
- **LSTM Neural Networks** for learning temporal patterns in hand movements
- **Data Augmentation** to improve model generalization
- **Real-time Recognition** capability via webcam

## 📊 Dataset

The project uses the **LeapGestRecog** dataset from Kaggle:
- **Source**: `gti-upm/leapgestrecog`
- **Structure**: 10 subjects × 10 gestures × multiple video sequences
- **Format**: 100 frames per gesture sequence (PNG images)
- The dataset is automatically downloaded to the current directory and cleaned up after training

## 🚀 Features

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

## 🛠️ Installation

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

## 📖 Usage

### Training the Model

Run the Jupyter notebook cells sequentially:

```python
# 1. Download dataset (Cell 1)

# 2. Import libraries and initialize MediaPipe (Cell 2)

# 3. Extract hand landmarks from all images (Cells 3-4)

# 4. Create sequences and apply augmentation (Cells 5-6)

# 5. Build and train LSTM model (Cells 7-8)

# 6. Evaluate model performance (Cells 9-11)

# 7. Cleanup dataset (Cell 12)
# Automatically removes downloaded files
```

### Real-time Recognition

**Option 1: Using the Notebook**

Uncomment and run the webcam demo in the notebook:

```python
recognizer = RealTimeGestureRecognizer('hand_gesture_lstm_model.h5', gesture_mapping)
recognizer.run_webcam_demo()
```

**Option 2: Using the Webcam Tester (Recommended)**

Use the comprehensive testing interface with accuracy tracking:

```bash
python webcam_gesture_tester.py
```

Features:
- ✅ Live gesture prediction with confidence scores
- ✅ Accuracy and error rate tracking
- ✅ Per-gesture performance statistics
- ✅ Interactive testing mode
- ✅ Export results to file

See [WEBCAM_TESTER_GUIDE.md](WEBCAM_TESTER_GUIDE.md) for detailed instructions.

Press 'q' to quit the webcam window.

## 📐 Model Architecture

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

## 📊 Performance

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance visualization
- **Classification Report**: Precision, recall, F1-score per gesture
- **Gesture-wise Analysis**: Individual gesture accuracy

## 🎨 Gestures Recognized

The model recognizes 10 different hand gestures from the LeapGestRecog dataset. Each gesture has unique characteristics captured through the temporal sequence of hand landmarks.

## 🔧 Hyperparameters

- **Sequence Length**: 30 frames
- **LSTM Units**: 128 → 128 → 64
- **Dropout Rate**: 0.3
- **Batch Size**: 32
- **Learning Rate**: 0.001 (with ReduceLROnPlateau)
- **Epochs**: 100 (with EarlyStopping)
- **Train/Val/Test Split**: 64%/16%/20%

## 📁 Project Structure

```
hand_gesture_recognition/
├── hand_gesture_recognition.ipynb  # Main training notebook
├── webcam_gesture_tester.py       # Webcam testing interface
├── README.md                       # This file
├── INSTALLATION.md                # Detailed installation guide
├── TECHNICAL_REPORT.md            # Detailed mathematical concepts
├── WEBCAM_TESTER_GUIDE.md         # Webcam tester user guide
├── LICENSE.md                      # License
├── pyproject.toml                 # Project configuration (uv)
├── hand_gesture_lstm_model.h5     # Saved model (generated)
├── gesture_mapping.json           # Gesture labels (generated)
├── gesture_test_results_*.txt     # Test results (generated)
└── leapgestrecog/                 # Dataset (auto-downloaded & auto-deleted)
```

## 🧹 Cleanup

The notebook automatically deletes the downloaded dataset after training to save disk space. This includes:
- The `leapgestrecog/` directory
- Any cached files from kagglehub

The cleanup happens automatically in the final cell of the notebook.

## 🔬 For More Details

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for a comprehensive explanation of all mathematical concepts, algorithms, and methodologies used in this project.

## 📝 License

See [LICENSE.md](LICENSE.md) for license information.
---