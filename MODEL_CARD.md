# Model Card: Hand Gesture Recognition LSTM

## Model Description

This model performs real-time hand gesture recognition using LSTM neural networks and MediaPipe hand pose estimation.

### Model Details

- **Developed by:** Abdul Ahad
- **Model type:** LSTM Sequential Neural Network
- **Language:** TensorFlow/Keras
- **License:** MIT
- **Model Architecture:** 3-layer LSTM with dense output layers

## Intended Use

### Primary Use Cases

- Real-time hand gesture recognition from webcam feeds
- Human-computer interaction applications
- Sign language recognition systems
- Gesture-controlled interfaces

### Out-of-Scope Uses

- Medical diagnosis
- Security/authentication systems (not designed for this purpose)
- Applications requiring 100% accuracy in critical scenarios

## Training Data

- **Dataset:** LeapGestRecog (gti-upm/leapgestrecog from Kaggle)
- **Structure:** 10 subjects × 10 gestures × multiple video sequences
- **Format:** 100 frames per gesture sequence (PNG images)
- **Preprocessing:** MediaPipe hand landmark extraction (21 landmarks × 3 coordinates = 63 features)
- **Augmentation:** Random noise, occlusion, scaling, and translation (3× data size)

## Model Architecture

```
Input Shape: (30, 63) - 30 frames × 63 features

Layer 1: LSTM(128, return_sequences=True)
         BatchNormalization + Dropout(0.3)

Layer 2: LSTM(128, return_sequences=True)
         BatchNormalization + Dropout(0.3)

Layer 3: LSTM(64)
         BatchNormalization + Dropout(0.3)

Layer 4: Dense(256, activation='relu')
         BatchNormalization + Dropout(0.3)

Layer 5: Dense(128, activation='relu')
         BatchNormalization + Dropout(0.3)

Output:  Dense(10, activation='softmax')
```

## Training Procedure

### Hyperparameters

- **Sequence Length:** 30 frames
- **LSTM Units:** 128 → 128 → 64
- **Dense Units:** 256 → 128
- **Dropout Rate:** 0.3
- **Batch Size:** 32
- **Initial Learning Rate:** 0.001
- **Optimizer:** Adam with ReduceLROnPlateau
- **Loss Function:** Categorical Crossentropy
- **Epochs:** Up to 100 (with EarlyStopping)

### Data Split

- **Training:** 64%
- **Validation:** 16%
- **Test:** 20%

## Performance

The model achieves high accuracy on the LeapGestRecog dataset test set. Performance metrics include:

- Overall accuracy
- Per-gesture precision, recall, and F1-score
- Confusion matrix analysis

See the technical report for detailed performance metrics.

## Limitations

1. **Lighting Conditions:** Performance may degrade in poor lighting
2. **Hand Visibility:** Requires clear view of hand landmarks
3. **Background Complexity:** May struggle with cluttered backgrounds
4. **Single Hand:** Designed for single-hand gestures
5. **Dataset Bias:** Trained on specific gesture types from LeapGestRecog

## How to Use

### Installation

```bash
uv pip install tensorflow mediapipe opencv-python numpy huggingface_hub
```

### Inference

```python
# Download and run inference
uv run python inference.py --repo a-01a/hand-gesture-recognition
```

Or programmatically:

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf
import json

model_path = hf_hub_download(repo_id="a-01a/hand-gesture-recognition", 
                              filename="hand_gesture_lstm_model.h5")
mapping_path = hf_hub_download(repo_id="a-01a/hand-gesture-recognition", 
                                filename="gesture_mapping.json")

model = tf.keras.models.load_model(model_path)

with open(mapping_path, 'r') as f:
    gesture_mapping = json.load(f)
```

## Citation

```bibtex
@misc{hand_gesture_lstm_2025,
  title={Hand Gesture Recognition using LSTM and MediaPipe},
  author={Abdul Ahad},
  year={2025},
  howpublished={https://huggingface.co/a-01a/hand-gesture-recognition},
  note={Real-time hand gesture recognition system using MediaPipe and LSTM networks}
}
```
