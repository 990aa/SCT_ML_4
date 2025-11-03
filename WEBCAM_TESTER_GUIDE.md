# Webcam Gesture Tester - User Guide

## Overview

The `webcam_gesture_tester.py` script provides a comprehensive real-time testing interface for the hand gesture recognition model with:

✅ **Live gesture prediction** with confidence scores  
✅ **Accuracy tracking** per gesture  
✅ **Error rate calculation**  
✅ **Visual feedback** with color-coded confidence  
✅ **Statistics export** to file  
✅ **Interactive testing mode**  

## Features

### 1. **Real-time Prediction Display**
- Shows current gesture prediction
- Confidence score with color-coded bar (Green: >80%, Orange: 50-80%, Red: <50%)
- FPS counter
- Hand landmark overlay

### 2. **All Predictions Panel**
- Shows probability for all gesture classes
- Sorted by confidence
- Visual bars for each prediction
- Top 10 predictions displayed

### 3. **Accuracy Tracking**
- Overall accuracy percentage
- Error rate calculation
- Per-gesture statistics
- Average confidence scores
- Total test count

### 4. **Testing Mode**
- Select a gesture to test using number keys (1-9)
- Record predictions for accuracy calculation
- Real-time feedback on correct/incorrect predictions

### 5. **Statistics Export**
- Save detailed results to timestamped file
- Per-gesture accuracy breakdown
- Confidence metrics
- Error analysis

## Installation & Setup

### Prerequisites
```bash
# Ensure you have trained the model first
# Run hand_gesture_recognition.ipynb completely
```

### Files Required
- `hand_gesture_lstm_model.h5` - Trained model (generated from notebook)
- `gesture_mapping.json` - Gesture labels (auto-generated from notebook)
- `webcam_gesture_tester.py` - This testing script

### Install Dependencies (if not already installed)
```bash
uv pip install opencv-python numpy mediapipe tensorflow
```

### Step-by-Step Testing Process

#### 1. **Start the Application**
```bash
uv run webcam_gesture_tester.py
```

#### 2. **Testing a Specific Gesture**
1. Press a number key (1-9) to select the gesture you want to test
2. Perform the gesture in front of the camera
3. Wait for the sequence to fill (30 frames)
4. Press **[Space]** when you're performing the gesture to record the result
5. The system will show ✓ CORRECT or ✗ WRONG

**Example:**
```
→ Testing mode: 01_palm
✓ CORRECT: Predicted=01_palm, True=01_palm, Conf=95.3%
```

#### 3. **View Live Statistics**
- Bottom panel shows overall accuracy and error rate
- Statistics update in real-time as you test

#### 4. **Save Results**
- Press **[S]** to save statistics to a file
- File format: `gesture_test_results_YYYYMMDD_HHMMSS.txt`

#### 5. **Reset Statistics**
- Press **[R]** to clear all statistics and start fresh

## Keyboard Controls

| Key | Action |
|-----|--------|
| **1-9** | Enter testing mode for gesture (see gesture mapping) |
| **Space** | Record current prediction as test result |
| **R** | Reset all statistics |
| **S** | Save statistics to file |
| **T** | Exit testing mode |
| **Q** or **ESC** | Quit application |

## Gesture Mapping

The gesture mapping is loaded from `gesture_mapping.json`. Typical gestures:

| Key | Gesture |
|-----|---------|
| **1** | 01_palm |
| **2** | 02_l |
| **3** | 03_fist |
| **4** | 04_fist_moved |
| **5** | 05_thumb |
| **6** | 06_index |
| **7** | 07_ok |
| **8** | 08_palm_moved |
| **9** | 09_c |

## Understanding the Interface

### Main Display
```
┌─────────────────────────────────────────────────┐
│ Gesture: 01_palm          FPS: 30.0            │
│ Confidence: 95.3%         Testing: 01_palm     │
│ ████████████████████░░░░░░░                    │
└─────────────────────────────────────────────────┘
```

### All Predictions Panel (Right Side)
```
┌─────────────────────────────┐
│ All Predictions:            │
│ 01_palm: 95.3% ████████████ │
│ 02_l:    3.2%  █            │
│ 03_fist: 1.5%  ░            │
│ ...                         │
└─────────────────────────────┘
```

### Statistics Panel (Bottom)
```
┌─────────────────────────────────────────────────┐
│ Overall Accuracy: 94.5%  Avg Confidence: 92.1% │
│ Error Rate: 5.5%                               │
│ Total Tests: 150                               │
└─────────────────────────────────────────────────┘
```

## Output Files

### Statistics File Format
```
==============================================================
Hand Gesture Recognition Testing Results
==============================================================

Test Date: 2025-11-03 14:30:45
Model: <keras model>

Overall Accuracy: 94.50%
Error Rate: 5.50%
Total Tests: 150
Total Correct: 142

Per-Gesture Statistics:
------------------------------------------------------------

01_palm:
  Accuracy: 96.00% (24/25)
  Average Confidence: 94.30%
  Error Rate: 4.00%

02_l:
  Accuracy: 92.00% (23/25)
  Average Confidence: 89.50%
  Error Rate: 8.00%
...
```

## Tips for Accurate Testing

### 1. **Lighting**
- Ensure good, even lighting
- Avoid backlighting
- Natural light or soft white light works best

### 2. **Camera Position**
- Position camera at chest/head height
- Keep hand 1-2 feet from camera
- Ensure entire hand is visible

### 3. **Hand Position**
- Face palm toward camera
- Keep hand steady for a moment
- Avoid rapid movements during testing

### 4. **Testing Process**
- Test each gesture multiple times (20+ samples)
- Test in different positions and angles
- Record both correct and incorrect predictions

### 5. **Background**
- Use a plain background if possible
- Avoid cluttered backgrounds
- Remove other hands from frame

## Troubleshooting

### Problem: "No hand detected"
**Solution:**
- Ensure your hand is clearly visible
- Check lighting conditions
- Move hand closer to camera
- Ensure palm is facing camera

### Problem: Low confidence scores
**Solution:**
- Perform gesture more clearly
- Hold position steady
- Ensure hand landmarks are detected (green overlay visible)
- Check if gesture matches training data

### Problem: Wrong predictions
**Solution:**
- Ensure you're performing the gesture correctly
- Check gesture mapping in startup message
- Verify model was trained properly
- Test with different lighting/angles

### Problem: Application is slow/laggy
**Solution:**
- Close other applications
- Reduce camera resolution in code
- Ensure TensorFlow is using GPU if available

### Problem: Model file not found
**Solution:**
```bash
# Run the notebook first to train the model
jupyter notebook hand_gesture_recognition.ipynb
# Or in VS Code, open and run all cells
```

## Performance Metrics Explained

### Overall Accuracy
```
Accuracy = (Correct Predictions / Total Tests) × 100%
```
Target: >90% for good model performance

### Error Rate
```
Error Rate = 1 - Accuracy = (Wrong Predictions / Total Tests) × 100%
```
Target: <10% for good model performance

### Confidence Score
- Probability the model assigns to its prediction
- Range: 0-100%
- Higher is better
- >80% is excellent
- 50-80% is moderate
- <50% is uncertain

### Per-Gesture Accuracy
- Individual accuracy for each gesture
- Helps identify which gestures need improvement
- Useful for targeted retraining

## Advanced Features

### Custom Gesture Mapping
Edit `gesture_mapping.json` to add/modify gestures:
```json
{
  "custom_gesture_1": 0,
  "custom_gesture_2": 1,
  ...
}
```

### Adjusting Confidence Thresholds
Edit the `get_confidence_color()` method in the script:
```python
def get_confidence_color(self, confidence):
    if confidence > 0.9:  # Change from 0.8
        return self.color_good
    ...
```

### Changing Sequence Length
Modify `SEQUENCE_LENGTH` in `main()` function (must match training):
```python
SEQUENCE_LENGTH = 30  # Match your training configuration
```

## Integration with Training Pipeline

The webcam tester automatically integrates with your training:

1. **Train Model** (Notebook)
   - Run `hand_gesture_recognition.ipynb`
   - Saves `hand_gesture_lstm_model.h5`
   - Saves `gesture_mapping.json`

2. **Test Model** (Webcam Tester)
   - Loads model and mapping automatically
   - Provides real-time testing interface
   - Exports detailed statistics

3. **Analyze Results**
   - Review saved statistics files
   - Identify poorly performing gestures
   - Retrain if needed

## Example Workflow

```bash
# 1. Train the model
jupyter notebook hand_gesture_recognition.ipynb
# (Run all cells)

# 2. Test the model
python webcam_gesture_tester.py

# 3. Perform tests
# Press [1] for palm gesture
# Perform palm gesture
# Press [Space] 20 times while performing gesture

# 4. Test other gestures
# Press [2-9] for other gestures
# Repeat testing process

# 5. Save results
# Press [S] to save statistics

# 6. Review results
# Check gesture_test_results_*.txt file

# 7. Retrain if needed based on results
```

## Support

For issues or questions:
1. Check this README
2. Review TECHNICAL_REPORT.md for model details
3. Check notebook for training configuration
4. Verify all dependencies are installed

## License

See LICENSE.md for license information.

---
