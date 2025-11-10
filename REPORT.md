# Technical Report: Hand Gesture Recognition using LSTM Networks

## Table of Contents

1. [Introduction](#1-introduction)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Hand Pose Estimation](#3-hand-pose-estimation)
4. [Data Augmentation](#4-data-augmentation)
5. [LSTM Architecture](#5-lstm-architecture)
6. [Training Methodology](#6-training-methodology)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Results and Analysis](#8-results-and-analysis)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Introduction

### 1.1 Problem Statement

Hand gesture recognition is a computer vision task that involves identifying specific hand movements from video sequences. The challenge lies in capturing both spatial (hand pose) and temporal (movement over time) information.

### 1.2 Approach Overview

Our solution employs a two-stage pipeline:
1. **Spatial Feature Extraction**: MediaPipe extracts 21 3D hand landmarks
2. **Temporal Pattern Recognition**: LSTM networks classify gesture sequences

---

## 2. Mathematical Foundations

### 2.1 Feature Space

Each hand pose is represented as a point in 63-dimensional space:

$$\mathbf{x}_t = [x_1, y_1, z_1, x_2, y_2, z_2, \ldots, x_{21}, y_{21}, z_{21}] \in \mathbb{R}^{63}$$

Where:
- $t$ represents the time step (frame number)
- $(x_i, y_i, z_i)$ are normalized 3D coordinates of landmark $i$
- Coordinates are normalized to $[0, 1]$ range relative to hand bounding box

### 2.2 Sequence Representation

A gesture sequence is a temporal series:

$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T] \in \mathbb{R}^{T \times 63}$$

Where:
- $T$ is the sequence length (30 frames in our implementation)
- Each $\mathbf{x}_t$ is a snapshot of hand pose at time $t$

### 2.3 Classification Objective

Given a sequence $\mathbf{X}$, predict the gesture class $y \in \{1, 2, \ldots, C\}$ where $C$ is the number of gesture classes (10 in our case):

$$\hat{y} = \arg\max_{c} P(y = c | \mathbf{X})$$

---

## 3. Hand Pose Estimation

### 3.1 MediaPipe Hands Algorithm

MediaPipe uses a two-stage approach:

#### Palm Detection
A lightweight CNN detects hand regions in the image:

$$\mathbf{B} = \text{PalmDetector}(\mathbf{I})$$

Where:
- $\mathbf{I}$ is the input image
- $\mathbf{B}$ is the bounding box containing the hand

#### Landmark Regression
A second CNN regresses 21 3D landmarks within the detected region:

$$\{\mathbf{p}_i\}_{i=1}^{21} = \text{LandmarkRegressor}(\mathbf{I}_{\mathbf{B}})$$

Where $\mathbf{p}_i = (x_i, y_i, z_i)$ represents the 3D position of landmark $i$.

### 3.2 Landmark Normalization

Raw coordinates are normalized to maintain scale invariance:

$$x_i' = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}$$

$$y_i' = \frac{y_i - y_{\min}}{y_{\max} - y_{\min}}$$

$$z_i' = \frac{z_i - z_{\min}}{z_{\max} - z_{\min}}$$

This ensures the model is invariant to hand size and distance from camera.

### 3.3 Hand Landmark Topology

The 21 landmarks follow a hierarchical structure:
- **Wrist** (landmark 0): Root of the kinematic chain
- **Thumb** (landmarks 1-4): 4 joints
- **Index to Pinky** (landmarks 5-20): 4 fingers × 4 joints each

---

## 4. Data Augmentation

Data augmentation artificially increases dataset diversity to improve generalization.

### 4.1 Random Noise Addition

Gaussian noise simulates sensor uncertainty:

$$\mathbf{x}_t' = \mathbf{x}_t + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

Where:
- $\sigma = 0.02$ (noise factor)
- $\mathcal{N}(0, \sigma^2)$ is a Gaussian distribution
- $\mathbf{I}$ is the identity matrix

**Purpose**: Models hand tracking jitter and sensor noise.

### 4.2 Random Occlusion

Randomly masks landmarks to simulate partial hand visibility:

$$x_{t,i}' = \begin{cases} 
x_{t,i} & \text{with probability } (1-p_{\text{occlude}}) \\
0 & \text{with probability } p_{\text{occlude}}
\end{cases}$$

Where $p_{\text{occlude}} = 0.1$.

**Purpose**: Improves robustness to missing detections.

### 4.3 Random Scaling

Simulates varying hand sizes:

$$\mathbf{x}_t' = s \cdot \mathbf{x}_t, \quad s \sim \mathcal{U}(0.8, 1.2)$$

Where $\mathcal{U}(a, b)$ is a uniform distribution.

**Purpose**: Creates scale invariance for different hand sizes.

### 4.4 Random Translation

Simulates hand position variations:

$$\mathbf{x}_t' = \mathbf{x}_t + \mathbf{t}$$

Where $\mathbf{t} = [t_x, t_y, t_z]$ and $t_i \sim \mathcal{U}(-0.1, 0.1)$.

**Purpose**: Models variations in hand placement within frame.

### 4.5 Augmentation Strategy

For each original sequence, we generate 2 augmented versions by randomly applying transformations:

$$\text{Dataset Size} = N_{\text{original}} \times (1 + N_{\text{aug}}) = N_{\text{original}} \times 3$$

---

## 5. LSTM Architecture

### 5.1 Long Short-Term Memory (LSTM) Basics

LSTM cells address the vanishing gradient problem in standard RNNs through gating mechanisms.

#### Cell State Update

The LSTM maintains a cell state $\mathbf{c}_t$ that acts as a memory:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

Where:
- $\odot$ denotes element-wise multiplication
- $\mathbf{f}_t$ is the forget gate (what to forget from previous state)
- $\mathbf{i}_t$ is the input gate (what new information to add)
- $\tilde{\mathbf{c}}_t$ is the candidate cell state

#### Gate Equations

**Forget Gate** (decides what to discard):
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input Gate** (decides what to add):
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate Cell State** (new information):
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Output Gate** (decides what to output):
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

**Hidden State** (current output):
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Where $\sigma$ is the sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$

### 5.2 Our LSTM Architecture

#### Layer 1: First LSTM Layer
- **Input**: $(T, 63)$ where $T=30$
- **Units**: 128
- **Output**: $(T, 128)$ (returns sequences)

$$\mathbf{H}^{(1)} = \text{LSTM}_{128}(\mathbf{X})$$

#### Layer 2: Batch Normalization
Normalizes layer inputs to stabilize training:

$$\mathbf{H}^{(1)'} = \frac{\mathbf{H}^{(1)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$

Where:
- $\mu_B$ is batch mean
- $\sigma_B^2$ is batch variance
- $\gamma, \beta$ are learnable parameters
- $\epsilon = 10^{-7}$ prevents division by zero

#### Layer 3: Dropout
Randomly sets neurons to zero to prevent overfitting:

$$h_i' = \begin{cases}
0 & \text{with probability } p \\
\frac{h_i}{1-p} & \text{with probability } (1-p)
\end{cases}$$

Where $p = 0.3$ (dropout rate).

#### Layers 4-9: Additional LSTM, BatchNorm, Dropout
Following the same pattern with:
- Second LSTM: 128 units (returns sequences)
- Third LSTM: 64 units (returns single vector)

#### Layer 10-14: Dense Layers
**Dense Layer 1**: 
$$\mathbf{z}^{(1)} = \text{ReLU}(\mathbf{W}^{(1)} \mathbf{h}_T + \mathbf{b}^{(1)})$$

Where $\mathbf{W}^{(1)} \in \mathbb{R}^{256 \times 64}$ and ReLU is:
$$\text{ReLU}(x) = \max(0, x)$$

**Dense Layer 2**:
$$\mathbf{z}^{(2)} = \text{ReLU}(\mathbf{W}^{(2)} \mathbf{z}^{(1)} + \mathbf{b}^{(2)})$$

Where $\mathbf{W}^{(2)} \in \mathbb{R}^{128 \times 256}$.

#### Output Layer: Softmax
$$P(y = c | \mathbf{X}) = \frac{e^{z_c^{(3)}}}{\sum_{j=1}^{C} e^{z_j^{(3)}}}$$

Where:
- $\mathbf{z}^{(3)} = \mathbf{W}^{(3)} \mathbf{z}^{(2)} + \mathbf{b}^{(3)}$
- $\mathbf{W}^{(3)} \in \mathbb{R}^{10 \times 128}$
- $C = 10$ (number of gesture classes)

### 5.3 Weight Initialization

Dense layers use He initialization for ReLU activations:

$$W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

Where $n_{\text{in}}$ is the number of input units.

---

## 6. Training Methodology

### 6.1 Loss Function

Sparse categorical cross-entropy:

$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \log P(y^{(n)} | \mathbf{X}^{(n)})$$

Where:
- $N$ is the batch size
- $y^{(n)}$ is the true label for sample $n$
- $P(y^{(n)} | \mathbf{X}^{(n)})$ is the predicted probability

### 6.2 Optimization Algorithm

Adam (Adaptive Moment Estimation) optimizer:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $g_t$ is the gradient at time $t$
- $m_t$ is the first moment (mean)
- $v_t$ is the second moment (variance)
- $\alpha = 0.001$ (learning rate)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- $\epsilon = 10^{-7}$

### 6.3 Learning Rate Scheduling

ReduceLROnPlateau reduces learning rate when validation loss plateaus:

$$\alpha_{\text{new}} = \alpha_{\text{old}} \times \text{factor}$$

Where factor = 0.5 if no improvement for 10 epochs.

### 6.4 Early Stopping

Training stops if validation loss doesn't improve for 20 epochs:

$$\text{Stop if } \min_{i \in [t-20, t]} \mathcal{L}_{\text{val}}^{(i)} = \mathcal{L}_{\text{val}}^{(t-20)}$$

### 6.5 Data Split

- **Training**: 64% of data
- **Validation**: 16% of data
- **Test**: 20% of data

Stratified split ensures equal class distribution:

$$\frac{N_c^{\text{train}}}{N^{\text{train}}} = \frac{N_c^{\text{total}}}{N^{\text{total}}} \quad \forall c \in \{1, \ldots, C\}$$

---

## 7. Evaluation Metrics

### 7.1 Accuracy

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

Where $\mathbb{1}[\cdot]$ is the indicator function.

### 7.2 Precision

For class $c$:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

Where:
- $TP_c$ = True Positives for class $c$
- $FP_c$ = False Positives for class $c$

### 7.3 Recall (Sensitivity)

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

Where $FN_c$ = False Negatives for class $c$.

### 7.4 F1-Score

Harmonic mean of precision and recall:

$$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 7.5 Confusion Matrix

$$C_{ij} = \sum_{n=1}^{N} \mathbb{1}[y^{(n)} = i \land \hat{y}^{(n)} = j]$$

Where:
- $C_{ij}$ is the number of samples of true class $i$ predicted as class $j$
- Diagonal elements $C_{ii}$ represent correct predictions

### 7.6 Per-Class Accuracy

$$\text{Accuracy}_c = \frac{C_{cc}}{\sum_{j=1}^{C} C_{cj}}$$

This measures how well the model performs on each specific gesture.

---

## 8. Results and Analysis

### 8.1 Model Capacity

Total parameters:

$$N_{\text{params}} = \sum_{\text{layers}} (n_{\text{in}} \times n_{\text{out}} + n_{\text{out}})$$

Approximate breakdown:
- **LSTM layers**: ~500K parameters
- **Dense layers**: ~40K parameters
- **Total**: ~540K trainable parameters

### 8.2 Training Dynamics

The model learns hierarchical representations:

**Early epochs** (1-20):
- Learning basic hand shapes
- High loss, low accuracy
- Rapid improvement in training metrics

**Middle epochs** (20-60):
- Learning temporal patterns
- Validation accuracy increases
- Learning rate may be reduced

**Late epochs** (60-100):
- Fine-tuning decision boundaries
- Minimal improvement
- Early stopping may trigger

### 8.3 Computational Complexity

**Forward Pass Time Complexity**:
$$\mathcal{O}(T \cdot d \cdot h + h \cdot D)$$

Where:
- $T = 30$ (sequence length)
- $d = 63$ (input features)
- $h = 128$ (hidden units)
- $D = 256 + 128 + 10$ (dense layer dimensions)

**Training Time Complexity**:
$$\mathcal{O}(E \cdot N \cdot (T \cdot d \cdot h + h \cdot D))$$

Where:
- $E \leq 100$ (max epochs)
- $N$ is dataset size

### 8.4 Advantages of LSTM for Gesture Recognition

1. **Temporal Modeling**: Captures sequential dependencies
2. **Variable-Length Handling**: Can process different sequence lengths
3. **Long-Range Dependencies**: Gates prevent vanishing gradients
4. **Feature Learning**: Automatically learns relevant temporal patterns

### 8.5 Potential Improvements

1. **Bidirectional LSTM**: Process sequences forward and backward
   $$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$

2. **Attention Mechanisms**: Weight important frames
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{i=1}^{T} \exp(e_i)}$$

3. **Transformer Architecture**: Self-attention for parallel processing

4. **Multi-Scale Features**: Combine different temporal resolutions

---

## 9. Conclusion

### 9.1 Summary

This project successfully implements a hand gesture recognition system using:
- MediaPipe for robust hand landmark extraction
- Data augmentation for improved generalization
- LSTM networks for temporal sequence classification
- Comprehensive evaluation metrics

### 9.2 Key Achievements

1. **End-to-End Pipeline**: From raw images to gesture predictions
2. **Real-Time Capability**: Processes webcam feed at interactive rates
3. **Robust Performance**: Handles variations in hand pose, scale, and position
4. **Efficient Training**: Converges in reasonable time with proper regularization

### 9.3 Mathematical Insights

- **Hierarchical Representation**: LSTM layers build increasingly abstract features
- **Gating Mechanisms**: Control information flow through time
- **Regularization**: BatchNorm and Dropout prevent overfitting
- **Augmentation**: Synthetic data improves generalization

### 9.4 Applications

- Sign language recognition
- Human-computer interaction
- Virtual reality controllers
- Assistive technologies

---

## 10. References

### Academic Papers

1. **MediaPipe Hands**:
   - Zhang, F., et al. (2020). "MediaPipe Hands: On-device Real-time Hand Tracking." arXiv preprint arXiv:2006.10214.

2. **LSTM Networks**:
   - Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation, 9(8), 1735-1780.

3. **Adam Optimizer**:
   - Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980.

4. **Batch Normalization**:
   - Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML.

5. **Dropout**:
   - Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR, 15(1), 1929-1958.

### Frameworks and Libraries

- **TensorFlow**: https://www.tensorflow.org/
- **MediaPipe**: https://google.github.io/mediapipe/
- **Scikit-learn**: https://scikit-learn.org/
- **OpenCV**: https://opencv.org/

### Dataset

- **LeapGestRecog Dataset**: from Kaggle (gti-upm/leapgestrecog)

---
