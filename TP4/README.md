# ROB311 - Multi-Model Emotion Recognition System

A comparative study of three emotion recognition approaches: **LBP+KNN**, **HOG+SVM**, and **mini-Xception CNN**.

##
🎥 [Demo Video](https://www.youtube.com/watch?v=f6zI0opmQNE)

---

## 📊 Methodology Overview

This project implements a complete emotion recognition pipeline comparing classical machine learning with deep learning approaches on facial expression recognition.

### Pipeline Architecture

```
Input Image → Face Detection → Crop & Align → Resize → Feature Extraction → Classification → Output
```

**Steps:**
1. **Face Detection**: OpenCV Haar Cascade detects faces in the input image
2. **Preprocessing**: Crop and resize detected face to standard dimensions
3. **Parallel Inference**: Run all three models simultaneously
4. **Output**: Display predictions with confidence scores and latency metrics

---

## 🔬 Model Descriptions

### 1. LBP + K-Nearest Neighbors (Baseline)

**Local Binary Patterns (LBP)** is a texture descriptor that compares each pixel with its neighbors.

**How it works:**
1. For each pixel, compare with 8-16 surrounding neighbors
2. Create binary code: `1` if neighbor ≥ center, `0` otherwise
3. Convert binary pattern to histogram (compact feature vector)
4. Use K-Nearest Neighbors (K=7) to classify based on feature similarity

**Training Process:**
```python
# Extract LBP features from training images
for each image in FER-2013:
    lbp_histogram = extract_lbp_features(image)
    features.append(lbp_histogram)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(features, labels)
```

**Characteristics:**
- **Speed**: Very fast (5-20ms)
- **Accuracy**: ~45% on FER-2013
- **Advantages**: Simple, interpretable, works on CPU
- **Use case**: Embedded systems, real-time applications

---

### 2. HOG + Linear SVM (Classical ML)

**Histogram of Oriented Gradients (HOG)** captures edge and gradient information.

**How it works:**
1. Compute gradients (edges) in the image
2. Divide image into cells (6×6 pixels)
3. For each cell, create histogram of gradient directions
4. Normalize across blocks and concatenate
5. Use Linear SVM (One-vs-Rest) for multi-class classification

**Training Process:**
```python
# Extract HOG features from training images
for each image in FER-2013:
    hog_features = extract_hog_features(image)
    features.append(hog_features)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train Linear SVM
svm = LinearSVC(C=0.1, class_weight='balanced')
svm.fit(features_scaled, labels)
```

**Characteristics:**
- **Speed**: Medium (10-35ms)
- **Accuracy**: ~50-55% on FER-2013
- **Advantages**: Better than LBP, still CPU-friendly
- **Use case**: Balanced performance/accuracy trade-off

---

### 3. mini-Xception CNN (Deep Learning)

**mini-Xception** is a lightweight CNN based on depthwise separable convolutions.

**Architecture:**
```
Input (64×64×1)
    ↓
Entry Flow: 2× Conv2D + BatchNorm + ReLU
    ↓
Module 1: SeparableConv2D (16 filters) + Residual Connection
    ↓
Module 2: SeparableConv2D (32 filters) + Residual Connection
    ↓
Module 3: SeparableConv2D (64 filters) + Residual Connection
    ↓
Module 4: SeparableConv2D (128 filters) + Residual Connection
    ↓
Global Average Pooling → Dense(7) → Softmax
```

**Key Features:**
- **Depthwise Separable Convolutions**: Reduce parameters while maintaining accuracy
- **Residual Connections**: Skip connections help training deeper networks
- **Batch Normalization**: Stabilizes training and improves convergence

**Training Process:**
```python
# Data augmentation
augmentation = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Train the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=100, batch_size=32)
```

**Characteristics:**
- **Accuracy**: ~65-67% on FER-2013
- **Advantages**: Best accuracy, learns hierarchical features
- **Use case**: When accuracy is priority over speed

---

## 🎯 Dataset: FER-2013

**Facial Expression Recognition 2013**
- **Size**: 35,887 grayscale images
- **Resolution**: 48×48 pixels
- **Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Split**: 
  - Training: ~28,709 images
  - Validation: ~3,589 images
  - Test: ~3,589 images
- **Source**: Kaggle Facial Expression Recognition Challenge

**Class Distribution:**
- Happy: ~8,989 (25%)
- Neutral: ~6,198 (17%)
- Sad: ~6,077 (17%)
- Angry: ~4,953 (14%)
- Surprise: ~4,002 (11%)
- Fear: ~5,121 (14%)
- Disgust: ~547 (2%) ⚠️ Highly imbalanced!

---

## 🔧 Training the Classical Models

### Training LBP+KNN

```python
# 1. Load FER-2013 dataset
df = pd.read_csv('fer2013.csv')

# 2. Extract LBP features
features = []
for pixels in df['pixels']:
    img = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
    lbp_hist = extract_lbp_features(img)  # Returns histogram
    features.append(lbp_hist)

# 3. Train KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(features, labels)

# 4. Save model
import pickle
with open('models/lbp_knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
```

**Output**: `lbp_knn_model.pkl` (~500KB)

### Training HOG+SVM

```python
# 1. Extract HOG features
features = []
for img in images:
    hog_feat = hog(img, orientations=9, pixels_per_cell=(6,6))
    features.append(hog_feat)

# 2. Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. Train SVM
from sklearn.svm import LinearSVC
svm = LinearSVC(C=0.1, max_iter=2000, class_weight='balanced')
svm.fit(features_scaled, labels)

# 4. Save both model and scaler
with open('models/hog_svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
with open('models/hog_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

**Output**: `hog_svm_model.pkl` (~2MB), `hog_scaler.pkl` (~100KB)

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install gradio opencv-python numpy scikit-image scikit-learn tensorflow

# Create models directory
mkdir models
```

### Download Pretrained Models

**Option 1: mini-Xception only (fastest)**
```bash
cd models
wget https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.119-0.65.hdf5
cd ..
```

**Option 2: Train all models yourself**
1. Download FER-2013: https://kaggle.com/datasets/msambare/fer2013
2. Run `train_classical_models.py` (creates LBP+KNN and HOG+SVM)
3. Download mini-Xception weights (above)

### Run the Application

```bash
python main.py
```

Open browser at `http://localhost:7860`

---

## 📁 Project Structure

```
ROB311/
├── main.py                        # Main Gradio application
├── train_classical_models.py     # Training script for LBP+KNN & HOG+SVM
├── models/                        # Pretrained models
│   ├── fer2013_mini_XCEPTION.hdf5
│   ├── lbp_knn_model.pkl
│   ├── hog_svm_model.pkl
│   └── hog_scaler.pkl
├── fer2013.csv                    # Dataset (optional, for training)
└── README.md
```