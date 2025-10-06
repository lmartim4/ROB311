"""
Train LBP+KNN and HOG+SVM models on FER-2013 dataset
Run this script to generate: lbp_knn_model.pkl, hog_svm_model.pkl, hog_scaler.pkl
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, hog
import os

print("="*60)
print("TRAINING CLASSICAL MODELS ON FER-2013")
print("="*60)

# Check if FER-2013 exists
if not os.path.exists('fer2013.csv'):
    print("\n❌ ERROR: fer2013.csv not found!")
    print("\nPlease download FER-2013 dataset:")
    print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Download fer2013.csv")
    print("3. Place it in the same directory as this script")
    print("\nAlternatively, you can use the FER-2013 from:")
    print("https://www.kaggle.com/datasets/deadskull7/fer2013")
    exit(1)

# Load FER-2013
print("\n📂 Loading FER-2013 dataset...")
df = pd.read_csv('fer2013.csv')
print(f"✓ Loaded {len(df)} images")

# Parse images
X_train = []
y_train = df['emotion'].values

print("\n🖼️  Parsing images...")
for pixels in df['pixels']:
    img = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
    X_train.append(img)

X_train = np.array(X_train)
print(f"✓ Dataset shape: {X_train.shape}")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# ============================================================================
# Train LBP + KNN
# ============================================================================
print("\n" + "="*60)
print("TRAINING LBP + KNN MODEL")
print("="*60)

print("Extracting LBP features... (this may take a few minutes)")
X_lbp = []
for i, img in enumerate(X_train):
    if i % 5000 == 0:
        print(f"  Progress: {i}/{len(X_train)}")
    
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    X_lbp.append(hist)

X_lbp = np.array(X_lbp)
print(f"✓ LBP features extracted: {X_lbp.shape}")

print("\nTraining KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
knn.fit(X_lbp, y_train)

print("Saving model...")
with open('models/lbp_knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("✓ LBP+KNN model saved to models/lbp_knn_model.pkl")

# Quick accuracy test
from sklearn.model_selection import train_test_split
X_lbp_train, X_lbp_test, y_lbp_train, y_lbp_test = train_test_split(X_lbp, y_train, test_size=0.2, random_state=42)
knn_test = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
knn_test.fit(X_lbp_train, y_lbp_train)
accuracy = knn_test.score(X_lbp_test, y_lbp_test)
print(f"📊 Estimated accuracy: {accuracy*100:.2f}%")

# ============================================================================
# Train HOG + SVM
# ============================================================================
print("\n" + "="*60)
print("TRAINING HOG + SVM MODEL")
print("="*60)

print("Extracting HOG features... (this may take a few minutes)")
X_hog = []
for i, img in enumerate(X_train):
    if i % 5000 == 0:
        print(f"  Progress: {i}/{len(X_train)}")
    
    features = hog(img, orientations=9, pixels_per_cell=(6, 6),
                   cells_per_block=(2, 2), feature_vector=True)
    X_hog.append(features)

X_hog = np.array(X_hog)
print(f"✓ HOG features extracted: {X_hog.shape}")

print("\nScaling features...")
scaler = StandardScaler()
X_hog_scaled = scaler.fit_transform(X_hog)

print("Training SVM classifier... (this may take 5-10 minutes)")
svm = LinearSVC(C=0.1, max_iter=2000, class_weight='balanced', verbose=1)
svm.fit(X_hog_scaled, y_train)

print("\nSaving models...")
with open('models/hog_svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
with open('models/hog_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✓ HOG+SVM model saved to models/hog_svm_model.pkl")
print("✓ Scaler saved to models/hog_scaler.pkl")

# Quick accuracy test
X_hog_train, X_hog_test, y_hog_train, y_hog_test = train_test_split(X_hog_scaled, y_train, test_size=0.2, random_state=42)
svm_test = LinearSVC(C=0.1, max_iter=2000, class_weight='balanced')
svm_test.fit(X_hog_train, y_hog_train)
accuracy = svm_test.score(X_hog_test, y_hog_test)
print(f"📊 Estimated accuracy: {accuracy*100:.2f}%")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  ✓ models/lbp_knn_model.pkl")
print("  ✓ models/hog_svm_model.pkl")
print("  ✓ models/hog_scaler.pkl")
print("\nYou can now run emotion_recognition.py with all models loaded!")
print("="*60)