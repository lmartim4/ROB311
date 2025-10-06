import gradio as gr
import cv2
import numpy as np
import time
from skimage.feature import local_binary_pattern, hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Emotion labels (FER-2013 format)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ============================================================================
# MODEL 1: LBP + KNN (Load Pretrained)
# ============================================================================
class LBPKNNModel:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
    def extract_lbp_features(self, image):
        """Extract LBP features from grayscale image"""
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist
    
    def load_model(self, model_path='models/lbp_knn_model.pkl'):
        """Load pretrained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_loaded = True
            print("✓ LBP+KNN model loaded successfully!")
        except FileNotFoundError:
            print("✗ LBP+KNN model not found. Please download it first (see README).")
            print("  Using fallback random predictions for demo purposes.")
            self.is_loaded = False
    
    def predict(self, face_gray):
        if not self.is_loaded or self.model is None:
            # Fallback: return random-ish predictions
            proba = np.random.dirichlet(np.ones(7))
            return proba
        features = self.extract_lbp_features(face_gray).reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        return proba

# ============================================================================
# MODEL 2: HOG + Linear SVM (Load Pretrained)
# ============================================================================
class HOGSVMModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
    def extract_hog_features(self, image):
        """Extract HOG features from grayscale image"""
        features = hog(image, orientations=9, pixels_per_cell=(6, 6),
                      cells_per_block=(2, 2), visualize=False, feature_vector=True)
        return features
    
    def load_model(self, model_path='models/hog_svm_model.pkl', scaler_path='models/hog_scaler.pkl'):
        """Load pretrained model and scaler"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_loaded = True
            print("✓ HOG+SVM model loaded successfully!")
        except FileNotFoundError:
            print("✗ HOG+SVM model not found. Please download it first (see README).")
            print("  Using fallback random predictions for demo purposes.")
            self.is_loaded = False
    
    def predict(self, face_gray):
        if not self.is_loaded or self.model is None:
            # Fallback: return random-ish predictions
            proba = np.random.dirichlet(np.ones(7))
            return proba
        features = self.extract_hog_features(face_gray).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        decision = self.model.decision_function(features_scaled)[0]
        # Convert to probabilities
        exp_scores = np.exp(decision - np.max(decision))
        proba = exp_scores / np.sum(exp_scores)
        return proba

# ============================================================================
# MODEL 3: mini-Xception CNN - ORIGINAL ARCHITECTURE from oarriaga
# ============================================================================
def create_mini_xception_original(input_shape=(64, 64, 1), num_classes=7):
    """
    Original mini-Xception architecture from oarriaga/face_classification
    https://github.com/oarriaga/face_classification
    Adapted for newer Keras versions
    """
    regularization = keras.regularizers.l2(0.01)
    
    img_input = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization)(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # module 1
    residual = layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    x = layers.SeparableConv2D(16, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(16, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    x = layers.SeparableConv2D(32, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    x = layers.SeparableConv2D(64, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(64, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    x = layers.SeparableConv2D(128, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(128, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = layers.Conv2D(num_classes, (3, 3), padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Activation('softmax', name='predictions')(x)

    model = keras.Model(img_input, output)
    return model

class MiniXceptionModel:
    def __init__(self):
        # Use the ORIGINAL architecture (64x64 input)
        self.model = create_mini_xception_original(input_shape=(64, 64, 1), num_classes=7)
        self.is_loaded = False
        
    def load_model(self, weights_path='models/fer2013_mini_XCEPTION.119-0.65.hdf5'):
        """Load pretrained weights from oarriaga repository"""
        try:
            # Try loading the complete model first (includes architecture)
            print(f"🔍 Trying to load complete model from: {weights_path}")
            self.model = keras.models.load_model(weights_path, compile=False)
            self.is_loaded = True
            print("✓ mini-Xception model loaded successfully (complete model)!")
            return
        except Exception as e:
            print(f"⚠ Could not load complete model: {e}")
            print("🔍 Trying to load weights only...")
        
        try:
            # Try loading weights with skip_mismatch
            self.model.load_weights(weights_path, skip_mismatch=True, by_name=True)
            self.is_loaded = True
            print("✓ mini-Xception weights loaded successfully (partial match)!")
        except Exception as e:
            print(f"✗ mini-Xception weights not found: {e}")
            print("  Please download from: https://github.com/oarriaga/face_classification")
            print("  Using fallback random predictions for demo purposes.")
            self.is_loaded = False
    
    def predict(self, face_gray):
        if not self.is_loaded:
            # Fallback: return random-ish predictions
            proba = np.random.dirichlet(np.ones(7))
            return proba
        
        # Get the actual input shape from the model
        input_shape = self.model.input_shape[1:3]  # (height, width)
        
        # Resize to match model's expected input
        face_resized = cv2.resize(face_gray, (input_shape[1], input_shape[0]))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)
        proba = self.model.predict(face_input, verbose=0)[0]
        return proba

# ============================================================================
# Initialize and Load Models
# ============================================================================
print("\n" + "="*60)
print("LOADING PRETRAINED MODELS")
print("="*60 + "\n")

lbp_knn = LBPKNNModel()
lbp_knn.load_model()

hog_svm = HOGSVMModel()
hog_svm.load_model()

mini_xception = MiniXceptionModel()
mini_xception.load_model()

print("\n" + "="*60)
print("READY! If models are missing, they'll use random predictions.")
print("See README.md for download instructions.")
print("="*60 + "\n")

# ============================================================================
# Face Detection and Preprocessing
# ============================================================================
def detect_and_preprocess_face(image):
    if image is None:
        return None, "No image provided"
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, "No face detected"
    
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (48, 48))
    
    image_with_box = image.copy()
    cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return face_resized, image_with_box

# ============================================================================
# Prediction Function
# ============================================================================
def predict_emotion(image):
    if image is None:
        return None, "Please provide an image", "", ""
    
    face_48x48, result = detect_and_preprocess_face(image)
    
    if face_48x48 is None:
        return result, result, "", ""
    
    # Model 1: LBP + KNN
    start_time = time.time()
    proba_lbp = lbp_knn.predict(face_48x48)
    latency_lbp = (time.time() - start_time) * 1000
    
    pred_idx_lbp = np.argmax(proba_lbp)
    confidence_lbp = proba_lbp[pred_idx_lbp]
    
    status_lbp = "✓ Using Pretrained Model" if lbp_knn.is_loaded else "⚠ Using Random Predictions (Model Not Loaded)"
    
    result_lbp = "<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>"
    result_lbp += "<h2>LBP + KNN (Baseline)</h2>"
    result_lbp += "<p style='font-size: 12px; opacity: 0.8;'>" + status_lbp + "</p>"
    result_lbp += "<h1 style='font-size: 48px; margin: 10px 0;'>" + EMOTIONS[pred_idx_lbp] + "</h1>"
    result_lbp += "<p style='font-size: 24px;'>Confidence: " + str(round(confidence_lbp*100, 1)) + "%</p>"
    result_lbp += "<p style='font-size: 18px;'>Latency: " + str(round(latency_lbp, 2)) + " ms</p>"
    result_lbp += "<hr style='margin: 15px 0;'><div style='text-align: left;'>"
    for i, emotion in enumerate(EMOTIONS):
        result_lbp += "<p>" + emotion + ": " + str(round(proba_lbp[i]*100, 1)) + "%</p>"
    result_lbp += "</div></div>"
    
    # Model 2: HOG + SVM
    start_time = time.time()
    proba_hog = hog_svm.predict(face_48x48)
    latency_hog = (time.time() - start_time) * 1000
    
    pred_idx_hog = np.argmax(proba_hog)
    confidence_hog = proba_hog[pred_idx_hog]
    
    status_hog = "✓ Using Pretrained Model" if hog_svm.is_loaded else "⚠ Using Random Predictions (Model Not Loaded)"
    
    result_hog = "<div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white;'>"
    result_hog += "<h2>HOG + Linear SVM (OvR)</h2>"
    result_hog += "<p style='font-size: 12px; opacity: 0.8;'>" + status_hog + "</p>"
    result_hog += "<h1 style='font-size: 48px; margin: 10px 0;'>" + EMOTIONS[pred_idx_hog] + "</h1>"
    result_hog += "<p style='font-size: 24px;'>Confidence: " + str(round(confidence_hog*100, 1)) + "%</p>"
    result_hog += "<p style='font-size: 18px;'>Latency: " + str(round(latency_hog, 2)) + " ms</p>"
    result_hog += "<hr style='margin: 15px 0;'><div style='text-align: left;'>"
    for i, emotion in enumerate(EMOTIONS):
        result_hog += "<p>" + emotion + ": " + str(round(proba_hog[i]*100, 1)) + "%</p>"
    result_hog += "</div></div>"
    
    # Model 3: mini-Xception (uses 64x64 internally)
    start_time = time.time()
    proba_cnn = mini_xception.predict(face_48x48)
    latency_cnn = (time.time() - start_time) * 1000
    
    pred_idx_cnn = np.argmax(proba_cnn)
    confidence_cnn = proba_cnn[pred_idx_cnn]
    
    status_cnn = "✓ Using Pretrained Model" if mini_xception.is_loaded else "⚠ Using Random Predictions (Model Not Loaded)"
    
    result_cnn = "<div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; color: white;'>"
    result_cnn += "<h2>mini-Xception (CNN)</h2>"
    result_cnn += "<p style='font-size: 12px; opacity: 0.8;'>" + status_cnn + "</p>"
    result_cnn += "<h1 style='font-size: 48px; margin: 10px 0;'>" + EMOTIONS[pred_idx_cnn] + "</h1>"
    result_cnn += "<p style='font-size: 24px;'>Confidence: " + str(round(confidence_cnn*100, 1)) + "%</p>"
    result_cnn += "<p style='font-size: 18px;'>Latency: " + str(round(latency_cnn, 2)) + " ms</p>"
    result_cnn += "<hr style='margin: 15px 0;'><div style='text-align: left;'>"
    for i, emotion in enumerate(EMOTIONS):
        result_cnn += "<p>" + emotion + ": " + str(round(proba_cnn[i]*100, 1)) + "%</p>"
    result_cnn += "</div></div>"
    
    return result, result_lbp, result_hog, result_cnn

# ============================================================================
# Gradio Interface
# ============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# Multi-Model Emotion Recognition System")
    gr.Markdown("### Compare LBP+KNN, HOG+SVM, and mini-Xception CNN side-by-side")
    gr.Markdown("**Pipeline:** Face Detection (Haar Cascade) -> Crop/Align -> Resize -> Parallel Inference")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="numpy", sources=["upload", "webcam"])
            predict_btn = gr.Button("Analyze Emotion", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="Detected Face", type="numpy")
    
    gr.Markdown("## Model Predictions")
    
    with gr.Row():
        output_lbp = gr.HTML(label="LBP + KNN")
        output_hog = gr.HTML(label="HOG + SVM")
        output_cnn = gr.HTML(label="mini-Xception")
    
    gr.Markdown("### ⚠️ Important:")
    gr.Markdown("If models show 'Using Random Predictions', please download the pretrained models following the instructions in **README.md**")
    gr.Markdown("mini-Xception uses the ORIGINAL architecture from oarriaga (64x64 input)")
    
    predict_btn.click(
        fn=predict_emotion,
        inputs=[input_image],
        outputs=[output_image, output_lbp, output_hog, output_cnn]
    )

if __name__ == "__main__":
    demo.launch()