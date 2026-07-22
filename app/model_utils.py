import os
import numpy as np
import pandas as pd
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "stage2_best.keras")
TABULAR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression_pipeline.pkl")

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
]

DISEASE_YIELD_ADJUSTMENT = {
    "Corn_(maize)___healthy": 1.00,
    "Corn_(maize)___Common_rust_": 0.85,
    "Corn_(maize)___Northern_Leaf_Blight": 0.80,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 0.85,
}

print("Loading CNN model...")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
print("Loading tabular model...")
tabular_model = joblib.load(TABULAR_MODEL_PATH)
print("Models loaded successfully.")


def preprocess_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)


def predict_disease_probs(image_path: str):
    img_batch = preprocess_image(image_path)
    return cnn_model.predict(img_batch, verbose=0)[0]


def predict_base_yield(tabular_features: dict) -> float:
    X_input = pd.DataFrame([tabular_features])
    return float(tabular_model.predict(X_input)[0])


def fuse_prediction(image_path: str, tabular_features: dict) -> dict:
    all_probs = predict_disease_probs(image_path)
    predicted_idx = np.argmax(all_probs)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(all_probs[predicted_idx])

    base_yield = predict_base_yield(tabular_features)

    p_healthy = float(all_probs[CLASS_NAMES.index("Corn_(maize)___healthy")])
    severity_scale = 1 - p_healthy

    max_reduction = 1 - DISEASE_YIELD_ADJUSTMENT[predicted_class]
    actual_reduction = max_reduction * severity_scale
    factor = 1 - actual_reduction

    final_yield = base_yield * factor

    return {
        "predicted_disease": predicted_class,
        "disease_confidence": round(confidence, 4),
        "p_healthy": round(p_healthy, 4),
        "severity_scale": round(severity_scale, 4),
        "base_yield_tons_per_hectare": round(float(base_yield), 3),
        "yield_adjustment_factor": round(float(factor), 4),
        "final_yield_tons_per_hectare": round(float(final_yield), 3),
    }