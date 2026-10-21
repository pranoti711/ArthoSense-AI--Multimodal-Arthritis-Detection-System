import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from modules.voice_model.preprocess_voice import preprocess_audio, find_voice_files
except ImportError:
    from preprocess_voice import preprocess_audio, find_voice_files

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "../../models/voice_model.h5")
DATA_DIR = os.path.join(BASE_DIR, "../../data/raw/voice/ravdess_kaggle")
SCALER_PATH = os.path.join(BASE_DIR, "voice_scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "voice_encoder.pkl")

_voice_model = None
_voice_scaler = None
_voice_encoder = None


def load_voice_model():
    global _voice_model
    if _voice_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Voice model not found at {MODEL_PATH}")
        _voice_model = tf.keras.models.load_model(MODEL_PATH)
    return _voice_model


def load_or_build_scaler_and_encoder():
    global _voice_scaler, _voice_encoder
    if _voice_scaler is not None and _voice_encoder is not None:
        return _voice_scaler, _voice_encoder

    if os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
        _voice_scaler = joblib.load(SCALER_PATH)
        _voice_encoder = joblib.load(ENCODER_PATH)
        print(f"✅ Loaded saved voice scaler and encoder from {BASE_DIR}")
        return _voice_scaler, _voice_encoder

    print("⏳ Building voice scaler and encoder from dataset...")
    files = find_voice_files(DATA_DIR)

    X = []
    y = []

    for file in files:
        features = preprocess_audio(file)
        label = extract_label(file)

        if features is not None and label is not None:
            X.append(features)
            y.append(label)

    if len(X) == 0:
        raise RuntimeError(f"No voice data found to build scaler/encoder in {DATA_DIR}")

    X = np.array(X)
    y = np.array(y)

    encoder = LabelEncoder()
    encoder.fit(y)

    scaler = StandardScaler()
    scaler.fit(X)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    _voice_scaler = scaler
    _voice_encoder = encoder
    print(f"✅ Built and saved voice scaler and encoder to {BASE_DIR}")
    return _voice_scaler, _voice_encoder


# ==============================
# SAME LABEL FUNCTION (IMPORTANT)
# ==============================
def extract_label(file_path):
    filename = os.path.basename(file_path)

    try:
        parts = filename.split("-")
        emotion_code = parts[2]

        if emotion_code in ["01", "03", "08"]:
            return "normal"
        elif emotion_code in ["04", "06"]:
            return "fatigue"
        elif emotion_code in ["05", "07"]:
            return "inflammation"
        else:
            return None

    except:
        return None


# ==============================
# LOAD DATA (FOR SCALER + LABELS)
# ==============================
def prepare_scaler_and_encoder():
    files = find_voice_files(DATA_DIR)

    X = []
    y = []

    for file in files:
        features = preprocess_audio(file)
        label = extract_label(file)

        if features is not None and label is not None:
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Encode labels
    encoder = LabelEncoder()
    encoder.fit(y)

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X)

    return scaler, encoder


# ==============================
# PREDICT FUNCTION
# ==============================
def predict(file_path):
    print(f"\n🎤 Processing file: {file_path}")

    # Load cached model, scaler, and encoder
    model = load_voice_model()
    scaler, encoder = load_or_build_scaler_and_encoder()

    # Preprocess input audio
    features = preprocess_audio(file_path)

    if features is None:
    
        return "error", 0.0
    try:
        # Scale input
        features = scaler.transform([features])

        # Predict
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction)
        label = encoder.inverse_transform([predicted_class])[0]
        confidence = float(np.max(prediction))

        print("\n✅ RESULT:")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}")
        return label, confidence

    except Exception as e:
        print(f"❌ Voice prediction failed: {e}")
        return "error", 0.0


# ==============================
# RUN TEST
# ==============================
if __name__ == "__main__":
    # 🔥 Give your own test audio file here
    test_file = os.path.join(DATA_DIR, "Actor_01/03-01-04-01-01-01-01.wav")

    predict(test_file)