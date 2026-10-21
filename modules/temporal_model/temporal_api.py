from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import os

app = Flask(__name__)

# -------- Load Model and Scaler --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "temporal_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "risk_scaler.pkl")

print("⏳ Loading model and scaler...")

# Load the model safely (skip compilation to avoid 'mse' error)
model = load_model(MODEL_PATH, compile=False)

# If you need the model compiled for evaluation/training, uncomment:
# model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

# Load the scaler
scaler = joblib.load(SCALER_PATH)

print("✅ Model and scaler loaded successfully!")


# -------- Prediction Route --------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)

        # Scale input
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)
        result = float(prediction[0][0])  # assuming single output

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)

