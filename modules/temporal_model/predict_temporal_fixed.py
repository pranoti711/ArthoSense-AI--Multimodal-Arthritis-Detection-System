import numpy as np
import joblib
from pathlib import Path
import sys
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore

sys.path.append(str(Path(__file__).resolve().parents[2]))

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "temporal_model.h5"
SCALER_PATH = SCRIPT_DIR / "risk_scaler.pkl"

print("⏳ Loading model and scaler...")

try:
    model = load_model(str(MODEL_PATH), compile=False)
    model.compile(optimizer="adam", loss="mse")
except Exception as e:
    print("❌ ERROR: temporal_model.h5 not found or corrupt.")
    print("Path tried:", MODEL_PATH)
    raise e

scaler = joblib.load(str(SCALER_PATH))

print("✅ Model and scaler loaded successfully!")

def predict_temporal(csv_path):
    """
    Predict arthritis risk score from CSV file.
    CSV columns: code,weight,height,age,gender
    Returns normalized 0-1 score.
    """
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    x = np.array([[row['code'], row['weight'], row['height'], row['age'], row['gender']]])

    x_scaled = scaler.transform(x)

    raw_score = float(model.predict(x_scaled)[0][0])
    
    score = 1 / (1 + np.exp(-raw_score / 1000))  
    
    print(f"Raw logit: {raw_score:.2f} -> Normalized score: {score:.3f}")
    return score

if __name__ == "__main__":
    csv_path = "temporal.csv"
    score = predict_temporal(csv_path)
    print("\n===============================")
    print("Temporal Risk Score:", round(score, 3))
    print("===============================")

