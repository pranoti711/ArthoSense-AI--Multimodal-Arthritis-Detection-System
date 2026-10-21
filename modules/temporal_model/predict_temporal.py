import numpy as np
import joblib
from pathlib import Path
import sys
from tensorflow.keras.models import load_model # type: ignore

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# -----------------------------------------------------
# CORRECT PATHS (relative to script location)
# -----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "temporal_model.h5"
SCALER_PATH = SCRIPT_DIR / "risk_scaler.pkl"

# -----------------------------------------------------
# LOAD MODEL + SCALER
# -----------------------------------------------------
print("⏳ Loading model and scaler...")

try:
    # Use compile_from_config=False to avoid Keras deserialization issues
    model = load_model(str(MODEL_PATH), compile=False)
    model.compile(optimizer="adam", loss="mse")
except Exception as e:
    print("❌ ERROR: temporal_model.h5 not found or corrupt.")
    print("Path tried:", MODEL_PATH)
    raise e

scaler = joblib.load(str(SCALER_PATH))

print("✅ Model and scaler loaded successfully!")


# -----------------------------------------------------
# PREDICTION FUNCTION
# -----------------------------------------------------
def predict_temporal(csv_path):
    """
    Predict arthritis risk score from CSV file.
    CSV should have columns: code, weight, height, age, gender
    Returns numeric score.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    code = row['code']
    weight = row['weight']
    height = row['height']
    age = row['age']
    gender = row['gender']

    # Prepare input
    x = np.array([[code, weight, height, age, gender]])

    # Scale input
    x_scaled = scaler.transform(x)

    # Predict
    score = float(model.predict(x_scaled)[0][0])
    
    # Apply sigmoid to normalize to 0-1
    score = 1 / (1 + np.exp(-score))

    return score


# -----------------------------------------------------
# MANUAL TEST RUN
# -----------------------------------------------------
if __name__ == "__main__":
    csv_path = input("Enter path to temporal CSV file: ").strip()
    score = predict_temporal(csv_path)
    print("\n==============================")
    print("Predicted Score:", round(score, 2))
    print("==============================")
