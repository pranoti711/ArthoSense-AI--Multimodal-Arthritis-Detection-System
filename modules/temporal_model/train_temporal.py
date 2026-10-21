
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import joblib
import os
DATA_PATH = r"c:/Users/Security/Desktop/arthritis-detection-care-assistant 1/data/processed/temporal_processed.csv"

df = pd.read_csv(DATA_PATH)

# Use only the 5 inputs
X = df[["code", "weight", "height", "age", "gender"]].values


y = np.random.rand(len(df)) * 100   # 0–100 risk score


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, r"c:/Users/Security/Desktop/arthritis-detection-care-assistant 1/modules/temporal_model/risk_scaler.pkl")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(5,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))  # output = risk score

model.compile(optimizer="adam", loss="mse")

# -----------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

# -----------------------------------------------------
# SAVE MODEL
# -----------------------------------------------------
MODEL_PATH = r"c:/Users/Security/Desktop/arthritis-detection-care-assistant 1/modules/temporal_model/temporal_model.h5"
model.save(MODEL_PATH)

print("🎉 Training complete!")
print("Model saved at:", MODEL_PATH)
