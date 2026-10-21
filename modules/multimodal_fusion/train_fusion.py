import os
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.dirname(__file__))
from fusion_layer import build_fusion_model

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../../models/fusion_model.h5")


# ==============================
# GENERATE TRAINING DATA
# ==============================
def generate_dummy_data(num_samples=200):
    """
    Simulated inputs from:
    [genomics_score, voice_score, imaging_score]
    """

    # Random scores between 0–1
    X = np.random.rand(num_samples, 3)

    y = []

    for sample in X:
        avg = np.mean(sample)

        # 🔹 Rule-based labeling
        if avg < 0.4:
            y.append(0)  # Low
        elif avg < 0.7:
            y.append(1)  # Medium
        else:
            y.append(2)  # High

    return np.array(X), np.array(y)


# ==============================
# TRAIN FUNCTION
# ==============================
def train():
    # 🔹 Generate data
    X, y = generate_dummy_data()

    print("Dataset shape:", X.shape)

    # 🔹 Build model
    model = build_fusion_model(input_dim=3, num_classes=3)

    model.summary()

    # 🔹 Train model
    model.fit(
        X,
        y,
        epochs=50,
        batch_size=16,
        validation_split=0.2
    )

    # 🔹 Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    print(f"\n✅ Fusion model saved at: {MODEL_SAVE_PATH}")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    train()