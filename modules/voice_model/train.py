import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

from preprocess_voice import preprocess_audio, find_voice_files

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)

# 👉 IMPORTANT: Point directly to RAVDESS folder
DATA_DIR = os.path.join(BASE_DIR, "../../data/raw/voice/ravdess_kaggle")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../../models/voice_model.h5")


# ==============================
# ✅ LABEL EXTRACTION (FIXED)
# ==============================
def extract_label(file_path):
    """
    Extract emotion from RAVDESS filename
    and map to arthritis-related classes
    """
    filename = os.path.basename(file_path)

    try:
        parts = filename.split("-")
        emotion_code = parts[2]

        # 🎯 Mapping for YOUR PROJECT
        if emotion_code in ["01", "03", "08"]:
            return "normal"
        elif emotion_code in ["04", "06"]:
            return "fatigue"
        elif emotion_code in ["05", "07"]:
            return "inflammation"
        else:
            return None

    except Exception as e:
        print(f"Label extraction error for {filename}: {e}")
        return None


# ==============================
# LOAD DATASET
# ==============================
def load_dataset(data_dir):
    files = find_voice_files(data_dir)

    X = []
    y = []

    print(f"Found {len(files)} audio files\n")

    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing {file}...", end=" ")

        features = preprocess_audio(file)

        if features is not None:
            label = extract_label(file)

            if label is not None:
                X.append(features)
                y.append(label)
                print(f"✓ ({label})")
            else:
                print("✗ Invalid label")
        else:
            print("✗ Failed")

    return np.array(X), np.array(y)


# ==============================
# BUILD MODEL
# ==============================
def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==============================
# TRAIN PIPELINE
# ==============================
def train():
    # 🔹 Load data
    X, y = load_dataset(DATA_DIR)

    if len(X) == 0:
        print("❌ No data found. Check dataset path.")
        return

    print("\nDataset shape:", X.shape)

    # 🔹 Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Classes:", list(encoder.classes_))

    # 🔹 Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔹 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    # 🔹 Build model
    model = build_model(input_dim=X.shape[1], num_classes=len(set(y_encoded)))

    model.summary()

    # 🔹 Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32
    )

    # 🔹 Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n🎯 Test Accuracy: {acc:.4f}")

    # 🔹 Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    print(f"\n✅ Model saved at: {MODEL_SAVE_PATH}")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    train()