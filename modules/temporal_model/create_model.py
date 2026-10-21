from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

# -----------------------------
# Create dummy data
# -----------------------------
X_train_dummy = np.random.rand(10, 5)  # 10 samples, 5 features
y_train_dummy = np.random.randint(0, 2, 10)

# -----------------------------
# Create and save scaler
# -----------------------------
scaler = StandardScaler()
scaler.fit(X_train_dummy)
SCALER_PATH = r'C:\Users\Pranoti munjankar\Desktop\arthritis-detection-care-assistant-local\modules\temporal_model\risk_scaler.pkl'
joblib.dump(scaler, SCALER_PATH)
print("✅ Scaler saved.")

# -----------------------------
# Create and save dummy Keras model
# -----------------------------
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train briefly on dummy data
model.fit(X_train_dummy, y_train_dummy, epochs=5, verbose=0)

MODEL_PATH = r"C:\Users\Security\Desktop\arthritis-detection-care-assistant 1\modules\temporal_model\temporal_tf_model.h5"
model.save(MODEL_PATH)
print("✅ Dummy model saved successfully.")
