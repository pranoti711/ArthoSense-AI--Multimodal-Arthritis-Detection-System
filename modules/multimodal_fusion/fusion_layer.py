import tensorflow as tf

# ==============================
# BUILD FUSION MODEL
# ==============================
def build_fusion_model(input_dim=3, num_classes=3):
    """
    input_dim = 3 → [genomics, voice, imaging]
    num_classes = 3 → [Low, Medium, High]
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        # 🔹 First layer
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # 🔹 Second layer
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # 🔹 Third layer
        tf.keras.layers.Dense(8, activation='relu'),

        # 🔹 Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # ==============================
    # COMPILE MODEL
    # ==============================
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==============================
# TEST MODEL
# ==============================
if __name__ == "__main__":
    model = build_fusion_model()
    model.summary()