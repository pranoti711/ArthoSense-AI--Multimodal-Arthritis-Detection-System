# model.py
# Defines the arthritis detection model
# Works with tabular/temporal preprocessed input

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore

def build_model(input_dim):
    """
    Create a simple feed-forward neural network for arthritis prediction.

    input_dim : number of features after preprocessing
    """

    model = Sequential()

    # Input layer
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))

    # Hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    # Output layer (binary classification)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    # Example: If preprocessing produced 20 features
    sample_dim = 20
    model = build_model(sample_dim)
    model.summary()


