import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

"""
This function converts the loss triangle into training data format for the BiLSTM model.
"""
def training_data (t: pd.DataFrame, length: int) -> tuple[np.ndarray, np.ndarray, float]:
    X = []
    y = []

    for _, row in t.iterrows():
        values = row.dropna().values.astype(float)
        if len(values) < length:   # Not enough data points to create a training sample, skip this row
            continue
        input_seq = values[:length]
        ultimate_loss = values[-1]
        X.append(input_seq)
        y.append(ultimate_loss)

        if not X:
            raise ValueError (
                "Not enough data to create training samples. " \
                "Please ensure that the loss triangle has sufficient data points for the specified sequence length."
            )

        X = np.array(X)
        y = np.array(y)

        # Scale the input features and target variable to improve model training stability
        scale = np.max(y)
        X_scaled = X/scale
        y_scaled = y/scale

        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # Reshape for LSTM input

    return X_scaled, y_scaled, scale


def build_bilstm_model(length: int) -> Sequential:
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=(length, 1)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def train_bilstm_model(X: np.ndarray, y: np.ndarray, length: int, epochs: int = 200, batch_size: int = 8):
    model = build_bilstm_model(length)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history


