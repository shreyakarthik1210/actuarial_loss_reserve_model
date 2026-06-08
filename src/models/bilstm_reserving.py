import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class _History:
    def __init__(self, loss: float):
        self.history = {"loss": [float(loss)]}


class _LinearFallbackModel:
    def __init__(self, coef: np.ndarray):
        self.coef = coef

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        flat = X.reshape(X.shape[0], -1)
        ones = np.ones((flat.shape[0], 1), dtype=np.float32)
        design = np.hstack([ones, flat])
        return (design @ self.coef).reshape(-1, 1)

# Prevent TensorFlow inter-op threading deadlock on macOS CPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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
        Input(shape=(length, 1)),
        Bidirectional(LSTM(8, return_sequences=False)),
        Dropout(0.1),
        Dense(8, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_bilstm_model(
    X: np.ndarray,
    y: np.ndarray,
    length: int,
    epochs: int = 25,
    batch_size: int = 4,
    verbose: int = 2,
):
    tf.keras.backend.clear_session()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Tiny datasets can stall TF RNN training 
    # Fall back to a fast linear model so the app remains responsive.
    if len(X) < 20:
        flat = X.reshape(X.shape[0], -1)
        ones = np.ones((flat.shape[0], 1), dtype=np.float32)
        design = np.hstack([ones, flat])
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        y_hat = design @ coef
        loss = float(np.mean((y_hat - y) ** 2))
        return _LinearFallbackModel(coef.astype(np.float32)), _History(loss)

    model = build_bilstm_model(length)

    use_validation = len(X) >= 10

    callbacks = []
    if use_validation:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        )

    history = model.fit(
        X,
        y,
        validation_split=0.2 if use_validation else 0.0,
        epochs=epochs,
        batch_size=min(batch_size, len(X)),
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,
    )

    return model, history


def predict_ultimates_bilstm(model: Sequential, t: pd.DataFrame, length: int, scale: float) -> pd.DataFrame:
    accident_years = []
    latest_losses = []
    inputs = []

    for accident_year, row in t.iterrows():
        values = row.dropna().values.astype(float)
        if len(values) < length:
            continue
        accident_years.append(accident_year)
        latest_losses.append(values[-1])
        inputs.append(values[:length] / scale)

    if not inputs:
        return pd.DataFrame(columns=["accident_year", "latest_loss", "bilstm_predicted_ultimate_loss", "bilstm_estimated_reserve"])

    # Single batched predict call — avoids per-row graph tracing overhead on CPU
    X_batch = np.array(inputs).reshape(len(inputs), length, 1)
    scaled_preds = model.predict(X_batch, verbose=0).flatten()

    predictions = []
    for accident_year, latest_loss, scaled_pred in zip(accident_years, latest_losses, scaled_preds):
        predicted_ultimate = scaled_pred * scale
        reserve = max(predicted_ultimate - latest_loss, 0)
        predictions.append({
            "accident_year": accident_year,
            "latest_loss": latest_loss,
            "bilstm_predicted_ultimate_loss": predicted_ultimate,
            "bilstm_estimated_reserve": reserve
        })

    return pd.DataFrame(predictions)