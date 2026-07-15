"""LSTM helpers for hourly electricity load forecasting (Colab-safe)."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.models import Sequential
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def make_sequences(
    values: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) sliding windows for one-step-ahead prediction."""
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len : i])
        y.append(values[i])
    return np.array(X), np.array(y)


def build_lstm_model(
    seq_len: int,
    units: int = 64,
    n_layers: int = 1,
    learning_rate: float = 0.001,
) -> Any:
    """Small LSTM to reduce Colab memory pressure."""
    if not TF_AVAILABLE:
        raise ImportError("tensorflow is required for LSTM modelling")

    tf.random.set_seed(42)
    model = Sequential()
    if n_layers == 1:
        model.add(LSTM(units, input_shape=(seq_len, 1)))
    else:
        model.add(LSTM(units, return_sequences=True, input_shape=(seq_len, 1)))
        model.add(LSTM(units // 2))

    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_lstm(
    train_values: np.ndarray,
    seq_len: int,
    units: int = 64,
    n_layers: int = 1,
    epochs: int = 15,
    batch_size: int = 64,
    val_split: float = 0.1,
) -> tuple[Any, MinMaxScaler]:
    """Fit scaler on train only, then train LSTM."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()

    X, y = make_sequences(scaled, seq_len)
    X = X.reshape(-1, seq_len, 1)

    model = build_lstm_model(seq_len, units=units, n_layers=n_layers)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=0,
    )
    return model, scaler


def recursive_lstm_forecast(
    model: Any,
    scaler: MinMaxScaler,
    history: np.ndarray,
    n_steps: int,
    seq_len: int,
    save_every: int = 2000,
    checkpoint: np.ndarray | None = None,
) -> np.ndarray:
    """
    One-step-ahead recursive forecast for `n_steps` hours.
    Optionally resume from a partial checkpoint array (original scale).
    """
    scaled_hist = scaler.transform(history.reshape(-1, 1)).flatten().tolist()
    preds_original: list[float] = []

    if checkpoint is not None and len(checkpoint) > 0:
        preds_original = checkpoint.tolist()
        scaled_ckpt = scaler.transform(np.array(checkpoint).reshape(-1, 1)).flatten()
        scaled_hist.extend(scaled_ckpt.tolist())

    start = len(preds_original)
    for step in tqdm(range(start, n_steps), desc="LSTM recursive forecast"):
        window = np.array(scaled_hist[-seq_len:]).reshape(1, seq_len, 1)
        next_scaled = float(model.predict(window, verbose=0)[0, 0])
        scaled_hist.append(next_scaled)
        preds_original.append(
            float(scaler.inverse_transform([[next_scaled]])[0, 0])
        )

        if save_every and (step + 1) % save_every == 0:
            gc.collect()

    return np.array(preds_original)


def hourly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24 * 7,
) -> dict[str, float]:
    """Evaluation metrics for hourly forecasts."""
    from utils import mase, rmse
    from sklearn.metrics import mean_absolute_error

    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)
    y_train_s = pd.Series(y_train)

    return {
        "MAE": float(mean_absolute_error(y_true_s, y_pred_s)),
        "RMSE": rmse(y_true_s, y_pred_s),
        "MASE": mase(y_true_s, y_pred_s, y_train_s, seasonality=seasonality),
        "Bias": float(np.mean(y_pred - y_true)),
    }
