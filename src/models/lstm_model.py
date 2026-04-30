"""
src/models/lstm_model.py
=========================
LSTM Multivariado Multi-Step en Keras/TensorFlow.

MODELO CORE de la tesis. Arquitectura definitiva:
    Input:   (batch, lookback, n_features)
    LSTM(64, return_sequences=True) -> Dropout(0.2)
    LSTM(32, return_sequences=False) -> Dropout(0.2)
    Dense(32, relu) -> Dense(horizon, linear)
    Output:  (batch, horizon)  <- prediccion de 'horizon' dias

REGLAS OBLIGATORIAS:
    R2: Seeds fijados antes de toda ejecucion.
    R3: baseline_metrics.json debe existir antes de llamar a train_lstm().
    R4: Predicciones pasadas por apply_physical_constraints() en inferencia.
    R5: configure_tensorflow_gpu() llamado antes de construir el modelo.
    R7: Hiperparametros desde config.yaml.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def check_baseline_gate(
    metrics_path: str = "results/metrics/baseline_metrics.json",
) -> None:
    """Verifica que el baseline fue ejecutado antes del LSTM.

    REGLA R3: NINGUN modelo de deep learning se entrena sin baseline previo.

    Args:
        metrics_path: Ruta al archivo de metricas del baseline.

    Raises:
        FileNotFoundError: Si baseline_metrics.json no existe.
    """
    if not Path(metrics_path).exists():
        raise FileNotFoundError(
            f"REGLA R3 VIOLADA: '{metrics_path}' no existe.\n"
            "Ejecutar primero:\n"
            "  python scripts/run_experiment.py --name baseline_naive --model naive"
        )
    with open(metrics_path) as f:
        data = json.load(f)
    best_nse = max(
        data.get("naive_baseline", {}).get("nse", -999),
        data.get("seasonal_naive", {}).get("nse", -999),
    )
    logger.info("Gate R3 OK. Mejor baseline NSE=%.4f", best_nse)


def build_lstm_model(config: dict, n_features: int):
    """Construye el modelo LSTM segun la arquitectura del informe.

    Arquitectura:
        LSTM(units[0], return_sequences=True) -> Dropout(dropout)
        LSTM(units[1], return_sequences=False) -> Dropout(dropout)
        Dense(32, relu) -> Dense(horizon, linear)

    Args:
        config: Configuracion del experimento (lee lstm.units, dropout, horizon, lookback).
        n_features: Numero de features en el tensor de entrada.

    Returns:
        Modelo Keras compilado y listo para entrenamiento.
    """
    import tensorflow as tf
    from tensorflow import keras

    lstm_cfg = config["lstm"]
    units       = lstm_cfg.get("units", [64, 32])
    dropout     = lstm_cfg.get("dropout", 0.2)
    lr          = lstm_cfg.get("learning_rate", 0.001)
    loss_name   = lstm_cfg.get("loss", "mse")          # 'mse' | 'mae' | 'huber'
    huber_delta = lstm_cfg.get("huber_delta", 0.5)     # solo aplica si loss='huber'
    horizon  = config["forecast_horizon"]
    lookback = config["lookback_window"]

    inputs = keras.Input(shape=(lookback, n_features), name="input_seq")
    x = keras.layers.LSTM(units[0], return_sequences=True,  name="lstm_1")(inputs)
    x = keras.layers.Dropout(dropout,                        name="dropout_1")(x)
    x = keras.layers.LSTM(units[1], return_sequences=False, name="lstm_2")(x)
    x = keras.layers.Dropout(dropout,                        name="dropout_2")(x)
    x = keras.layers.Dense(32, activation="relu",            name="dense_hidden")(x)
    outputs = keras.layers.Dense(horizon, activation="linear", name="output")(x)

    model = keras.Model(inputs, outputs, name="orinoco_lstm")

    # Seleccion de funcion de perdida
    if loss_name == "huber":
        loss_fn = keras.losses.Huber(delta=huber_delta)
        logger.info("Loss: Huber(delta=%.2f) -- atenua outliers >%.2f en espacio escalado",
                    huber_delta, huber_delta)
    elif loss_name == "mae":
        loss_fn = "mae"
        logger.info("Loss: MAE")
    else:
        loss_fn = "mse"
        logger.info("Loss: MSE (default)")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["mae"],
    )
    logger.info(
        "Modelo LSTM construido: lookback=%d, n_features=%d, units=%s, dropout=%.2f, horizon=%d",
        lookback, n_features, units, dropout, horizon,
    )
    model.summary(print_fn=logger.info)
    return model


def train_lstm(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    experiment_name: str = "lstm",
) -> dict:
    """Entrena el modelo LSTM con EarlyStopping y ReduceLROnPlateau.

    Args:
        model: Modelo Keras compilado (output de build_lstm_model).
        X_train: Tensores de entrenamiento. Shape: (n, lookback, n_features).
        y_train: Targets de entrenamiento. Shape: (n, horizon).
        X_val: Tensores de validacion.
        y_val: Targets de validacion.
        config: Configuracion del experimento.
        experiment_name: Nombre para guardar el checkpoint.

    Returns:
        history.history: Dict con 'loss', 'val_loss', 'mae', 'val_mae' por epoca.
    """
    from tensorflow import keras

    lstm_cfg = config["lstm"]
    patience  = lstm_cfg.get("patience", 15)
    max_epochs = lstm_cfg.get("max_epochs", 200)
    batch_size = lstm_cfg.get("batch_size", 64)
    model_path = f"results/models/{experiment_name}_best.keras"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(5, patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(
        "Iniciando entrenamiento: epochs=%d, batch=%d, patience=%d",
        max_epochs, batch_size, patience,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    best_ep  = int(np.argmin(history.history["val_loss"])) + 1
    best_val = float(min(history.history["val_loss"]))
    logger.info(
        "Entrenamiento completo: %d epocas | mejor val_loss=%.6f en epoca %d",
        len(history.history["loss"]), best_val, best_ep,
    )
    return history.history


def apply_physical_constraints(
    predictions: np.ndarray,
    config: dict,
    y_train_max: Optional[float] = None,
) -> np.ndarray:
    """Aplica restricciones fisicas del rio a las predicciones.

    REGLA R4:
        - Nivel nunca negativo: max(0, prediccion)
        - No exceder maximo historico + 15%
        - Alerta si cambio diario > 1.5 m (fisicamente improbable)

    Args:
        predictions: Array de predicciones en metros (escala original).
                     Shape: (n,) o (n, horizon).
        config: Configuracion del experimento.
        y_train_max: Maximo historico del train set (para calcular techo).
                     Si None, no se aplica el techo.

    Returns:
        Predicciones con restricciones fisicas aplicadas.
    """
    physical = config.get("physical", {})
    min_level = physical.get("min_level_m", 0.0)
    max_mult  = physical.get("max_level_multiplier", 1.15)
    max_daily = physical.get("max_daily_change_m", 1.5)

    preds = np.copy(predictions)

    # R4.1: Nivel nunca negativo
    preds = np.maximum(preds, min_level)

    # R4.2: No exceder maximo historico + 15%
    if y_train_max is not None:
        ceiling = y_train_max * max_mult
        n_clipped = int(np.sum(preds > ceiling))
        preds = np.minimum(preds, ceiling)
        if n_clipped > 0:
            logger.warning("R4: %d predicciones clippeadas al techo fisico (%.2f m)", n_clipped, ceiling)

    # R4.3: Alerta si cambio diario implausible (solo para series 1D o ultimo horizonte)
    flat = preds.ravel() if preds.ndim == 1 else preds[:, -1]
    daily_changes = np.abs(np.diff(flat))
    n_suspicious = int(np.sum(daily_changes > max_daily))
    if n_suspicious > 0:
        logger.warning(
            "R4: %d cambios diarios > %.1f m detectados (fisicamente sospechosos)",
            n_suspicious, max_daily,
        )
    return preds
