"""
src/models/lstm_model.py
=========================
LSTM Multivariado Multi-Step en Keras/TensorFlow.

MODELO CORE de la tesis. Implementa la arquitectura definitiva:
    Input: (batch, lookback, n_features)
    LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2)
    Dense(32, relu) → Dense(7, linear)
    Output: (batch, 7)  ← 7 días de predicción

REGLAS OBLIGATORIAS:
    R2: Seeds fijados antes de toda ejecución.
    R3: baseline_metrics.json debe existir antes de llamar fit().
    R4: Predicciones pasadas por clamp físico (no negativas).
    R5: GPU memory growth activado.
    R7: Hiperparámetros desde config.yaml.
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 3b (LSTM)
# Prerequisito: results/metrics/baseline_metrics.json debe existir


def check_baseline_gate(metrics_path: str = "results/metrics/baseline_metrics.json") -> None:
    """Verifica que el baseline fue ejecutado antes del LSTM.

    REGLA R3: NINGÚN modelo de deep learning se entrena sin baseline previo.

    Args:
        metrics_path: Ruta al archivo de métricas del baseline.

    Raises:
        FileNotFoundError: Si baseline_metrics.json no existe.
    """
    import json
    from pathlib import Path

    if not Path(metrics_path).exists():
        raise FileNotFoundError(
            f"REGLA R3 VIOLADA: '{metrics_path}' no existe. "
            "Ejecutar primero notebooks/03_baseline.ipynb y generar las métricas del baseline."
        )


def build_lstm_model(config: dict, n_features: int):
    """Construye el modelo LSTM según la arquitectura del informe.

    Arquitectura:
        LSTM(64, return_sequences=True) → Dropout(0.2)
        LSTM(32, return_sequences=False) → Dropout(0.2)
        Dense(32, relu) → Dense(horizon, linear)

    Args:
        config: Configuración del experimento (lee lstm.units, dropout, etc.).
        n_features: Número de features en el tensor de entrada.

    Returns:
        Modelo Keras compilado.
    """
    raise NotImplementedError("Implementar en Fase 3b")


def train_lstm(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> dict:
    """Entrena el modelo LSTM con EarlyStopping.

    Args:
        model: Modelo Keras compilado.
        X_train: Tensores de entrenamiento. Shape: (n, lookback, n_features).
        y_train: Targets de entrenamiento. Shape: (n, horizon).
        X_val: Tensores de validación.
        y_val: Targets de validación.
        config: Configuración del experimento.

    Returns:
        History dict con train_loss y val_loss por época.
    """
    raise NotImplementedError("Implementar en Fase 3b")


def apply_physical_constraints(predictions: np.ndarray, config: dict) -> np.ndarray:
    """Aplica restricciones físicas del río a las predicciones.

    REGLA R4:
        - Nivel nunca negativo: max(0, predicción)
        - No exceder máximo histórico + 15%

    Args:
        predictions: Array de predicciones en metros (escala original).
        config: Configuración del experimento.

    Returns:
        Predicciones con restricciones físicas aplicadas.
    """
    raise NotImplementedError("Implementar en Fase 3b")
