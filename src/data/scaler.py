"""
src/data/scaler.py
==================
Escalado de datos sin data leakage.

REGLA R1 — MANDAMIENTO FUNDAMENTAL:
    El scaler se AJUSTA (fit) EXCLUSIVAMENTE con datos de entrenamiento.
    Se aplica (transform) a train, val y test por SEPARADO.
    Se GUARDA con joblib para reutilización en inferencia.

Flujo obligatorio:
    scaler = fit_scaler(df_train)
    df_train_scaled = transform(scaler, df_train)
    df_val_scaled   = transform(scaler, df_val)     # NO fit aquí
    df_test_scaled  = transform(scaler, df_test)    # NO fit aquí
    save_scaler(scaler, path)
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 2 (Preprocessing)


def fit_scaler(df_train: pd.DataFrame) -> MinMaxScaler:
    """Ajusta el MinMaxScaler SOLO con datos de entrenamiento.

    Args:
        df_train: DataFrame del split de train (1974-2015).

    Returns:
        MinMaxScaler ajustado. Rango: [0, 1].
    """
    raise NotImplementedError("Implementar en Fase 2")


def transform(scaler: MinMaxScaler, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el scaler ajustado a un DataFrame.

    NOTA: Este método NUNCA ajusta el scaler. Solo transforma.

    Args:
        scaler: MinMaxScaler previamente ajustado con fit_scaler().
        df: DataFrame a transformar (train, val o test).

    Returns:
        DataFrame escalado con los mismos índices y columnas.
    """
    raise NotImplementedError("Implementar en Fase 2")


def inverse_transform(scaler: MinMaxScaler, arr: np.ndarray) -> np.ndarray:
    """Revierte el escalado para obtener predicciones en metros.

    Args:
        scaler: MinMaxScaler previamente ajustado.
        arr: Array escalado (predicciones del modelo).

    Returns:
        Array en metros (escala original).
    """
    raise NotImplementedError("Implementar en Fase 2")


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    """Guarda el scaler en disco con joblib.

    Args:
        scaler: MinMaxScaler ajustado.
        path: Ruta de destino (ej: 'results/models/scaler.joblib').
    """
    raise NotImplementedError("Implementar en Fase 2")


def load_scaler(path: str) -> MinMaxScaler:
    """Carga el scaler desde disco.

    Args:
        path: Ruta al archivo .joblib.

    Returns:
        MinMaxScaler listo para transformar.
    """
    raise NotImplementedError("Implementar en Fase 2")
