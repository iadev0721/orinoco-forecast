"""
src/models/naive_baseline.py
==============================
Modelos Naive de referencia. DEBEN ejecutarse antes que cualquier red neuronal.

REGLA R3: BASELINE PRIMERO. results/metrics/baseline_metrics.json
debe existir antes de entrenar LSTM o Transformer.

Modelos implementados:
    - NaiveBaseline: ŷ(t+h) = y(t) para h = 1,...,7
    - SeasonalNaive: ŷ(t+h) = y(t+h-365)
"""
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 3a (Baseline)


class NaiveBaseline:
    """Modelo Naive: predice el último valor conocido para todos los horizontes.

    ŷ(t+h) = y(t)  para h = 1,..., forecast_horizon

    Justificación (Hyndman, 2018): El modelo más simple posible.
    Todo modelo de ML debe demostrar que lo supera.
    """

    def __init__(self, horizon: int = 7) -> None:
        """
        Args:
            horizon: Número de días a predecir.
        """
        self.horizon = horizon

    def predict(self, y_last: np.ndarray) -> np.ndarray:
        """Genera predicciones naive.

        Args:
            y_last: Array de últimos valores conocidos. Shape: (n_samples,).

        Returns:
            Predicciones. Shape: (n_samples, horizon).
        """
        raise NotImplementedError("Implementar en Fase 3a")


class SeasonalNaive:
    """Modelo Seasonal Naive: predice el valor del mismo día del año anterior.

    ŷ(t+h) = y(t+h-365)

    Captura la estacionalidad dominante del Orinoco.
    """

    def __init__(self, horizon: int = 7) -> None:
        """
        Args:
            horizon: Número de días a predecir.
        """
        self.horizon = horizon

    def predict(self, df: pd.DataFrame, target_col: str, forecast_date: pd.Timestamp) -> np.ndarray:
        """Genera predicciones seasonal naive.

        Args:
            df: DataFrame histórico completo con índice DatetimeIndex.
            target_col: Nombre de la columna target.
            forecast_date: Fecha desde la cual se predice.

        Returns:
            Predicciones. Shape: (horizon,).
        """
        raise NotImplementedError("Implementar en Fase 3a")
