"""
src/features/lag_calculator.py
================================
Cálculo empírico de lag times hidrológicos entre estaciones.

El lag time entre estaciones NO debe adivinarse: debe calcularse
mediante la función de correlación cruzada (cross-correlation) con los
datos reales. Los valores de docs/STATION_TOPOLOGY.md son estimados iniciales.

Los lag times calculados aquí determinan el lookback window mínimo del modelo.
Ver: PC-01-03 en docs/01_MANIFESTO_EDA.md.
"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 1 (EDA)


def compute_cross_correlation(
    upstream: pd.Series,
    downstream: pd.Series,
    max_lag: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula la correlación cruzada entre dos estaciones.

    Args:
        upstream: Serie temporal de la estación aguas arriba.
        downstream: Serie temporal de la estación aguas abajo.
        max_lag: Lag máximo a evaluar en días.

    Returns:
        Tupla (lags, correlaciones) donde lags son los desfases evaluados.
    """
    raise NotImplementedError("Implementar en Fase 1")


def find_optimal_lag(
    upstream: pd.Series,
    downstream: pd.Series,
    max_lag: int = 60,
) -> int:
    """Encuentra el lag óptimo que maximiza la correlación cruzada.

    Args:
        upstream: Serie temporal de la estación aguas arriba.
        downstream: Serie temporal de la estación aguas abajo.
        max_lag: Lag máximo a evaluar en días.

    Returns:
        Lag óptimo en días.
    """
    raise NotImplementedError("Implementar en Fase 1")


def compute_all_lag_times(df: pd.DataFrame, target_station: str) -> Dict[str, int]:
    """Calcula todos los lag times del sistema Orinoco.

    Pares evaluados:
        - Pares adyacentes (ej. ayacucho → caicara)
        - Estaciones origen → target (ej. origen → target_station)

    Args:
        df: DataFrame con columnas por estación.
        target_station: Estación definida como target en config.

    Returns:
        Dict con lag times en días por par de estaciones.
        Formato: {"source_to_target_total_days": 21, ...}
    """
    raise NotImplementedError("Implementar en Fase 1")
