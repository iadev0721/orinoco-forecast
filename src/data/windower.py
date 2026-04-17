"""
src/data/windower.py
====================
Generación de ventanas deslizantes (sliding windows) para modelos de series temporales.

Las ventanas transforman el DataFrame 2D (tiempo × features) en tensores 3D:
    X: (n_samples, lookback, n_features)
    y: (n_samples, forecast_horizon)

REGLA R1 — BORDES CRONOLÓGICOS:
    La última ventana de TRAIN no puede tener como target días de VAL.
    La última ventana de VAL no puede tener como target días de TEST.
    Ver PC-02-02 en docs/02_MANIFESTO_PREPROCESSING.md.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 2 (Preprocessing)


def create_windows(
    df: pd.DataFrame,
    target_col: str,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera ventanas deslizantes X (input) e y (target).

    Args:
        df: DataFrame escalado con índice DatetimeIndex.
        target_col: Nombre de la columna target (ej: 'palua').
        lookback: Número de días de historia como input.
        horizon: Número de días a predecir (horizonte de predicción).

    Returns:
        Tupla (X, y) donde:
            X.shape = (n_samples, lookback, n_features)
            y.shape = (n_samples, horizon)
    """
    raise NotImplementedError("Implementar en Fase 2")


def verify_no_leakage_at_boundaries(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    lookback: int,
    horizon: int,
) -> bool:
    """Verifica que las ventanas en el borde train/val son correctas.

    La última ventana de train debe tener su último TARGET en train_end.
    El solapamiento en el INPUT (lookback) es PERMITIDO y ESPERADO.

    Args:
        df_train: DataFrame del split train.
        df_val: DataFrame del split validation.
        lookback: Ventana de historia.
        horizon: Horizonte de predicción.

    Returns:
        True si los bordes son correctos.

    Raises:
        AssertionError: Si hay leakage en los bordes.
    """
    raise NotImplementedError("Implementar en Fase 2")
