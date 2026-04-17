"""
src/features/temporal_features.py
===================================
Features temporales cíclicas para el modelado de series temporales.

Las fechas tienen naturaleza cíclica (el día 365 está cerca del día 1).
La codificación sin/cos preserva esta continuidad para el modelo.

REGLA R1 APLICADA: La climatología base se calcula SOLO sobre el train set.
REGLA R6: No hardcodear valores; usar config.yaml para hiperparámetros.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 1 (Feature Engineering)


def add_cyclical_day_of_year(df: pd.DataFrame) -> pd.DataFrame:
    """Añade codificación cíclica del día del año al DataFrame.

    Codificación:
        day_sin = sin(2π × dayofyear / 365.25)
        day_cos = cos(2π × dayofyear / 365.25)

    NOTA: NO incluir el año como feature (leakage temporal de tendencia).

    Args:
        df: DataFrame con índice DatetimeIndex.

    Returns:
        DataFrame con columnas 'day_sin' y 'day_cos' añadidas.
    """
    raise NotImplementedError("Implementar en Fase 1")


def add_rate_of_change(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Añade features de tasa de cambio (velocidad del río).

    Features creadas:
        {target_col}_delta_1d: cambio en 1 día (¿subiendo o bajando?)
        {target_col}_delta_7d: cambio en 7 días (tendencia semanal)

    Args:
        df: DataFrame con índice DatetimeIndex.
        target_col: Nombre de la columna target (ej: 'palua').

    Returns:
        DataFrame con columnas de delta añadidas.
    """
    raise NotImplementedError("Implementar en Fase 1")


def add_rolling_statistics(
    df: pd.DataFrame,
    target_col: str,
    windows: list = [7, 14, 30],
) -> pd.DataFrame:
    """Añade estadísticas rodantes (rolling features).

    Features creadas por ventana:
        {target_col}_rolling_mean_{w}
        {target_col}_rolling_std_{w}

    Args:
        df: DataFrame con índice DatetimeIndex.
        target_col: Nombre de la columna target.
        windows: Lista de tamaños de ventana en días.

    Returns:
        DataFrame con columnas de rolling statistics añadidas.
    """
    raise NotImplementedError("Implementar en Fase 1")


def add_climatological_anomaly(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Añade la anomalía respecto al ciclo medio histórico del train set.

    La climatología se calcula SOLO sobre df_train (anti-leakage).
    Se aplica a df completo.

    Args:
        df: DataFrame completo a transformar.
        df_train: DataFrame del split de train para calcular climatología.
        target_col: Nombre de la columna target.

    Returns:
        DataFrame con columna '{target_col}_anomaly' añadida.
    """
    raise NotImplementedError("Implementar en Fase 1")
