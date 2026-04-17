"""
src/evaluation/metrics.py
==========================
Métricas hidrológicas y de ML para evaluación de modelos de predicción del Orinoco.

Métricas implementadas:
    - MAE (Mean Absolute Error) — en metros, interpretable
    - RMSE (Root Mean Squared Error) — penaliza errores grandes
    - NSE (Nash-Sutcliffe Efficiency) — estándar en hidrología
    - KGE (Kling-Gupta Efficiency) — diagnóstico de componentes de error
    - PTE (Peak Timing Error) — crítico para alertas tempranas
    - VolumeErrorRatio — sesgo en eventos de crecida

Criterios de éxito (de config.yaml):
    NSE > 0.80 (aceptable), NSE > 0.90 (bueno)
    KGE > 0.75
    PTE < 3 días
"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 5 (Evaluación)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Mean Absolute Error en metros.

    Args:
        y_true: Valores observados del nivel del río (metros).
        y_pred: Valores predichos por el modelo (metros).

    Returns:
        MAE en metros.
    """
    raise NotImplementedError("Implementar en Fase 5")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Root Mean Squared Error en metros.

    Args:
        y_true: Valores observados.
        y_pred: Valores predichos.

    Returns:
        RMSE en metros.
    """
    raise NotImplementedError("Implementar en Fase 5")


def compute_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Nash-Sutcliffe Efficiency.

    NSE = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²

    Interpretación:
        NSE = 1.0: Predicción perfecta.
        NSE = 0.0: No mejor que predecir la media histórica.
        NSE < 0.0: PEOR que predecir la media histórica.

    Criterio de éxito: NSE > 0.80 (aceptable), NSE > 0.90 (bueno).

    Args:
        y_true: Valores observados.
        y_pred: Valores predichos.

    Returns:
        NSE. Rango: (-∞, 1].
    """
    raise NotImplementedError("Implementar en Fase 5")


def compute_kge(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Calcula Kling-Gupta Efficiency y sus componentes.

    KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
    Donde:
        r = correlación de Pearson (timing)
        α = ratio de desviaciones estándar (variabilidad)
        β = ratio de medias (sesgo/bias)

    Ventaja sobre NSE: Descompone el error en 3 componentes diagnosticables.

    Criterio de éxito: KGE > 0.75.

    Args:
        y_true: Valores observados.
        y_pred: Valores predichos.

    Returns:
        Tupla (kge, r, alpha, beta).
    """
    raise NotImplementedError("Implementar en Fase 5")


def classify_regime(date: pd.Timestamp) -> str:
    """Clasifica una fecha en su régimen hidrológico del Orinoco.

    Args:
        date: Fecha a clasificar.

    Returns:
        Uno de: 'aguas_bajas', 'ascenso', 'aguas_altas', 'descenso'.
    """
    month = date.month
    if month in [1, 2, 3, 4]:
        return "aguas_bajas"
    elif month in [5, 6, 7]:
        return "ascenso"
    elif month in [8, 9]:
        return "aguas_altas"
    else:
        return "descenso"


def compute_metrics_by_regime(
    y_true: pd.Series,
    y_pred: pd.Series,
    dates: pd.DatetimeIndex,
) -> Dict[str, Dict[str, float]]:
    """Calcula todas las métricas desagregadas por régimen hidrológico.

    Args:
        y_true: Serie de valores observados con índice DatetimeIndex.
        y_pred: Serie de valores predichos.
        dates: Índice temporal de las predicciones.

    Returns:
        Dict anidado: {régimen: {métrica: valor}}
    """
    raise NotImplementedError("Implementar en Fase 5")


def detect_shadow_effect(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Detecta el shadow effect (lag-1 copying) en las predicciones.

    Si corr(ŷ(t), y(t-1)) > corr(ŷ(t), y(t)), el modelo está copiando
    el último valor conocido en lugar de predecir. BANDERA ROJA.

    Args:
        y_true: Valores reales.
        y_pred: Predicciones del modelo.

    Returns:
        Dict con 'corr_pred_real', 'corr_pred_lag1', 'shadow_effect_detected'.
    """
    raise NotImplementedError("Implementar en Fase 5")
