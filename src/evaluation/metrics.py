"""
src/evaluation/metrics.py
==========================
Metricas hidrologicas y de ML para evaluacion de modelos de prediccion del Orinoco.

Metricas implementadas:
    - MAE (Mean Absolute Error) -- en metros, interpretable
    - RMSE (Root Mean Squared Error) -- penaliza errores grandes
    - NSE (Nash-Sutcliffe Efficiency) -- estandar en hidrologia
    - KGE (Kling-Gupta Efficiency) -- diagnostico de componentes de error
    - detect_shadow_effect -- bandera roja si el modelo copia el ultimo valor

Criterios de exito (de config.yaml):
    NSE > 0.80 (aceptable), NSE > 0.90 (bueno)
    KGE > 0.75
"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Mean Absolute Error en metros.

    Args:
        y_true: Valores observados del nivel del rio (metros).
        y_pred: Valores predichos por el modelo (metros).

    Returns:
        MAE en metros.
    """
    return float(np.mean(np.abs(y_true.ravel() - y_pred.ravel())))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Root Mean Squared Error en metros.

    Args:
        y_true: Valores observados.
        y_pred: Valores predichos.

    Returns:
        RMSE en metros.
    """
    return float(np.sqrt(np.mean((y_true.ravel() - y_pred.ravel()) ** 2)))


def compute_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Nash-Sutcliffe Efficiency.

    NSE = 1 - sum(y - y_hat)^2 / sum(y - y_mean)^2

    Interpretacion:
        NSE = 1.0: Prediccion perfecta.
        NSE = 0.0: No mejor que predecir la media historica.
        NSE < 0.0: PEOR que predecir la media historica.

    Criterio de exito: NSE > 0.80 (aceptable), NSE > 0.90 (bueno).

    Args:
        y_true: Valores observados.
        y_pred: Valores predichos.

    Returns:
        NSE. Rango: (-inf, 1].
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else -np.inf


def compute_kge(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Calcula Kling-Gupta Efficiency y sus componentes.

    KGE = 1 - sqrt[(r-1)^2 + (alpha-1)^2 + (beta-1)^2]
    Donde:
        r     = correlacion de Pearson (timing)
        alpha = ratio de desviaciones estandar (variabilidad)
        beta  = ratio de medias (sesgo/bias)

    Ventaja sobre NSE: Descompone el error en 3 componentes diagnosticables.
    Criterio de exito: KGE > 0.75.

    Args:
        y_true: Valores observados.
        y_pred: Valores predichos.

    Returns:
        Tupla (kge, r, alpha, beta).
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    r     = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else np.nan
    beta  = float(np.mean(y_pred) / np.mean(y_true)) if np.mean(y_true) != 0 else np.nan
    kge   = float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
    return kge, r, alpha, beta


def classify_regime(date: pd.Timestamp) -> str:
    """Clasifica una fecha en su regimen hidrologico del Orinoco.

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


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula MAE, RMSE, NSE y KGE en un solo diccionario.

    Args:
        y_true: Valores observados.
        y_pred: Valores predichos.

    Returns:
        Dict con mae, rmse, nse, kge, kge_r, kge_alpha, kge_beta.
    """
    kge, r, alpha, beta = compute_kge(y_true, y_pred)
    return {
        "mae":       compute_mae(y_true, y_pred),
        "rmse":      compute_rmse(y_true, y_pred),
        "nse":       compute_nse(y_true, y_pred),
        "kge":       kge,
        "kge_r":     r,
        "kge_alpha": alpha,
        "kge_beta":  beta,
    }


def compute_metrics_by_regime(
    y_true: pd.Series,
    y_pred: pd.Series,
    dates: pd.DatetimeIndex,
) -> Dict[str, Dict[str, float]]:
    """Calcula todas las metricas desagregadas por regimen hidrologico.

    Args:
        y_true: Serie de valores observados con indice DatetimeIndex.
        y_pred: Serie de valores predichos.
        dates: Indice temporal de las predicciones.

    Returns:
        Dict anidado: {regimen: {metrica: valor}}
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    regimes = [classify_regime(d) for d in dates]
    regime_series = pd.Series(regimes, index=dates)

    results = {"global": compute_all_metrics(y_true, y_pred)}
    for regime in ["aguas_bajas", "ascenso", "aguas_altas", "descenso"]:
        mask = np.array(regime_series == regime)
        if mask.sum() < 5:
            results[regime] = {}
            continue
        results[regime] = compute_all_metrics(y_true[mask], y_pred[mask])
    return results


def detect_shadow_effect(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Detecta el shadow effect (lag-1 copying) en las predicciones.

    Si corr(y_hat(t), y(t-1)) > corr(y_hat(t), y(t)), el modelo esta copiando
    el ultimo valor conocido en lugar de predecir. BANDERA ROJA (AGENT_RULES R8).

    Args:
        y_true: Valores reales.
        y_pred: Predicciones del modelo.

    Returns:
        Dict con 'corr_pred_real', 'corr_pred_lag1', 'shadow_effect_detected'.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    min_len = min(len(y_true), len(y_pred))
    y_true  = y_true[:min_len]
    y_pred  = y_pred[:min_len]

    corr_real = float(np.corrcoef(y_pred[1:], y_true[1:])[0, 1])
    corr_lag1 = float(np.corrcoef(y_pred[1:], y_true[:-1])[0, 1])
    shadow    = bool(corr_lag1 > corr_real)

    if shadow:
        logger.warning(
            "BANDERA ROJA — Shadow Effect detectado: corr(pred, y_lag1)=%.4f > corr(pred, y_true)=%.4f",
            corr_lag1, corr_real,
        )
    else:
        logger.info(
            "Shadow Effect: NO detectado. corr(pred, y_true)=%.4f > corr(pred, y_lag1)=%.4f",
            corr_real, corr_lag1,
        )

    return {
        "corr_pred_real":       corr_real,
        "corr_pred_lag1":       corr_lag1,
        "shadow_effect_detected": shadow,
    }
