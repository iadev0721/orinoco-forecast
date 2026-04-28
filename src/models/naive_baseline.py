"""
src/models/naive_baseline.py
==============================
Modelos Naive de referencia. DEBEN ejecutarse antes que cualquier red neuronal.

REGLA R3: BASELINE PRIMERO. results/metrics/baseline_metrics.json
debe existir antes de entrenar LSTM o Transformer.

Modelos implementados:
    - NaiveBaseline    : y_hat(t+h) = y(t)         para h = 1,...,7
    - SeasonalNaive    : y_hat(t+h) = y(t+h-365)
"""
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NaiveBaseline:
    """Modelo Naive: predice el ultimo valor conocido para todos los horizontes.

    y_hat(t+h) = y(t)  para h = 1,..., forecast_horizon

    Justificacion (Hyndman, 2018): El modelo mas simple posible.
    Todo modelo de ML debe demostrar que lo supera.
    """

    def __init__(self, horizon: int = 7) -> None:
        """
        Args:
            horizon: Numero de dias a predecir.
        """
        self.horizon = horizon

    def predict(self, y_last: np.ndarray) -> np.ndarray:
        """Genera predicciones naive.

        Args:
            y_last: Array de ultimos valores conocidos. Shape: (n_samples,).

        Returns:
            Predicciones. Shape: (n_samples, horizon).
        """
        return np.tile(y_last[:, np.newaxis], (1, self.horizon)).astype(np.float32)


class SeasonalNaive:
    """Modelo Seasonal Naive: predice el valor del mismo dia del ano anterior.

    y_hat(t+h) = y(t+h-365)

    Captura la estacionalidad dominante del Orinoco.
    """

    def __init__(self, horizon: int = 7) -> None:
        """
        Args:
            horizon: Numero de dias a predecir.
        """
        self.horizon = horizon

    def predict(self, df: pd.DataFrame, target_col: str, forecast_date: pd.Timestamp) -> np.ndarray:
        """Genera predicciones seasonal naive.

        Args:
            df: DataFrame historico completo con indice DatetimeIndex.
            target_col: Nombre de la columna target.
            forecast_date: Fecha desde la cual se predice.

        Returns:
            Predicciones. Shape: (horizon,).
        """
        preds = []
        for h in range(1, self.horizon + 1):
            target_date = forecast_date + pd.Timedelta(days=h)
            seasonal_date = target_date - pd.DateOffset(years=1)
            if seasonal_date in df.index:
                val = df.loc[seasonal_date, target_col]
            else:
                candidates = df.loc[:seasonal_date, target_col]
                val = candidates.iloc[-1] if len(candidates) > 0 else np.nan
            preds.append(val)
        return np.array(preds, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula MAE, RMSE, NSE y KGE.

    Args:
        y_true: Valores observados. Shape: (n,) o (n, horizon).
        y_pred: Valores predichos. Misma forma.

    Returns:
        Diccionario con las metricas.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    nse    = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else -np.inf

    r     = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else np.nan
    beta  = float(np.mean(y_pred) / np.mean(y_true)) if np.mean(y_true) != 0 else np.nan
    kge   = float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

    return {
        "mae": mae, "rmse": rmse, "nse": nse, "kge": kge,
        "kge_r": r, "kge_alpha": alpha, "kge_beta": beta,
    }


def run_baselines(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    horizon: int,
    output_path: str = "results/metrics/baseline_metrics.json",
) -> Dict:
    """Ejecuta Naive y SeasonalNaive sobre el test set y persiste las metricas.

    REGLA R3: Este archivo es el gate que desbloquea el entrenamiento LSTM.

    Args:
        df: DataFrame historico completo (train+val+test) para SeasonalNaive.
        df_test: Subconjunto de test unicamente.
        target_col: Columna del target.
        horizon: Dias a predecir.
        output_path: Ruta de salida del JSON de metricas.

    Returns:
        Diccionario con metricas de ambos modelos.
    """
    naive   = NaiveBaseline(horizon=horizon)
    sea_nav = SeasonalNaive(horizon=horizon)

    y_trues_naive, y_preds_naive   = [], []
    y_trues_season, y_preds_season = [], []

    dates = df_test.index
    max_t = len(dates) - horizon

    for i in range(max_t):
        t = dates[i]
        y_future = df_test[target_col].iloc[i + 1: i + 1 + horizon].values
        if len(y_future) < horizon:
            continue

        y_last  = np.array([df_test[target_col].iloc[i]])
        p_naive = naive.predict(y_last)[0]
        y_trues_naive.append(y_future)
        y_preds_naive.append(p_naive)

        p_season = sea_nav.predict(df, target_col, t)
        y_trues_season.append(y_future)
        y_preds_season.append(p_season)

    y_trues_naive  = np.array(y_trues_naive)
    y_preds_naive  = np.array(y_preds_naive)
    y_trues_season = np.array(y_trues_season)
    y_preds_season = np.array(y_preds_season)

    metrics_naive  = compute_metrics(y_trues_naive,  y_preds_naive)
    metrics_season = compute_metrics(y_trues_season, y_preds_season)

    results = {
        "split": "test",
        "target": target_col,
        "horizon_days": horizon,
        "n_samples": len(y_trues_naive),
        "naive_baseline": metrics_naive,
        "seasonal_naive": metrics_season,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Baseline metrics guardadas en: %s", output_path)

    logger.info("=== RESULTADOS BASELINE (Test Set) ===")
    logger.info("Naive        -> MAE: %.3f m | RMSE: %.3f m | NSE: %.4f | KGE: %.4f",
                metrics_naive["mae"], metrics_naive["rmse"],
                metrics_naive["nse"],  metrics_naive["kge"])
    logger.info("SeasonalNaive-> MAE: %.3f m | RMSE: %.3f m | NSE: %.4f | KGE: %.4f",
                metrics_season["mae"], metrics_season["rmse"],
                metrics_season["nse"],  metrics_season["kge"])
    return results
