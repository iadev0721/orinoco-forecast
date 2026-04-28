"""
src/evaluation/scenario_evaluator.py
=======================================
Evaluacion de modelos por escenario/regimen hidrologico.

Genera la tabla de resultados exigida en docs/04_MANIFESTO_EVALUATION.md:

| Metrica | Global | Aguas Bajas | Ascenso | Aguas Altas | Descenso |
|---------|--------|-------------|---------|-------------|----------|
| MAE (m) |   --   |      --     |    --   |      --     |    --    |
| RMSE (m)|   --   |      --     |    --   |      --     |    --    |
| NSE     |   --   |      --     |    --   |      --     |    --    |
| KGE     |   --   |      --     |    --   |      --     |    --    |
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics_by_regime

logger = logging.getLogger(__name__)

REGIMES = ["global", "aguas_bajas", "ascenso", "aguas_altas", "descenso"]
REGIME_LABELS = {
    "global":      "Global",
    "aguas_bajas": "Aguas Bajas",
    "ascenso":     "Ascenso",
    "aguas_altas": "Aguas Altas",
    "descenso":    "Descenso",
}


def evaluate_model_by_regime(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Evalua un modelo y genera la tabla completa por regimen.

    Args:
        model_name: Nombre del modelo (ej: 'LSTM', 'SARIMA', 'Naive').
        y_true: Valores observados del nivel del rio.
        y_pred: Predicciones del modelo.
        dates: Indice temporal de las predicciones.

    Returns:
        DataFrame con metricas por regimen hidrologico.
        Filas: metricas (MAE, RMSE, NSE, KGE).
        Columnas: regimenes (Global, Aguas Bajas, ...).
    """
    metrics_by_regime = compute_metrics_by_regime(y_true, y_pred, dates)

    rows = []
    for metric_key, metric_label in [
        ("mae",  "MAE (m)"),
        ("rmse", "RMSE (m)"),
        ("nse",  "NSE"),
        ("kge",  "KGE"),
    ]:
        row = {"Metrica": metric_label, "Modelo": model_name}
        for regime in REGIMES:
            val = metrics_by_regime.get(regime, {}).get(metric_key, float("nan"))
            row[REGIME_LABELS[regime]] = round(val, 4) if not np.isnan(val) else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows).set_index(["Modelo", "Metrica"])
    logger.info("Tabla de evaluacion por regimen generada para '%s'", model_name)
    return df


def compare_all_models(
    results: Dict[str, pd.DataFrame],
    output_path: str,
) -> pd.DataFrame:
    """Genera la tabla comparativa de todos los modelos y la guarda en CSV.

    Args:
        results: Dict {nombre_modelo: DataFrame de metricas por regimen}.
        output_path: Ruta de salida CSV.

    Returns:
        DataFrame concatenado con todos los modelos.
    """
    if not results:
        logger.warning("No hay resultados para comparar.")
        return pd.DataFrame()

    combined = pd.concat(results.values())
    combined.to_csv(output_path)
    logger.info("Tabla comparativa guardada en: %s", output_path)

    # Imprimir en consola de forma legible
    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA DE MODELOS — TEST SET")
    print("=" * 70)
    print(combined.to_string())
    print("=" * 70 + "\n")
    return combined
