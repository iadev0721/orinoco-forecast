"""
src/evaluation/scenario_evaluator.py
=======================================
Evaluación de modelos por escenario/régimen hidrológico.

Genera la tabla de resultados exigida en docs/04_MANIFESTO_EVALUATION.md:

| Métrica | Global | Aguas Bajas | Ascenso | Aguas Altas | Descenso |
|---------|--------|-------------|---------|-------------|----------|
| MAE (m) |   —    |      —      |    —    |      —      |    —     |
| RMSE (m)|   —    |      —      |    —    |      —      |    —     |
| NSE     |   —    |      —      |    —    |      —      |    —     |
| KGE     |   —    |      —      |    —    |      —      |    —     |
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 5 (Evaluación)


def evaluate_model_by_regime(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Evalúa un modelo y genera la tabla completa por régimen.

    Args:
        model_name: Nombre del modelo (ej: 'LSTM', 'SARIMA', 'Naive').
        y_true: Valores observados del nivel del río.
        y_pred: Predicciones del modelo.
        dates: Índice temporal de las predicciones.

    Returns:
        DataFrame con métricas por régimen hidrológico.
    """
    raise NotImplementedError("Implementar en Fase 5")


def compare_all_models(
    results: Dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    """Genera la tabla comparativa de todos los modelos y la guarda en CSV.

    Args:
        results: Dict {nombre_modelo: DataFrame de métricas}.
        output_path: Ruta de salida CSV.
    """
    raise NotImplementedError("Implementar en Fase 5")
