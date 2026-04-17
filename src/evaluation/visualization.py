"""
src/evaluation/visualization.py
=================================
Gráficas de publicación para la tesis sobre predicción hidrológica.

Estándar de salida: PNG 300 DPI (publicable en revista científica).
Directorio de salida: results/figures/

Gráficas requeridas (docs/04_MANIFESTO_EVALUATION.md):
    1. Serie temporal test set: reales vs predichos (todos los modelos)
    2. Scatter plot real vs predicho por régimen hidrológico
    3. Heatmap de métricas (modelos × régimen)
    4. Curvas de aprendizaje (train_loss vs val_loss)
    5. Análisis de residuos por régimen
    6. Zoom en eventos de crecida (mejores y peores días)
    7. Heatmap de attention weights (si aplica)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Estilo de publicación
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# TODO: Implementar en Fase 5 (Evaluación)


def plot_predictions_vs_actual(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    output_path: str,
    regime_colors: bool = True,
) -> None:
    """Gráfica de serie temporal: valores reales vs predichos de todos los modelos.

    Args:
        dates: Índice temporal del test set.
        y_true: Valores reales.
        predictions_dict: Dict {nombre_modelo: predicciones}.
        output_path: Ruta de salida del PNG.
        regime_colors: Si True, colorea el fondo según régimen hidrológico.
    """
    raise NotImplementedError("Implementar en Fase 5")


def plot_scatter_by_regime(
    y_true: pd.Series,
    y_pred: pd.Series,
    dates: pd.DatetimeIndex,
    model_name: str,
    output_path: str,
) -> None:
    """Scatter plot real vs predicho coloreado por régimen hidrológico.

    Args:
        y_true: Valores reales.
        y_pred: Predicciones del modelo.
        dates: Índice temporal.
        model_name: Nombre del modelo para el título.
        output_path: Ruta de salida del PNG.
    """
    raise NotImplementedError("Implementar en Fase 5")


def plot_learning_curves(
    train_loss: List[float],
    val_loss: List[float],
    model_name: str,
    output_path: str,
    early_stopping_epoch: Optional[int] = None,
) -> None:
    """Curvas de aprendizaje (train vs validation loss por época).

    Args:
        train_loss: Loss de entrenamiento por época.
        val_loss: Loss de validación por época.
        model_name: Nombre del modelo.
        output_path: Ruta de salida del PNG.
        early_stopping_epoch: Época donde se activó EarlyStopping.
    """
    raise NotImplementedError("Implementar en Fase 5")


def plot_metrics_heatmap(
    comparison_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Heatmap de métricas por modelo y régimen hidrológico.

    Args:
        comparison_df: DataFrame con modelos como filas y métricas×régimen como columnas.
        output_path: Ruta de salida del PNG.
    """
    raise NotImplementedError("Implementar en Fase 5")
