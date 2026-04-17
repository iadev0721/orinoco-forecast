"""
src/models/statistical_baseline.py
=====================================
Modelos estadísticos de referencia: SARIMA y Prophet.

SARIMA es el benchmark estándar en hidrología (Hyndman, 2018).
Si el LSTM no supera a SARIMA, no se justifica la complejidad de deep learning.

REGLA R3: Este módulo genera results/metrics/baseline_metrics.json,
que es el gate de entrada para el entrenamiento de LSTM.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 3a (Baseline)


class SARIMABaseline:
    """Modelo SARIMA para predicción univariada de la estación target.

    Justificación: Captura autocorrelación y estacionalidad.
    Es el benchmark estándar en series temporales hidrológicas.

    Nota: SARIMA es univariado. No usa estaciones aguas arriba.
    Si el LSTM multivariado no lo supera, la complejidad no está justificada.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Configuración del experimento.
        """
        self.config = config
        self.model = None

    def fit(self, y_train: pd.Series) -> None:
        """Ajusta el modelo SARIMA sobre los datos de entrenamiento.

        Args:
            y_train: Serie temporal de la estación target (train set).
        """
        raise NotImplementedError("Implementar en Fase 3a")

    def predict(self, steps: int) -> np.ndarray:
        """Genera predicciones h-pasos hacia adelante.

        Args:
            steps: Número de pasos a predecir.

        Returns:
            Predicciones. Shape: (steps,).
        """
        raise NotImplementedError("Implementar en Fase 3a")


def save_baseline_metrics(metrics: Dict, output_path: str) -> None:
    """Guarda las métricas de todos los baselines en JSON.

    Este archivo es el gate de entrada para el entrenamiento de LSTM.
    Ver REGLA R3 en AGENT_RULES.md.

    Args:
        metrics: Diccionario con métricas de todos los modelos baseline.
        output_path: Ruta de salida (ej: 'results/metrics/baseline_metrics.json').
    """
    raise NotImplementedError("Implementar en Fase 3a")
