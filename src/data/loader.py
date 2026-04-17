"""
src/data/loader.py
==================
Carga y validación inicial de los datos hidrométricos del Orinoco.

Responsabilidades:
    - Leer dataset_original.xlsx y dataset_imputado_simpleml.csv
    - Validar columnas, tipos de datos e índice temporal
    - Detectar anomalías de sensor (valores físicamente imposibles)
    - Retornar DataFrames con índice DatetimeIndex

REGLA R6: logging, no print(). Type hints obligatorios.
REGLA R7: Rutas leídas desde config.yaml, no hardcodeadas.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 0


def load_config(config_path: str = "config.yaml") -> dict:
    """Carga la configuración central del experimento.

    Args:
        config_path: Ruta al archivo config.yaml.

    Returns:
        Diccionario con toda la configuración del experimento.
    """
    raise NotImplementedError("Implementar en Fase 0")


def load_raw_original(config: dict) -> pd.DataFrame:
    """Carga el dataset original con brechas (INTOCABLE — solo lectura).

    Args:
        config: Configuración del experimento cargada con load_config().

    Returns:
        DataFrame con índice DatetimeIndex y columnas por estación.
    """
    raise NotImplementedError("Implementar en Fase 0")


def load_raw_imputed(config: dict) -> pd.DataFrame:
    """Carga el dataset imputado por Simple ML (BAJO SOSPECHA — sujeto a auditoría).

    Args:
        config: Configuración del experimento cargada con load_config().

    Returns:
        DataFrame con índice DatetimeIndex y columnas por estación.
    """
    raise NotImplementedError("Implementar en Fase 0")


def detect_sensor_anomalies(series: pd.Series) -> pd.Series:
    """Detecta patrones típicos de fallos de sensor en datos hidrométricos.

    Patrones detectados:
        1. Valores constantes > 3 días (sensor atascado).
        2. Saltos > 2m en un día durante aguas bajas.
        3. Valores negativos (físicamente imposible).
        4. Valores > máximo histórico + 20%.

    Args:
        series: Serie temporal del nivel del río en metros.

    Returns:
        Serie de flags (0 = normal, >0 = sospechoso).
    """
    raise NotImplementedError("Implementar en Fase 0")


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Valida que el DataFrame cumple los requisitos mínimos.

    Args:
        df: DataFrame a validar.

    Returns:
        True si el DataFrame es válido.

    Raises:
        ValueError: Si el DataFrame no pasa alguna validación.
    """
    raise NotImplementedError("Implementar en Fase 0")
