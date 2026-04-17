"""
src/data/splitter.py
====================
Partición CRONOLÓGICA del dataset en train/val/test.

REGLA R1 — CERO DATA LEAKAGE:
    El split SIEMPRE es el PRIMER paso del pipeline.
    NUNCA se escalan datos antes de particionar.

Fechas de corte (de config.yaml):
    - Train:      1974-01-01 → 2015-12-31
    - Validation: 2016-01-01 → 2020-12-31
    - Test:       2021-01-01 → 2025-02-24
"""
import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 2 (Preprocessing)


def chronological_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Particiona el DataFrame en splits cronológicos estrictos.

    No hay aleatoriedad. No hay shuffle. El tiempo va hacia adelante.

    Args:
        df: DataFrame con índice DatetimeIndex ordenado cronológicamente.
        train_end: Fecha fin del split de train (ej: "2015-12-31").
        val_end: Fecha fin del split de validación (ej: "2020-12-31").

    Returns:
        Tupla (df_train, df_val, df_test).

    Raises:
        ValueError: Si las fechas de corte no son válidas o el DataFrame no está ordenado.
    """
    raise NotImplementedError("Implementar en Fase 2")
