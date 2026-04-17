"""
tests/test_no_data_leakage.py
================================
Tests que verifican que el pipeline NO introduce data leakage.

REGLA R1: El scaler se ajusta EXCLUSIVAMENTE con datos de entrenamiento.
El split cronológico es siempre el PRIMER paso.

Estos tests deben pasar en VERDE antes de cualquier entrenamiento.
Ejecutar con: pytest tests/test_no_data_leakage.py -v
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# TODO: Implementar en Fase 2 (Preprocessing)
# Conectar con src.data.splitter, src.data.scaler, src.data.windower


class TestScalerFitOnlyOnTrain:
    """Verifica que el scaler se ajusta solo sobre el train set."""

    def test_scaler_min_within_train_bounds(self) -> None:
        """El mínimo del scaler debe corresponder al train set, no al total."""
        pytest.skip("Implementar en Fase 2 — conectar con src.data.scaler")

    def test_scaler_max_within_train_bounds(self) -> None:
        """El máximo del scaler debe corresponder al train set, no al total."""
        pytest.skip("Implementar en Fase 2 — conectar con src.data.scaler")

    def test_scaler_not_refitted_on_val(self) -> None:
        """El scaler NO debe ajustarse de nuevo sobre val o test."""
        pytest.skip("Implementar en Fase 2 — verificar que transform() no llama fit()")


class TestChronologicalSplit:
    """Verifica que el split es estrictamente cronológico."""

    def test_train_ends_before_val(self) -> None:
        """El último día de train debe ser anterior al primer día de val."""
        pytest.skip("Implementar en Fase 2 — conectar con src.data.splitter")

    def test_val_ends_before_test(self) -> None:
        """El último día de val debe ser anterior al primer día de test."""
        pytest.skip("Implementar en Fase 2 — conectar con src.data.splitter")

    def test_no_overlap_in_targets(self) -> None:
        """Los targets (y) de train no deben solapar con los días de val."""
        pytest.skip("Implementar en Fase 2 — conectar con src.data.windower")


class TestWindowBoundaries:
    """Verifica que las ventanas no introducen leakage en los bordes."""

    def test_last_train_window_target_in_train(self) -> None:
        """El target de la última ventana de train debe estar en el período train."""
        pytest.skip("Implementar en Fase 2 — verificar windower.verify_no_leakage_at_boundaries()")

    def test_no_nan_in_tensors(self) -> None:
        """Los tensores finales X e y no deben contener NaN."""
        pytest.skip("Implementar en Fase 2 — verificar PC-02-05")
