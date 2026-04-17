"""
tests/test_physical_constraints.py
=====================================
Tests que verifican restricciones físicas del río Orinoco en las predicciones.

REGLA R4: El nivel del río NUNCA puede ser negativo.
Las predicciones deben pasar por clamp físico antes de ser reportadas.

Ejecutar con: pytest tests/test_physical_constraints.py -v
"""
import pytest
import numpy as np


# TODO: Implementar en Fase 3b (LSTM) y Fase 4 (Transformer)


class TestPhysicalConstraints:
    """Verifica que las predicciones respetan la física del río."""

    def test_predictions_non_negative(self) -> None:
        """Las predicciones no deben contener valores negativos (REGLA R4)."""
        pytest.skip("Implementar en Fase 3b — conectar con lstm_model.apply_physical_constraints()")

    def test_predictions_not_exceed_historical_max(self) -> None:
        """Las predicciones no deben exceder el máximo histórico + 15%."""
        pytest.skip("Implementar en Fase 3b — verificar clamp de predicciones")

    def test_daily_change_alert_threshold(self) -> None:
        """Cambios diarios > 1.5m deben generar una alerta (warning)."""
        pytest.skip("Implementar en Fase 3b — verificar logging de cambios abruptos")

    def test_river_level_always_positive(self) -> None:
        """El nivel del río en todo el dataset procesado debe ser ≥ 0."""
        pytest.skip("Implementar en Fase 0 — conectar con loader.validate_dataframe()")


class TestBaselineGate:
    """Verifica que la guardia de baseline funciona correctamente."""

    def test_lstm_raises_without_baseline_metrics(self) -> None:
        """check_baseline_gate() debe fallar si baseline_metrics.json no existe."""
        pytest.skip("Implementar en Fase 3b — conectar con lstm_model.check_baseline_gate()")
