"""
tests/test_window_integrity.py
=================================
Tests que verifican la integridad de las ventanas deslizantes.

Valida que las dimensiones y el contenido de los tensores X e y
son correctas para el modelado.

Ejecutar con: pytest tests/test_window_integrity.py -v
"""
import pytest
import numpy as np


# TODO: Implementar en Fase 2 (Preprocessing)


class TestWindowDimensions:
    """Verifica que los tensores tienen las dimensiones correctas."""

    def test_x_train_is_3d(self) -> None:
        """X_train debe ser 3D: (n_samples, lookback, n_features)."""
        pytest.skip("Implementar en Fase 2 — conectar con windower")

    def test_y_train_is_2d(self) -> None:
        """y_train debe ser 2D: (n_samples, horizon)."""
        pytest.skip("Implementar en Fase 2 — conectar con windower")

    def test_lookback_matches_config(self) -> None:
        """La dimensión lookback de X debe coincidir con config.yaml."""
        pytest.skip("Implementar en Fase 2")

    def test_horizon_matches_config(self) -> None:
        """La dimensión horizon de y debe coincidir con config.yaml."""
        pytest.skip("Implementar en Fase 2")


class TestWindowContent:
    """Verifica el contenido semántico de las ventanas."""

    def test_y_is_correct_future_slice(self) -> None:
        """El target y(i) debe ser los 'horizon' días siguientes al input X(i)."""
        pytest.skip("Implementar en Fase 2")

    def test_sufficient_lookback_for_max_lag(self) -> None:
        """El lookback debe cubrir el lag máximo Ayacucho→Palúa (PC-02-03)."""
        pytest.skip("Implementar en Fase 2 — leer eda_lag_times.json y verificar")
