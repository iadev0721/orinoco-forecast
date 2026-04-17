"""
src/models/transformer_model.py
=================================
Transformer para Series Temporales en PyTorch. PLUS ACADÉMICO.

El mecanismo de self-attention permite al modelo ponderar directamente
qué días del pasado son más relevantes para la predicción.

RESTRICCIONES DE HARDWARE (RTX 3060, 6 GB VRAM):
    - batch_size MÁXIMO: 32
    - d_model MÁXIMO: 64
    - Si OOM: reducir batch_size a 16, no reducir d_model.

REGLA R5:
    import torch
    torch.cuda.empty_cache()  # Al inicio de cada experimento

REGLA R3: check_baseline_gate() debe ejecutarse antes del entrenamiento.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# TODO: Implementar en Fase 4 (Plus Académico)
# Prerequisito: LSTM entrenado y evaluado (baseline_metrics.json existe)


def build_transformer_model(config: dict, n_features: int, horizon: int):
    """Construye el modelo Transformer según la configuración.

    Arquitectura:
        Positional Encoding
        TransformerEncoder (num_layers capas de self-attention)
        Linear(d_model → horizon)

    Hiperparámetros (de config.yaml):
        d_model: 64
        nhead: 4
        num_layers: 2
        dim_feedforward: 128
        batch_size: 32 (máximo RTX 3060)

    Args:
        config: Configuración del experimento.
        n_features: Número de features en el tensor de entrada.
        horizon: Número de días a predecir.

    Returns:
        Modelo PyTorch nn.Module.
    """
    raise NotImplementedError("Implementar en Fase 4")


def train_transformer(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> dict:
    """Entrena el modelo Transformer con EarlyStopping.

    Args:
        model: Modelo PyTorch.
        X_train: Tensores de entrenamiento.
        y_train: Targets de entrenamiento.
        X_val: Tensores de validación.
        y_val: Targets de validación.
        config: Configuración del experimento.

    Returns:
        Dict con train_loss y val_loss por época.
    """
    raise NotImplementedError("Implementar en Fase 4")
