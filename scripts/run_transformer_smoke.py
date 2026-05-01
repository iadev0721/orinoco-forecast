"""
scripts/run_transformer_smoke.py
=================================
Smoke-test breve para el Transformer: entrena 2 épocas con batch pequeño
para validar el pipeline sin sobrecargar la CPU/GPU.

Uso:
  python scripts/run_transformer_smoke.py

El script evita importar TensorFlow y no toca configuraciones globales.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Añadir raíz del repo a sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.pipeline import load_config, build_tensors
from src.models.transformer_model import build_transformer_model, train_transformer, predict_transformer
from src.evaluation.experiment_tracker import ExperimentTracker
from src.evaluation.metrics import compute_all_metrics, detect_shadow_effect
from src.utils.gpu_config import configure_pytorch_gpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config()

    # Overrides seguros para smoke-test
    cfg = dict(cfg)  # shallow copy
    transformer_cfg = dict(cfg.get("transformer", {}))
    transformer_cfg["max_epochs"] = 2
    transformer_cfg["batch_size"] = 8
    cfg["transformer"] = transformer_cfg

    # Semillas (sin importar TensorFlow)
    np.random.seed(cfg.get("seed", 42))
    try:
        import torch

        torch.manual_seed(cfg.get("seed", 42))
    except Exception:
        logger.debug("PyTorch no disponible para setear semilla.")

    # Preparar dispositivo PyTorch
    configure_pytorch_gpu()

    features_path = transformer_cfg.get("features_path")
    if not features_path:
        raise ValueError("'features_path' no definido en config.yaml bajo la sección transformer.")
    logger.info("Usando features: %s", features_path)

    tensors = build_tensors(features_path=features_path, cfg_override=cfg)
    X_train = tensors["X_train"]
    y_train = tensors["y_train"]
    X_val = tensors["X_val"]
    y_val = tensors["y_val"]
    X_test = tensors["X_test"]
    y_test = tensors["y_test"]
    scaler_y = tensors["scaler_y"]
    test_dates = tensors["test_dates"]

    tracker = ExperimentTracker(experiment_name="transformer_smoke", model_type="transformer")
    tracker.log_config(cfg)

    model = build_transformer_model(cfg, n_features=X_train.shape[2], horizon=cfg["forecast_horizon"])
    history = train_transformer(model, X_train, y_train, X_val, y_val, config=cfg, experiment_name=tracker.name)
    tracker.log_training_history(history)

    def inv(arr_2d: np.ndarray) -> np.ndarray:
        result = np.zeros_like(arr_2d)
        for col in range(arr_2d.shape[1]):
            result[:, col] = scaler_y.inverse_transform(arr_2d[:, col].reshape(-1, 1)).ravel()
        return result

    y_pred_real = inv(predict_transformer(model, X_test, batch_size=transformer_cfg.get("batch_size", 8)))
    y_test_real = inv(y_test)

    # No aplicar restricciones físicas aquí para mantener test limpio; solo calculamos métricas
    test_metrics = compute_all_metrics(y_test_real, y_pred_real)
    tracker.log_metrics(test_metrics, split="test")

    detect_shadow_effect(
        np.array(y_test_real[:, -1]),
        np.array(y_pred_real[:, -1]),
    )

    tracker.log_predictions(test_dates, y_test_real[:, -1], y_pred_real[:, -1], horizon=cfg["forecast_horizon"])
    exp_dir = tracker.save()
    logger.info("Smoke-test completado. Experimento guardado en: %s", exp_dir)


if __name__ == "__main__":
    main()
