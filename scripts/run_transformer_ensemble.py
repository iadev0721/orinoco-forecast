"""
scripts/run_transformer_ensemble.py
=====================================
Ensemble de N Transformers con semillas distintas y paradigma residual.

Sigue la misma filosofía que run_ensemble.py del LSTM gold standard:
    - N miembros entrenados con semillas distintas
    - Promedio simple de predicciones (reduce varianza estocástica)
    - Comparación directa contra el LSTM gold standard

Uso:
    python scripts/run_transformer_ensemble.py --name ensemble_transformer_v1 --n 5
    python scripts/run_transformer_ensemble.py --name ensemble_transformer_xl --n 5 --d_model 128 --num_layers 3

Referencia gold standard LSTM:
    ensemble_lb150_lags_xl -> MAE=13.3cm | NSE=0.9959 | KGE=0.9943
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import build_tensors, load_config
from src.evaluation.experiment_tracker import ExperimentTracker
from src.evaluation.metrics import compute_all_metrics, detect_shadow_effect
from src.models.lstm_model import apply_physical_constraints
from src.models.transformer_model import (
    build_transformer_model,
    predict_transformer,
    train_transformer,
)
from src.utils.gpu_config import configure_pytorch_gpu
from src.utils.reproducibility import set_global_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1011]


def run_ensemble(cfg: dict, args: argparse.Namespace) -> None:
    """Entrena N Transformers y promedia sus predicciones."""
    configure_pytorch_gpu()

    # Aplicar overrides de arquitectura del Transformer
    transformer_cfg = dict(cfg.get("transformer", {}))
    if args.d_model is not None:
        transformer_cfg["d_model"] = args.d_model
    if args.nhead is not None:
        transformer_cfg["nhead"] = args.nhead
    if args.num_layers is not None:
        transformer_cfg["num_layers"] = args.num_layers
    if args.dim_feedforward is not None:
        transformer_cfg["dim_feedforward"] = args.dim_feedforward
    if args.dropout is not None:
        transformer_cfg["dropout"] = args.dropout
    if args.lr is not None:
        transformer_cfg["learning_rate"] = args.lr
    if args.epochs is not None:
        transformer_cfg["max_epochs"] = args.epochs
    if args.patience is not None:
        transformer_cfg["patience"] = args.patience
    if args.loss is not None:
        transformer_cfg["loss"] = args.loss
    if args.huber_delta is not None:
        transformer_cfg["huber_delta"] = args.huber_delta
    cfg["transformer"] = transformer_cfg

    logger.info("Configuracion Transformer: %s", transformer_cfg)

    features_path = transformer_cfg.get(
        "features_path", "data/processed/dataset_orinoco_features.csv"
    )

    # Construir tensores una sola vez (todos los miembros usan el mismo dataset)
    tensors = build_tensors(features_path=features_path, cfg_override=cfg)
    X_train    = tensors["X_train"]
    y_train    = tensors["y_train"]
    X_val      = tensors["X_val"]
    y_val      = tensors["y_val"]
    X_test     = tensors["X_test"]
    y_test     = tensors["y_test"]
    scaler_y   = tensors["scaler_y"]
    test_dates = tensors["test_dates"]
    use_residual = tensors["use_residual"]
    anch_test  = tensors["anch_test"]
    anch_val   = tensors["anch_val"]

    logger.info(
        "Tensores: X_train=%s | X_val=%s | X_test=%s | use_residual=%s",
        X_train.shape, X_val.shape, X_test.shape, use_residual,
    )

    # ──────────────────────────────────────────────────────────────
    # Funciones de reconstruccion
    # ──────────────────────────────────────────────────────────────
    def inv(arr_2d: np.ndarray) -> np.ndarray:
        result = np.zeros_like(arr_2d)
        for col in range(arr_2d.shape[1]):
            result[:, col] = scaler_y.inverse_transform(
                arr_2d[:, col].reshape(-1, 1)
            ).ravel()
        return result

    def reconstruct(raw: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        if use_residual:
            return inv(anchors.reshape(-1, 1) + raw)
        return inv(raw)

    # Ground truth en metros
    y_test_real = inv(y_test + anch_test.reshape(-1, 1)) if use_residual else inv(y_test)
    y_val_real  = inv(y_val  + anch_val.reshape(-1,  1)) if use_residual else inv(y_val)
    y_train_max = float(scaler_y.inverse_transform([[1.0]])[0, 0])

    # ──────────────────────────────────────────────────────────────
    # Entrenamiento de miembros
    # ──────────────────────────────────────────────────────────────
    seeds = SEEDS[: args.n]
    preds_test_members: list[np.ndarray] = []
    preds_val_members:  list[np.ndarray] = []

    logger.info("=" * 60)
    logger.info("ENSEMBLE: %d miembros | semillas: %s", args.n, seeds)
    logger.info("=" * 60)

    for i, seed in enumerate(seeds, start=1):
        logger.info("──────────── Miembro %d/%d | seed=%d ────────────", i, args.n, seed)
        set_global_seeds(seed)

        model = build_transformer_model(
            cfg,
            n_features=X_train.shape[2],
            horizon=cfg["forecast_horizon"],
        )
        history = train_transformer(
            model, X_train, y_train, X_val, y_val,
            config=cfg,
            experiment_name=f"{args.name}_member{i}",
        )
        best_val = min(history["val_loss"])
        logger.info(
            "Miembro %d: entrenado %d epocas | mejor val_loss=%.6f",
            i, len(history["val_loss"]), best_val,
        )

        raw_test = predict_transformer(
            model, X_test, batch_size=transformer_cfg.get("batch_size", 32)
        )
        raw_val = predict_transformer(
            model, X_val, batch_size=transformer_cfg.get("batch_size", 32)
        )
        y_pred_m = apply_physical_constraints(
            reconstruct(raw_test, anch_test), cfg, y_train_max=y_train_max
        )
        y_pred_val_m = reconstruct(raw_val, anch_val)

        mae_m = np.mean(np.abs(y_test_real[:, -1] - y_pred_m[:, -1])) * 100
        logger.info("Miembro %d: MAE_test=%.3fm", i, mae_m / 100)

        preds_test_members.append(y_pred_m)
        preds_val_members.append(y_pred_val_m)

    # ──────────────────────────────────────────────────────────────
    # Promediar predicciones del ensemble
    # ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ENSEMBLE: promediando %d miembros", args.n)
    logger.info("=" * 60)

    y_pred_ensemble_test = np.mean(preds_test_members, axis=0)
    y_pred_ensemble_val  = np.mean(preds_val_members,  axis=0)

    y_pred_ensemble_test = apply_physical_constraints(
        y_pred_ensemble_test, cfg, y_train_max=y_train_max
    )

    # ──────────────────────────────────────────────────────────────
    # Corrección de sesgo (bias correction) usando validación
    # ──────────────────────────────────────────────────────────────
    # Sesgo = diferencia media entre predicción y realidad en validación.
    # Se calcula sobre el horizonte completo y se aplica al test.
    # Esto elimina la sobreestimación sistemática producida por MSE loss.
    bias_val = np.mean(y_pred_ensemble_val - y_val_real)  # escalar en metros
    y_pred_ensemble_test_bc = y_pred_ensemble_test - bias_val
    logger.info(
        "Corrección de sesgo (val): bias=%.4f m (%.2f cm) → aplicado al test",
        bias_val, bias_val * 100,
    )

    test_metrics    = compute_all_metrics(y_test_real, y_pred_ensemble_test)
    test_metrics_bc = compute_all_metrics(y_test_real, y_pred_ensemble_test_bc)
    val_metrics     = compute_all_metrics(y_val_real,  y_pred_ensemble_val)

    logger.info(
        "[ENSEMBLE TEST     ] MAE=%.3f m | RMSE=%.3f m | NSE=%.4f | KGE=%.4f",
        test_metrics["mae"], test_metrics["rmse"],
        test_metrics["nse"], test_metrics["kge"],
    )
    logger.info(
        "[ENSEMBLE TEST +BC ] MAE=%.3f m | RMSE=%.3f m | NSE=%.4f | KGE=%.4f  ← bias corregido",
        test_metrics_bc["mae"], test_metrics_bc["rmse"],
        test_metrics_bc["nse"], test_metrics_bc["kge"],
    )
    logger.info(
        "[ENSEMBLE VAL      ] MAE=%.3f m | RMSE=%.3f m | NSE=%.4f | KGE=%.4f",
        val_metrics["mae"], val_metrics["rmse"],
        val_metrics["nse"], val_metrics["kge"],
    )

    # Comparar contra baseline
    baseline_path = cfg["results"]["baseline_metrics"]
    if Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        best_nse = max(
            baseline["naive_baseline"]["nse"],
            baseline["seasonal_naive"]["nse"],
        )
        if test_metrics["nse"] > best_nse:
            logger.info(
                "OK: Ensemble NSE=%.4f supera baseline NSE=%.4f (+%.4f)",
                test_metrics["nse"], best_nse,
                test_metrics["nse"] - best_nse,
            )
        else:
            logger.warning(
                "BANDERA ROJA: Ensemble NSE=%.4f NO supera baseline NSE=%.4f.",
                test_metrics["nse"], best_nse,
            )

    detect_shadow_effect(
        pd.Series(y_test_real[:, -1]),
        pd.Series(y_pred_ensemble_test[:, -1]),
    )

    # ──────────────────────────────────────────────────────────────
    # Guardar resultados
    # ──────────────────────────────────────────────────────────────
    tracker = ExperimentTracker(
        experiment_name=args.name,
        model_type="transformer_ensemble_n{}".format(args.n),
    )
    tracker.log_config(cfg)
    tracker.log_metrics(test_metrics,    split="test")
    tracker.log_metrics(test_metrics_bc, split="test_bias_corrected")
    tracker.log_metrics(val_metrics,     split="val")
    # Guardar predicciones bias-corrected como las "oficiales" del experimento
    tracker.log_predictions(
        test_dates,
        y_test_real[:, -1],
        y_pred_ensemble_test_bc[:, -1],
        horizon=cfg["forecast_horizon"],
    )

    # Añadir metadata del ensemble al JSON final
    exp_dir = tracker.save()
    metrics_path = Path(exp_dir) / "metrics.json"
    with open(metrics_path) as f:
        meta = json.load(f)
    meta["n_members"]       = args.n
    meta["seeds"]           = seeds
    meta["use_residual"]    = use_residual
    meta["arch"] = {
        "d_model":       transformer_cfg.get("d_model", 64),
        "nhead":         transformer_cfg.get("nhead", 4),
        "num_layers":    transformer_cfg.get("num_layers", 2),
        "dim_feedforward": transformer_cfg.get("dim_feedforward", 128),
        "lookback":      cfg["lookback_window"],
    }
    with open(metrics_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info(
        "Ensemble '%s' completado -> %s", args.name, exp_dir
    )
    logger.info(
        "RESUMEN FINAL — Test (sin BC): MAE=%.3fcm | NSE=%.4f | KGE=%.4f",
        test_metrics["mae"]*100, test_metrics["nse"], test_metrics["kge"],
    )
    logger.info(
        "RESUMEN FINAL — Test (con BC): MAE=%.3fcm | NSE=%.4f | KGE=%.4f  ← GOLD",
        test_metrics_bc["mae"]*100, test_metrics_bc["nse"], test_metrics_bc["kge"],
    )
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble de Transformers para orinoco-forecast.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--name", required=True,
                        help="Nombre del experimento (ej. ensemble_transformer_v1)")
    parser.add_argument("--n", type=int, default=5,
                        help="Número de miembros del ensemble (default: 5)")
    # Overrides de arquitectura Transformer
    parser.add_argument("--d_model",        type=int,   default=None)
    parser.add_argument("--nhead",          type=int,   default=None)
    parser.add_argument("--num_layers",     type=int,   default=None)
    parser.add_argument("--dim_feedforward",type=int,   default=None)
    parser.add_argument("--dropout",        type=float, default=None)
    parser.add_argument("--lr",             type=float, default=None)
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--patience",       type=int,   default=None)
    parser.add_argument("--loss",           type=str,   default=None,
                        help="Función de pérdida: 'mse' (default) | 'huber'")
    parser.add_argument("--huber-delta",    type=float, default=None, dest="huber_delta",
                        help="Delta para Huber loss (default: 0.5)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TRANSFORMER ENSEMBLE: %s | N=%d", args.name, args.n)
    logger.info("=" * 60)

    cfg = load_config("config.yaml")
    run_ensemble(cfg, args)


if __name__ == "__main__":
    main()
