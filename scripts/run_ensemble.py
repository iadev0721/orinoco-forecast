"""
scripts/run_ensemble.py
========================
Entrena N modelos LSTM con semillas distintas y promedia sus predicciones.

Guarda resultados en results/experiments/{name}/ con estructura estandar:
    config_used.yaml      <- config exacta usada en este run
    metrics.json          <- MAE, RMSE, NSE, KGE + metadatos del ensemble
    training_history.json <- curvas de loss promediadas entre miembros
    predictions_test.csv  <- predicciones vs real (horizonte completo, 7 pasos)

Uso:
    python scripts/run_ensemble.py --name ensemble_lb150_lags_xl --n 5 --lookback 150 --units 128 64
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # sin display (compatible con Colab y headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import build_tensors, load_config
from src.evaluation.experiment_tracker import ExperimentTracker
from src.evaluation.metrics import compute_all_metrics, detect_shadow_effect
from src.models.lstm_model import (
    apply_physical_constraints,
    build_lstm_model,
    check_baseline_gate,
    train_lstm,
)
from src.utils.gpu_config import configure_tensorflow_gpu
from src.utils.reproducibility import set_global_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Seeds para los N miembros del ensemble (diversidad controlada)
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1011, 2024, 314, 99, 7, 2048]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble de N modelos LSTM para Orinoco Forecast.")
    parser.add_argument("--name",     required=True, help="Nombre del experimento ensemble")
    parser.add_argument("--n",        type=int, default=5, help="Numero de modelos en el ensemble (default: 5)")
    parser.add_argument("--lookback", type=int, default=None, help="Override: lookback_window")
    parser.add_argument("--units",    type=int, nargs="+", default=None, help="Override: lstm.units")
    parser.add_argument("--features_path", type=str, default=None, help="Override: ruta al CSV de features")
    parser.add_argument("--train_end", type=str, default=None, help="Override: train_end")
    parser.add_argument("--val_end",   type=str, default=None, help="Override: val_end")
    args = parser.parse_args()

    assert args.n <= len(ENSEMBLE_SEEDS), f"Maximo {len(ENSEMBLE_SEEDS)} modelos soportados."

    logger.info("=" * 60)
    logger.info("ENSEMBLE: %s | N=%d miembros", args.name, args.n)
    logger.info("=" * 60)

    cfg = load_config("config.yaml")
    if args.lookback:  cfg["lookback_window"] = args.lookback
    if args.units:     cfg["lstm"]["units"]    = args.units
    if args.train_end: cfg["train_end"] = args.train_end
    if args.val_end:   cfg["val_end"]   = args.val_end
    cfg["_features_path"] = args.features_path or "data/processed/dataset_orinoco_features.csv"

    check_baseline_gate(cfg["results"]["baseline_metrics"])
    configure_tensorflow_gpu()

    features_path = cfg["_features_path"]
    use_residual  = cfg.get("use_residual", False)

    # ──────────────────────────────────────────────────────────────
    # Construir tensores UNA sola vez (mismo split para todos los miembros)
    # ──────────────────────────────────────────────────────────────
    tensors    = build_tensors(cfg_override=cfg, features_path=features_path)
    X_train    = tensors["X_train"]
    y_train    = tensors["y_train"]
    X_val      = tensors["X_val"]
    y_val      = tensors["y_val"]
    X_test     = tensors["X_test"]
    y_test     = tensors["y_test"]
    scaler_y   = tensors["scaler_y"]
    test_dates = tensors["test_dates"]
    anch_test  = tensors["anch_test"]   # nivel base normalizado (anchor para reconstruccion residual)
    anch_val   = tensors["anch_val"]

    # Con use_residual=True, build_tensors ya entrega y_train/y_val como deltas.
    # NO se vuelve a calcular el delta aqui (evitar doble sustraccion).
    y_train_delta = y_train
    y_val_delta   = y_val
    if use_residual:
        logger.info("MODO RESIDUAL ACTIVO: build_tensors entrega deltas normalizados.")

    def inv(arr_2d: np.ndarray) -> np.ndarray:
        """Desnormaliza un array 2D columna a columna."""
        result = np.zeros_like(arr_2d)
        for col in range(arr_2d.shape[1]):
            result[:, col] = scaler_y.inverse_transform(arr_2d[:, col].reshape(-1, 1)).ravel()
        return result

    def reconstruct_real(delta_or_abs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Reconstruye predicciones en metros reales.
        Si use_residual: nivel_pred = inv(anchor + delta)
        Si no: nivel_pred = inv(abs)
        """
        if use_residual:
            return inv(anchor[:, np.newaxis] + delta_or_abs)
        return inv(delta_or_abs)

    # Ground truth en metros reales (para evaluacion)
    y_test_real = reconstruct_real(y_test, anch_test)
    y_val_real  = reconstruct_real(y_val,  anch_val)

    # ──────────────────────────────────────────────────────────────
    # Entrenar N miembros
    # ──────────────────────────────────────────────────────────────
    all_preds_test = []  # (n_test, horizon) en metros reales
    all_preds_val  = []  # (n_val, horizon) en metros reales
    all_histories  = []  # history.history dict por miembro

    seeds = ENSEMBLE_SEEDS[:args.n]
    for i, seed in enumerate(seeds):
        logger.info("-" * 50)
        logger.info("Miembro %d/%d | seed=%d", i + 1, args.n, seed)
        logger.info("-" * 50)

        set_global_seeds(seed)
        model = build_lstm_model(cfg, n_features=X_train.shape[2])

        member_history = train_lstm(
            model, X_train, y_train_delta, X_val, y_val_delta,
            config=cfg,
            experiment_name=f"{args.name}_seed{seed}",
        )
        all_histories.append(member_history)

        pred_test_real = reconstruct_real(model.predict(X_test, verbose=0), anch_test)
        pred_val_real  = reconstruct_real(model.predict(X_val,  verbose=0), anch_val)

        all_preds_test.append(pred_test_real)
        all_preds_val.append(pred_val_real)

        mae_m   = np.abs(pred_test_real - y_test_real).mean()
        best_ep = int(np.argmin(member_history["val_loss"])) + 1
        logger.info(
            "Miembro %d: MAE_test=%.3fm | best_epoch=%d | best_val_loss=%.6f",
            i + 1, mae_m, best_ep, min(member_history["val_loss"]),
        )

    # ──────────────────────────────────────────────────────────────
    # Promediar predicciones del ensemble
    # ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ENSEMBLE: promediando %d miembros", args.n)

    y_pred_ensemble_test = np.mean(all_preds_test, axis=0)  # (n_test, horizon)
    y_pred_ensemble_val  = np.mean(all_preds_val,  axis=0)

    # Restricciones fisicas
    y_train_max = float(scaler_y.inverse_transform([[1.0]])[0, 0])
    y_pred_ensemble_test = apply_physical_constraints(y_pred_ensemble_test, cfg, y_train_max)

    # Metricas
    test_metrics = compute_all_metrics(y_test_real, y_pred_ensemble_test)
    val_metrics  = compute_all_metrics(y_val_real,  y_pred_ensemble_val)

    logger.info(
        "[ENSEMBLE TEST] MAE=%.3f m | RMSE=%.3f m | NSE=%.4f | KGE=%.4f",
        test_metrics["mae"], test_metrics["rmse"], test_metrics["nse"], test_metrics["kge"],
    )
    logger.info(
        "[ENSEMBLE VAL ] MAE=%.3f m | RMSE=%.3f m | NSE=%.4f | KGE=%.4f",
        val_metrics["mae"], val_metrics["rmse"], val_metrics["nse"], val_metrics["kge"],
    )

    detect_shadow_effect(
        pd.Series(y_test_real[:, -1]),
        pd.Series(y_pred_ensemble_test[:, -1]),
    )

    # Comparar con baseline
    with open(cfg["results"]["baseline_metrics"]) as f:
        baseline = json.load(f)
    best_baseline_nse = max(baseline["naive_baseline"]["nse"], baseline["seasonal_naive"]["nse"])
    if test_metrics["nse"] > best_baseline_nse:
        logger.info(
            "OK: Ensemble NSE=%.4f supera baseline NSE=%.4f (+%.4f)",
            test_metrics["nse"], best_baseline_nse, test_metrics["nse"] - best_baseline_nse,
        )
    else:
        logger.warning(
            "BANDERA ROJA: Ensemble NSE=%.4f NO supera baseline NSE=%.4f",
            test_metrics["nse"], best_baseline_nse,
        )

    # ──────────────────────────────────────────────────────────────
    # Promediar curvas de entrenamiento de todos los miembros
    # ──────────────────────────────────────────────────────────────
    # Truncar al miembro mas corto (early stopping puede parar en distintas epocas)
    min_epochs = min(len(h["loss"]) for h in all_histories)
    avg_history: dict = {
        key: [float(np.mean([h[key][ep] for h in all_histories]))
              for ep in range(min_epochs)]
        for key in all_histories[0].keys()
    }
    avg_history["_member_epochs"] = [len(h["loss"]) for h in all_histories]
    avg_history["_member_best_val_loss"] = [float(min(h["val_loss"])) for h in all_histories]

    # ──────────────────────────────────────────────────────────────
    # Persistir con ExperimentTracker (formato estandar)
    # ──────────────────────────────────────────────────────────────
    tracker = ExperimentTracker(
        experiment_name=args.name,
        model_type=f"lstm_ensemble_n{args.n}",
    )
    tracker.log_config(cfg)
    tracker.log_metrics(test_metrics, split="test")
    tracker.log_metrics(val_metrics,  split="val")
    tracker.log_training_history(avg_history)
    # Guardar horizonte completo (7 pasos) en predictions_test.csv
    tracker.log_predictions(
        test_dates,
        y_test_real,
        y_pred_ensemble_test,
        horizon=cfg["forecast_horizon"],
    )
    exp_dir = tracker.save()

    # Anotar metadata especifica del ensemble en metrics.json
    meta_path = Path(exp_dir) / "metrics.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["n_members"]    = args.n
    meta["seeds"]        = seeds
    meta["use_residual"] = use_residual
    meta["arch"] = {
        "lookback": cfg["lookback_window"],
        "units":    cfg["lstm"]["units"],
        "loss":     cfg["lstm"].get("loss", "mse"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ──────────────────────────────────────────────────────────────
    # Gráficos: predicciones y learning curve
    # ──────────────────────────────────────────────────────────────
    _save_plots(
        exp_dir=Path(exp_dir),
        test_dates=test_dates,
        y_true=y_test_real,
        y_pred=y_pred_ensemble_test,
        test_metrics=test_metrics,
        all_histories=all_histories,
        seeds=seeds,
        model_label="LSTM" if args.n == 1 else f"LSTM Ensemble ({args.n} miembros)",
        horizon=cfg["forecast_horizon"],
    )

    logger.info("=" * 60)
    logger.info("Ensemble '%s' completado -> %s", args.name, exp_dir)
    logger.info(
        "RESUMEN FINAL — Test: MAE=%.2fcm | NSE=%.4f | KGE=%.4f",
        test_metrics["mae"] * 100, test_metrics["nse"], test_metrics["kge"],
    )
    logger.info("=" * 60)


def _save_plots(
    exp_dir: Path,
    test_dates,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_metrics: dict,
    all_histories: list,
    seeds: list,
    model_label: str = "Ensemble",
    horizon: int = 7,
) -> None:
    """Genera y guarda predicciones.png y learning_curve.png en exp_dir."""

    # ── 1. Predicciones vs observado (horizonte t+H) ──────────
    y_true_plot = y_true[:, -1]
    y_pred_plot = y_pred[:, -1]
    err         = np.abs(y_true_plot - y_pred_plot)
    mae_cm      = test_metrics["mae"] * 100

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    ax.plot(test_dates, y_true_plot, color="#2196F3", lw=1.5,
            label="Observado (Palúa)", alpha=0.9)
    ax.plot(test_dates, y_pred_plot, color="#4CAF50", lw=1.5,
            label=f"{model_label} t+{horizon} (MAE={mae_cm:.1f} cm)", alpha=0.85)
    ax.fill_between(test_dates,
                    y_pred_plot - test_metrics["mae"],
                    y_pred_plot + test_metrics["mae"],
                    alpha=0.15, color="#4CAF50", label="±1 MAE")
    ax.set_title(f"Predicciones {model_label} vs Observado — Test Set",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Nivel del río Palúa (m)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.fill_between(test_dates, err, alpha=0.5, color="#FF5722")
    ax2.axhline(test_metrics["mae"], color="red", linestyle="--",
                linewidth=1.5, label=f"MAE={mae_cm:.1f} cm")
    ax2.set_title(f"Error Absoluto Diario (t+{horizon})", fontsize=12, fontweight="bold")
    ax2.set_ylabel("|Error| (m)")
    ax2.set_xlabel("Fecha")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    pred_png = exp_dir / "predicciones.png"
    plt.savefig(pred_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado: %s", pred_png)

    # ── 2. Learning curve ────────────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(12, 5))

    for hist, seed in zip(all_histories, seeds):
        ep = range(1, len(hist["val_loss"]) + 1)
        ax3.plot(ep, hist["val_loss"], lw=1, alpha=0.5, label=f"seed={seed}")

    min_len   = min(len(h["val_loss"]) for h in all_histories)
    avg_val   = np.mean([h["val_loss"][:min_len] for h in all_histories], axis=0)
    avg_train = np.mean([h["loss"][:min_len]     for h in all_histories], axis=0)
    ep_avg    = range(1, min_len + 1)
    ax3.plot(ep_avg, avg_val,   color="black", lw=2.5, linestyle="--",
             label="Promedio val_loss")
    ax3.plot(ep_avg, avg_train, color="gray",  lw=1.5, linestyle="-.",
             label="Promedio train_loss")

    ax3.set_title(f"Curva de Aprendizaje — {model_label}",
                  fontsize=13, fontweight="bold")
    ax3.set_xlabel("Época")
    ax3.set_ylabel("Loss (escala normalizada)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_yscale("log")

    plt.tight_layout()
    lc_png = exp_dir / "learning_curve.png"
    plt.savefig(lc_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Learning curve guardada: %s", lc_png)


if __name__ == "__main__":
    main()
