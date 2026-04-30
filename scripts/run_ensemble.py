"""
scripts/run_ensemble.py
========================
Entrena N modelos LSTM con semillas distintas y promedia sus predicciones.

Uso:
    python scripts/run_ensemble.py --name ensemble_residual_lb90 --n 5 --lookback 90

Guarda resultados en:
    results/experiments/{name}/
        metrics.json, predictions_test.csv, predictions_val.csv
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import build_tensors, load_config, split_data
from src.evaluation.experiment_tracker import ExperimentTracker
from src.evaluation.metrics import compute_all_metrics
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

    # Construir tensores UNA sola vez (mismo split para todos los miembros)
    tensors    = build_tensors(cfg_override=cfg, features_path=features_path)
    X_train    = tensors["X_train"]
    y_train    = tensors["y_train"]
    X_val      = tensors["X_val"]
    y_val      = tensors["y_val"]
    X_test     = tensors["X_test"]
    y_test     = tensors["y_test"]
    scaler_y   = tensors["scaler_y"]
    test_dates = tensors["test_dates"]

    # --- Modo residual: transformar targets a deltas ---
    if use_residual:
        logger.info("MODO RESIDUAL ACTIVO")
        df_full    = pd.read_csv(features_path, parse_dates=["fecha"]).set_index("fecha").sort_index()
        target_col = cfg["target_station"]
        lookback   = cfg["lookback_window"]
        horizon    = cfg["forecast_horizon"]

        df_tr, df_va, df_te = split_data(df_full, cfg["train_end"], cfg["val_end"])

        y_raw_tr = scaler_y.transform(df_tr[[target_col]]).ravel()
        y_raw_va = scaler_y.transform(df_va[[target_col]]).ravel()
        y_raw_te = scaler_y.transform(df_te[[target_col]]).ravel()

        def get_base_levels(y_raw, n_samples):
            return np.array([y_raw[i + lookback - 1] for i in range(n_samples)], dtype=np.float32)

        base_train = get_base_levels(y_raw_tr, len(y_train))
        base_val   = get_base_levels(y_raw_va, len(y_val))
        base_test  = get_base_levels(y_raw_te, len(y_test))

        y_train_orig = y_train.copy()
        y_val_orig   = y_val.copy()
        y_test_orig  = y_test.copy()

        y_train_delta = y_train - base_train[:, np.newaxis]
        y_val_delta   = y_val   - base_val[:, np.newaxis]
    else:
        y_train_delta = y_train
        y_val_delta   = y_val

    def inv(arr_2d):
        result = np.zeros_like(arr_2d)
        for col in range(arr_2d.shape[1]):
            result[:, col] = scaler_y.inverse_transform(arr_2d[:, col].reshape(-1, 1)).ravel()
        return result

    # --- Entrenar N miembros ---
    all_preds_test = []   # lista de (n_test, horizon) arrays en metros reales
    all_preds_val  = []

    seeds = ENSEMBLE_SEEDS[:args.n]
    for i, seed in enumerate(seeds):
        logger.info("-" * 50)
        logger.info("Miembro %d/%d | seed=%d", i + 1, args.n, seed)
        logger.info("-" * 50)

        set_global_seeds(seed)
        model = build_lstm_model(cfg, n_features=X_train.shape[2])

        train_lstm(
            model, X_train, y_train_delta, X_val, y_val_delta,
            config=cfg,
            experiment_name=f"{args.name}_seed{seed}",
        )

        if use_residual:
            delta_test = model.predict(X_test, verbose=0)
            delta_val  = model.predict(X_val,  verbose=0)
            pred_test_scaled = base_test[:, np.newaxis] + delta_test
            pred_val_scaled  = base_val[:, np.newaxis]  + delta_val
            pred_test_real = inv(pred_test_scaled)
            pred_val_real  = inv(pred_val_scaled)
        else:
            pred_test_real = inv(model.predict(X_test, verbose=0))
            pred_val_real  = inv(model.predict(X_val,  verbose=0))

        all_preds_test.append(pred_test_real)
        all_preds_val.append(pred_val_real)
        logger.info("Miembro %d: MAE_test=%.3fm", i + 1,
                    np.abs(pred_test_real - inv(y_test_orig if use_residual else y_test)).mean())

    # --- Promediar predicciones del ensemble ---
    logger.info("=" * 60)
    logger.info("ENSEMBLE: promediando %d miembros", args.n)

    y_pred_ensemble_test = np.mean(all_preds_test, axis=0)  # (n_test, horizon)
    y_pred_ensemble_val  = np.mean(all_preds_val,  axis=0)

    y_test_real = inv(y_test_orig if use_residual else y_test)
    y_val_real  = inv(y_val_orig  if use_residual else y_val)

    # Restricciones fisicas
    y_train_max = float(scaler_y.inverse_transform([[1.0]])[0, 0])
    y_pred_ensemble_test = apply_physical_constraints(y_pred_ensemble_test, cfg, y_train_max)

    # Metricas
    test_metrics = compute_all_metrics(y_test_real, y_pred_ensemble_test)
    val_metrics  = compute_all_metrics(y_val_real,  y_pred_ensemble_val)

    logger.info("[ENSEMBLE TEST] MAE=%.3f m | RMSE=%.3f m | NSE=%.4f | KGE=%.4f",
                test_metrics["mae"], test_metrics["rmse"], test_metrics["nse"], test_metrics["kge"])
    logger.info("[ENSEMBLE VAL ] MAE=%.3f m | RMSE=%.3f m | NSE=%.4f | KGE=%.4f",
                val_metrics["mae"], val_metrics["rmse"], val_metrics["nse"], val_metrics["kge"])

    # Comparar con baseline
    with open(cfg["results"]["baseline_metrics"]) as f:
        baseline = json.load(f)
    best_baseline_nse = max(baseline["naive_baseline"]["nse"], baseline["seasonal_naive"]["nse"])
    if test_metrics["nse"] > best_baseline_nse:
        logger.info("OK: Ensemble NSE=%.4f supera baseline NSE=%.4f (+%.4f)",
                    test_metrics["nse"], best_baseline_nse,
                    test_metrics["nse"] - best_baseline_nse)
    else:
        logger.warning("BANDERA ROJA: Ensemble NSE=%.4f NO supera baseline NSE=%.4f",
                       test_metrics["nse"], best_baseline_nse)

    # Guardar resultados
    out_dir = Path(f"results/experiments/{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "experiment_name": args.name,
        "model_type": f"lstm_ensemble_n{args.n}",
        "n_members": args.n,
        "seeds": seeds,
        "metrics": {"test": test_metrics, "val": val_metrics},
        "config": {"lookback": cfg["lookback_window"], "use_residual": use_residual},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Guardar predicciones (ultimo paso del horizonte)
    preds_df = pd.DataFrame({
        "fecha":  test_dates,
        "y_true": y_test_real[:, -1],
        "y_pred": y_pred_ensemble_test[:, -1],
    })
    preds_df.to_csv(out_dir / "predictions_test.csv", index=False)

    logger.info("=" * 60)
    logger.info("Ensemble '%s' completado -> %s", args.name, out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
