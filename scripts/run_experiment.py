"""
scripts/run_experiment.py
==========================
Lanzador general de experimentos. Cualquier miembro del equipo usa este
script para entrenar un modelo, evaluar y guardar resultados.

Los resultados se guardan en:
    results/experiments/{nombre}/
        metrics.json, config_used.yaml, predictions_test.csv, training_history.json

Uso:
    # Baseline
    python scripts/run_experiment.py --name baseline_naive --model naive

    # LSTM con config por defecto
    python scripts/run_experiment.py --name lstm_v1 --model lstm

    # LSTM con lookback diferente (override de config.yaml)
    python scripts/run_experiment.py --name lstm_lookback45 --model lstm --lookback 45

    # LSTM con unidades distintas
    python scripts/run_experiment.py --name lstm_128units --model lstm --units 128 64

    # Transformer
    python scripts/run_experiment.py --name transformer_v1 --model transformer

Comparar todos los experimentos despues:
    python scripts/compare_experiments.py
"""
import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import build_tensors, load_config, split_data
from src.evaluation.experiment_tracker import ExperimentTracker
from src.evaluation.metrics import compute_all_metrics, detect_shadow_effect
from src.models.lstm_model import (
    apply_physical_constraints,
    build_lstm_model,
    check_baseline_gate,
    train_lstm,
)
from src.models.naive_baseline import run_baselines, NaiveBaseline, SeasonalNaive
from src.utils.gpu_config import configure_tensorflow_gpu
from src.utils.reproducibility import log_environment_versions, set_global_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENTRENADORES POR TIPO DE MODELO
# ──────────────────────────────────────────────────────────────────

def run_naive(cfg: dict, tracker: ExperimentTracker) -> None:
    """Ejecuta los modelos Naive y los registra como un experimento."""
    features_path = "data/processed/joined_legacy_nasa.csv"
    df = pd.read_csv(features_path, parse_dates=["fecha"]).set_index("fecha").sort_index()
    _, _, df_test = split_data(df, cfg["train_end"], cfg["val_end"])

    target_col = cfg["target_station"]
    horizon    = cfg["forecast_horizon"]
    baseline_out = cfg["results"]["baseline_metrics"]

    results = run_baselines(
        df=df,
        df_test=df_test,
        target_col=target_col,
        horizon=horizon,
        output_path=baseline_out,
    )

    # Registrar metricas del Naive (el mejor de los dos como el "experimento")
    best = "seasonal_naive" if (
        results["seasonal_naive"]["nse"] > results["naive_baseline"]["nse"]
    ) else "naive_baseline"
    tracker.log_metrics(results[best], split="test")
    tracker.log_metrics(results["naive_baseline"], split="test_naive")
    tracker.log_metrics(results["seasonal_naive"], split="test_seasonal")

    # Guardar predicciones del SeasonalNaive para graficas
    sea = SeasonalNaive(horizon=horizon)
    naive = NaiveBaseline(horizon=horizon)
    dates_list, y_trues, y_preds = [], [], []
    max_t = len(df_test) - horizon
    for i in range(max_t):
        t = df_test.index[i]
        y_fut = df_test[target_col].iloc[i + 1: i + 1 + horizon].values
        if len(y_fut) < horizon:
            continue
        p = sea.predict(df, target_col, t)
        dates_list.append(t)
        y_trues.append(y_fut)
        y_preds.append(p)

    if dates_list:
        tracker.log_predictions(
            pd.DatetimeIndex(dates_list),
            np.array(y_trues),
            np.array(y_preds),
            horizon=horizon,
        )


def run_lstm(cfg: dict, tracker: ExperimentTracker) -> None:
    """Entrena el modelo LSTM y registra el experimento."""
    # Gate R3 (usa lstm_model.check_baseline_gate)
    check_baseline_gate(cfg["results"]["baseline_metrics"])

    # R5: Configurar GPU (usa gpu_config)
    configure_tensorflow_gpu()

    # Construir tensores — pasamos cfg ya modificado para que los overrides
    # (ej. --lookback 90) afecten también a los tensores de entrada
    tensors    = build_tensors(cfg_override=cfg)
    X_train    = tensors["X_train"]
    y_train    = tensors["y_train"]
    X_val      = tensors["X_val"]
    y_val      = tensors["y_val"]
    X_test     = tensors["X_test"]
    y_test     = tensors["y_test"]
    scaler_y   = tensors["scaler_y"]
    test_dates = tensors["test_dates"]
    use_diff   = tensors.get("use_diff", False)
    test_anchors_local = tensors.get("test_anchors_local", None)  # (n_test,)
    val_anchors_local  = tensors.get("val_anchors_local",  None)  # (n_val,)

    logger.info("Shape tensores -> X_train:%s  y_train:%s", X_train.shape, y_train.shape)

    # Construir modelo (usa lstm_model.build_lstm_model)
    model = build_lstm_model(cfg, n_features=X_train.shape[2])

    # Entrenamiento (usa lstm_model.train_lstm)
    history = train_lstm(
        model, X_train, y_train, X_val, y_val,
        config=cfg,
        experiment_name=tracker.name,
    )
    tracker.log_training_history(history)

    # Invertir escala (y reconstruir niveles absolutos si use_diff)
    def inv(arr_2d: np.ndarray, anchors_local: np.ndarray = None) -> np.ndarray:
        """Desnormaliza arr_2d. En modo use_diff reconstruye nivel = ancla + cumsum(deltas).

        anchors_local: array (n_samples,) con el nivel real absoluto delúltimo timestep
        conocido de cada ventana. Evita acumulación de errores fila a fila.
        """
        result = np.zeros_like(arr_2d)
        for col in range(arr_2d.shape[1]):
            result[:, col] = scaler_y.inverse_transform(
                arr_2d[:, col].reshape(-1, 1)
            ).ravel()
        if use_diff and anchors_local is not None:
            reconstructed = np.zeros_like(result)
            for i in range(len(result)):
                # nivel[t+1..t+H] = ancla + cumsum(Δ[t+1..t+H])
                reconstructed[i] = anchors_local[i] + np.cumsum(result[i])
            return reconstructed
        return result

    if use_diff:
        logger.info("Modo Δ: reconstruyendo niveles absolutos con anclas locales.")

    y_pred_real = inv(model.predict(X_test, verbose=0), anchors_local=test_anchors_local)
    y_test_real = inv(y_test,                           anchors_local=test_anchors_local)

    # R4: Restricciones físicas
    # En modo use_diff el scaler_y fue ajustado sobre deltas, NO sobre niveles;
    # su máximo (~0.6 m) no es el techo real del río. Solo aplicamos R4 en modo nivel.
    if not use_diff:
        y_train_max = float(scaler_y.inverse_transform([[1.0]])[0, 0])
        y_pred_real = apply_physical_constraints(y_pred_real, cfg, y_train_max=y_train_max)
    else:
        y_pred_real = np.maximum(y_pred_real, 0.0)  # mínimo físico: nivel >= 0

    # Metricas test (usa metrics.compute_all_metrics)
    test_metrics = compute_all_metrics(y_test_real, y_pred_real)
    tracker.log_metrics(test_metrics, split="test")

    # Métricas validación
    y_val_pred_real = inv(model.predict(X_val, verbose=0), anchors_local=val_anchors_local)
    y_val_real      = inv(y_val,                           anchors_local=val_anchors_local)
    val_metrics = compute_all_metrics(y_val_real, y_val_pred_real)
    tracker.log_metrics(val_metrics, split="val")

    # Bandera roja: LSTM vs baseline
    with open(cfg["results"]["baseline_metrics"]) as f:
        baseline = json.load(f)
    best_baseline_nse = max(
        baseline["naive_baseline"]["nse"],
        baseline["seasonal_naive"]["nse"],
    )
    if test_metrics["nse"] <= best_baseline_nse:
        logger.warning(
            "BANDERA ROJA: LSTM NSE=%.4f NO supera baseline NSE=%.4f.",
            test_metrics["nse"], best_baseline_nse,
        )
    else:
        logger.info(
            "OK: LSTM NSE=%.4f supera baseline NSE=%.4f (mejora: +%.4f)",
            test_metrics["nse"], best_baseline_nse,
            test_metrics["nse"] - best_baseline_nse,
        )

    # Shadow effect (bandera roja R8)
    detect_shadow_effect(
        pd.Series(y_test_real[:, -1]),
        pd.Series(y_pred_real[:, -1]),
    )

    # Guardar predicciones (horizonte t+7, ultimo paso del forecast)
    tracker.log_predictions(
        test_dates,
        y_test_real[:, -1],
        y_pred_real[:, -1],
        horizon=cfg["forecast_horizon"],
    )


def run_transformer(cfg: dict, tracker: ExperimentTracker) -> None:
    """Placeholder para el Transformer (PyTorch). Implementar en Fase 4."""
    raise NotImplementedError(
        "Transformer aun no implementado. Ver src/models/transformer_model.py.\n"
        "Implementar en Fase 4 de la tesis."
    )


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

RUNNERS = {
    "naive":       run_naive,
    "lstm":        run_lstm,
    "transformer": run_transformer,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lanzador general de experimentos Orinoco Forecast.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--name",   required=True,
                        help="Nombre unico del experimento (ej. lstm_lookback30)")
    parser.add_argument("--model",  required=True, choices=list(RUNNERS.keys()),
                        help="Tipo de modelo a entrenar: naive | lstm | transformer")
    # Overrides de config.yaml
    parser.add_argument("--lookback", type=int, default=None,
                        help="Override: lookback_window en config.yaml")
    parser.add_argument("--units",    type=int, nargs="+", default=None,
                        help="Override: lstm.units (ej. --units 128 64)")
    parser.add_argument("--dropout",  type=float, default=None,
                        help="Override: lstm.dropout")
    parser.add_argument("--lr",       type=float, default=None,
                        help="Override: lstm.learning_rate")
    parser.add_argument("--batch",    type=int, default=None,
                        help="Override: lstm.batch_size")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EXPERIMENTO: %s | MODELO: %s", args.name, args.model)
    logger.info("=" * 60)

    cfg = load_config("config.yaml")

    # Aplicar overrides
    if args.lookback is not None:
        cfg["lookback_window"] = args.lookback
        logger.info("Override: lookback_window = %d", args.lookback)
    if args.units is not None:
        cfg["lstm"]["units"] = args.units
        logger.info("Override: lstm.units = %s", args.units)
    if args.dropout is not None:
        cfg["lstm"]["dropout"] = args.dropout
        logger.info("Override: lstm.dropout = %.2f", args.dropout)
    if args.lr is not None:
        cfg["lstm"]["learning_rate"] = args.lr
        logger.info("Override: lstm.learning_rate = %g", args.lr)
    if args.batch is not None:
        cfg["lstm"]["batch_size"] = args.batch
        logger.info("Override: lstm.batch_size = %d", args.batch)

    # R2: Seeds
    set_global_seeds(cfg["seed"])
    log_environment_versions()

    # Tracker del experimento
    tracker = ExperimentTracker(experiment_name=args.name, model_type=args.model)
    tracker.log_config(cfg)

    # Ejecutar el modelo correspondiente
    RUNNERS[args.model](cfg, tracker)

    # Persistir todo
    exp_dir = tracker.save()

    logger.info("=" * 60)
    logger.info("Experimento '%s' completado -> %s", args.name, exp_dir)
    logger.info("Para ver comparaciones: python scripts/compare_experiments.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
