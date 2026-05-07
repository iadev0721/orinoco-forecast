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
from src.models.naive_baseline import run_baselines, SeasonalNaive
# Transformer imports are lazy (inside run_transformer) to avoid
# requiring torch when only running LSTM experiments.
from src.utils.gpu_config import configure_pytorch_gpu
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
    features_path = cfg.get("_features_path", "data/processed/dataset_orinoco_features.csv")
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
    """Entrena el modelo LSTM y registra el experimento.
    
    Si cfg['use_residual'] es True, el modelo aprende a predecir el DELTA
    (cambio en nivel) en lugar del nivel absoluto. La prediccion final se
    reconstruye como: nivel_pred = nivel_actual + delta_pred.
    Esto fuerza al modelo a competir directamente con el baseline Naive
    (que predice delta=0 siempre), usando los features climaticos solo
    para detectar CUANDO el rio va a cambiar.
    """
    # Gate R3 (usa lstm_model.check_baseline_gate)
    check_baseline_gate(cfg["results"]["baseline_metrics"])

    # R5: Configurar GPU (usa gpu_config)
    configure_tensorflow_gpu()

    features_path = cfg.get("_features_path", "data/processed/dataset_orinoco_features.csv")
    use_residual  = cfg.get("use_residual", False)

    # Construir tensores
    tensors    = build_tensors(cfg_override=cfg, features_path=features_path)
    X_train    = tensors["X_train"]
    y_train    = tensors["y_train"]
    X_val      = tensors["X_val"]
    y_val      = tensors["y_val"]
    X_test     = tensors["X_test"]
    y_test     = tensors["y_test"]
    scaler_y   = tensors["scaler_y"]
    test_dates = tensors["test_dates"]

    logger.info("Shape tensores -> X_train:%s  y_train:%s", X_train.shape, y_train.shape)

    # --- ARQUITECTURA RESIDUAL (predecir delta en lugar de nivel absoluto) ---
    # y_base_train[i] = nivel escalado en el ultimo paso del lookback de la muestra i
    # El LSTM aprende: y_delta = y_target - y_base
    # En inferencia: y_pred = y_base + y_delta_pred
    if use_residual:
        logger.info("MODO RESIDUAL ACTIVO: el modelo aprende delta = nivel_futuro - nivel_actual")
        # X_train shape: (n, lookback, n_features). La columna del target en X
        # no esta disponible directamente, pero si y_train[i,0] es el nivel en t+1,
        # el nivel en t (ultimo del lookback) esta en y_train[i-1, -1] aproximadamente.
        # Mas simple y correcto: usar el ultimo valor conocido de y antes del horizonte.
        # y_train[i] corresponde a data_y[i+lookback:i+lookback+horizon]
        # El nivel base es data_y[i+lookback-1] = y_train[i-1,-1] para i>0
        # Usamos el valor de y_train desplazado 1 sample atras, horizon=1 como base
        # Para evitar complejidad: extraemos el nivel base de los tensores X (ultima fila)
        # target_col_idx: buscar el indice del target en feature_cols (NO incluido en X)
        # En cambio, usamos y_train mismo: nivel_base[i] = nivel en t = y_train[i,0] - delta[i,0]
        # La forma mas directa: reconstruir y_base desde el scaler y los datos crudos.
        # y_train escalado: shape (n, horizon). Necesitamos y[i, j] - y[i, 0] + y_base[i]
        # donde y_base[i] es el nivel en t (ultimo dia del lookback de muestra i).
        # Implementacion: el nivel en t para la muestra i es el ultimo elemento de X[i]
        # que corresponde al target. Sin embargo X no incluye el target (excluido por get_feature_columns).
        # Solucion limpia: cargar la serie del target y alinearla.
        df_full = pd.read_csv(features_path, parse_dates=["fecha"]).set_index("fecha").sort_index()
        target_col = cfg["target_station"]
        lookback   = cfg["lookback_window"]
        horizon    = cfg["forecast_horizon"]
        train_end  = cfg["train_end"]
        val_end    = cfg["val_end"]

        from src.data.pipeline import split_data
        df_tr, df_va, df_te = split_data(df_full, train_end, val_end)

        # Nivel base escalado: el ultimo dia del lookback para cada muestra
        y_raw_tr = scaler_y.transform(df_tr[[target_col]]).ravel()
        y_raw_va = scaler_y.transform(df_va[[target_col]]).ravel()
        y_raw_te = scaler_y.transform(df_te[[target_col]]).ravel()

        def get_base_levels(y_raw, n_samples):
            """Nivel base (ultimo dia del lookback) para cada muestra."""
            # muestra i: lookback termina en indice i+lookback-1
            return np.array([y_raw[i + lookback - 1] for i in range(n_samples)], dtype=np.float32)

        base_train = get_base_levels(y_raw_tr, len(y_train))  # (n_train,)
        base_val   = get_base_levels(y_raw_va, len(y_val))
        base_test  = get_base_levels(y_raw_te, len(y_test))

        # Transformar targets a deltas: delta[i,j] = nivel[i,j] - base[i]
        y_train_orig = y_train.copy()
        y_val_orig   = y_val.copy()
        y_test_orig  = y_test.copy()

        y_train = y_train - base_train[:, np.newaxis]  # broadcast sobre horizon
        y_val   = y_val   - base_val[:, np.newaxis]
        y_test  = y_test  - base_test[:, np.newaxis]

        logger.info("Delta stats (train) - mean:%.4f std:%.4f min:%.4f max:%.4f",
                    y_train.mean(), y_train.std(), y_train.min(), y_train.max())

    # Construir modelo
    model = build_lstm_model(cfg, n_features=X_train.shape[2])

    # Entrenamiento
    history = train_lstm(
        model, X_train, y_train, X_val, y_val,
        config=cfg,
        experiment_name=tracker.name,
    )
    tracker.log_training_history(history)

    # Invertir escala de predicciones a metros reales
    def inv(arr_2d: np.ndarray) -> np.ndarray:
        result = np.zeros_like(arr_2d)
        for col in range(arr_2d.shape[1]):
            result[:, col] = scaler_y.inverse_transform(
                arr_2d[:, col].reshape(-1, 1)
            ).ravel()
        return result

    if use_residual:
        # Predicciones de delta (en espacio escalado)
        delta_pred_test = model.predict(X_test, verbose=0)  # (n, horizon)
        delta_pred_val  = model.predict(X_val,  verbose=0)

        # Reconstruir nivel: nivel_pred = nivel_base + delta_pred
        y_pred_scaled_test = base_test[:, np.newaxis] + delta_pred_test
        y_pred_scaled_val  = base_val[:, np.newaxis]  + delta_pred_val

        y_pred_real     = inv(y_pred_scaled_test)
        y_test_real     = inv(y_test_orig)
        y_val_pred_real = inv(y_pred_scaled_val)
        y_val_real      = inv(y_val_orig)
        logger.info("RESIDUAL: predicciones reconstruidas como nivel_base + delta_pred")
    else:
        y_pred_real     = inv(model.predict(X_test, verbose=0))
        y_test_real     = inv(y_test)
        y_val_pred_real = inv(model.predict(X_val, verbose=0))
        y_val_real      = inv(y_val)

    # R4: Restricciones fisicas sobre predicciones
    y_train_max = float(scaler_y.inverse_transform([[1.0]])[0, 0])
    y_pred_real = apply_physical_constraints(y_pred_real, cfg, y_train_max=y_train_max)

    # Metricas test
    test_metrics = compute_all_metrics(y_test_real, y_pred_real)
    tracker.log_metrics(test_metrics, split="test")

    # Metricas validacion
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
    """Entrena el Transformer y registra el experimento.

    Las importaciones de PyTorch/Transformer son lazy para no romper
    entornos donde solo está instalado TensorFlow (LSTM).
    Soporta el paradigma residual (use_residual: true en config.yaml):
        - El modelo aprende a predecir Δ = nivel_futuro - nivel_actual
        - Las predicciones se reconstruyen sumando el anchor (nivel actual)
        - Esto permite comparación directa y justa contra el LSTM gold standard
    """
    from src.models.transformer_model import (  # noqa: PLC0415
        build_transformer_model,
        predict_transformer,
        train_transformer,
    )
    check_baseline_gate(cfg["results"]["baseline_metrics"])
    configure_pytorch_gpu()

    transformer_cfg = cfg.get("transformer", {})
    features_path = transformer_cfg.get("features_path", "data/processed/dataset_orinoco_features.csv")

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
    anch_test  = tensors["anch_test"]   # (n,) niveles actuales normalizados — test
    anch_val   = tensors["anch_val"]    # (n,) niveles actuales normalizados — val

    logger.info("Shape tensores -> X_train:%s  y_train:%s", X_train.shape, y_train.shape)
    if use_residual:
        logger.info("Paradigma RESIDUAL activo: el modelo predice Δ. Reconstrucción posterior con anchor.")

    model = build_transformer_model(cfg, n_features=X_train.shape[2], horizon=cfg["forecast_horizon"])
    history = train_transformer(
        model, X_train, y_train, X_val, y_val,
        config=cfg, experiment_name=tracker.name,
    )
    tracker.log_training_history(history)

    def inv(arr_2d: np.ndarray) -> np.ndarray:
        """Invierte la normalización de un array 2D columna a columna."""
        result = np.zeros_like(arr_2d)
        for col in range(arr_2d.shape[1]):
            result[:, col] = scaler_y.inverse_transform(
                arr_2d[:, col].reshape(-1, 1)
            ).ravel()
        return result

    def reconstruct(raw_output: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Convierte la salida del modelo a metros reales.

        - use_residual=True : raw_output = delta_norm → sumar anchor antes de inv_transform
        - use_residual=False: raw_output = nivel_norm → inv_transform directo
        """
        if use_residual:
            norm_absolute = anchors.reshape(-1, 1) + raw_output  # (n, horizon)
            return inv(norm_absolute)
        return inv(raw_output)

    raw_test = predict_transformer(model, X_test, batch_size=transformer_cfg.get("batch_size", 32))
    raw_val  = predict_transformer(model, X_val,  batch_size=transformer_cfg.get("batch_size", 32))

    y_pred_real     = reconstruct(raw_test, anch_test)
    y_test_real     = inv(y_test + anch_test.reshape(-1, 1)) if use_residual else inv(y_test)
    y_val_pred_real = reconstruct(raw_val,  anch_val)
    y_val_real      = inv(y_val  + anch_val.reshape(-1,  1)) if use_residual else inv(y_val)

    y_train_max = float(scaler_y.inverse_transform([[1.0]])[0, 0])
    y_pred_real = apply_physical_constraints(y_pred_real, cfg, y_train_max=y_train_max)

    test_metrics = compute_all_metrics(y_test_real, y_pred_real)
    tracker.log_metrics(test_metrics, split="test")

    val_metrics = compute_all_metrics(y_val_real, y_val_pred_real)
    tracker.log_metrics(val_metrics, split="val")

    with open(cfg["results"]["baseline_metrics"]) as f:
        baseline = json.load(f)
    best_baseline_nse = max(
        baseline["naive_baseline"]["nse"],
        baseline["seasonal_naive"]["nse"],
    )
    if test_metrics["nse"] <= best_baseline_nse:
        logger.warning(
            "BANDERA ROJA: Transformer NSE=%.4f NO supera baseline NSE=%.4f.",
            test_metrics["nse"], best_baseline_nse,
        )
    else:
        logger.info(
            "OK: Transformer NSE=%.4f supera baseline NSE=%.4f (mejora: +%.4f)",
            test_metrics["nse"], best_baseline_nse,
            test_metrics["nse"] - best_baseline_nse,
        )

    detect_shadow_effect(
        pd.Series(y_test_real[:, -1]),
        pd.Series(y_pred_real[:, -1]),
    )

    tracker.log_predictions(
        test_dates,
        y_test_real[:, -1],
        y_pred_real[:, -1],
        horizon=cfg["forecast_horizon"],
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
    parser.add_argument("--features_path", type=str, default=None,
                        help="Override: ruta al CSV de features (default: dataset_orinoco_features.csv)")
    parser.add_argument("--train_end", type=str, default=None,
                        help="Override: fecha de fin del train set (YYYY-MM-DD)")
    parser.add_argument("--val_end",   type=str, default=None,
                        help="Override: fecha de fin del val set (YYYY-MM-DD)")
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
    if args.train_end is not None:
        cfg["train_end"] = args.train_end
        logger.info("Override: train_end = %s", args.train_end)
    if args.val_end is not None:
        cfg["val_end"] = args.val_end
        logger.info("Override: val_end = %s", args.val_end)
    # Guardar features_path en cfg para que run_lstm/run_naive lo usen
    cfg["_features_path"] = args.features_path or "data/processed/dataset_orinoco_features.csv"
    logger.info("Features path: %s", cfg["_features_path"])

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
