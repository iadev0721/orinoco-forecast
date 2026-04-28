"""
scripts/evaluate_experiment.py
================================
Evalua UN solo experimento de forma independiente.

Genera un reporte completo en results/figures/{nombre}/
sin necesidad de tener otros experimentos ni compartir codigo.

Cada miembro del equipo puede correr este script sobre su propio
entrenamiento, en su propia rama del repositorio.

Figuras generadas:
    predictions_vs_actual.png  -- serie temporal observado vs predicho
    scatter_by_regime.png      -- scatter coloreado por regimen hidrologico
    residuals_by_regime.png    -- boxplot de residuos por regimen
    extreme_events.png         -- zoom en crecidas maximas del test
    loss_curves.png            -- curvas de entrenamiento (solo modelos DL)
    metrics_table.csv          -- tabla completa de metricas por regimen

Uso:
    # Evaluar por nombre (busca en results/experiments/{nombre}/)
    python scripts/evaluate_experiment.py --name lstm_v1

    # Evaluar por ruta explicita (util en otras ramas o equipos)
    python scripts/evaluate_experiment.py --path /ruta/a/mi/experimento/

    # Solo imprimir metricas sin generar graficas
    python scripts/evaluate_experiment.py --name lstm_v1 --metrics-only
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import (
    compute_all_metrics, compute_metrics_by_regime, detect_shadow_effect
)
from src.evaluation.scenario_evaluator import evaluate_model_by_regime
from src.evaluation.visualization import (
    plot_extreme_events,
    plot_learning_curves,
    plot_predictions_vs_actual,
    plot_residuals_by_regime,
    plot_scatter_by_regime,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_experiment(exp_path: Path) -> dict:
    """Carga todos los artefactos de un experimento desde su directorio.

    Args:
        exp_path: Ruta al directorio del experimento.

    Returns:
        Dict con: metrics, config, history, predictions, name, model_type.

    Raises:
        FileNotFoundError: Si el directorio o metrics.json no existen.
    """
    if not exp_path.exists():
        raise FileNotFoundError(f"Directorio de experimento no encontrado: {exp_path}")

    metrics_file = exp_path / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"'metrics.json' no encontrado en {exp_path}.\n"
            "Asegurate de que el experimento se guardo correctamente con ExperimentTracker."
        )

    with open(metrics_file) as f:
        summary = json.load(f)

    result = {
        "name":       summary.get("experiment_name", exp_path.name),
        "model_type": summary.get("model_type", "unknown"),
        "metrics":    summary.get("metrics", {}),
        "training":   summary.get("training", {}),
        "config":     {},
        "history":    {},
        "predictions": None,
    }

    # Config
    config_file = exp_path / "config_used.yaml"
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            result["config"] = yaml.safe_load(f)

    # Historia de entrenamiento
    hist_file = exp_path / "training_history.json"
    if hist_file.exists():
        with open(hist_file) as f:
            result["history"] = json.load(f)

    # Predicciones
    pred_file = exp_path / "predictions_test.csv"
    if pred_file.exists():
        result["predictions"] = pd.read_csv(pred_file, parse_dates=["fecha"])

    return result


def print_metrics_report(exp: dict) -> None:
    """Imprime un reporte de metricas en consola."""
    name = exp["name"]
    print()
    print("=" * 65)
    print(f"  REPORTE DE EVALUACION: {name.upper()}")
    print("=" * 65)
    print(f"  Modelo     : {exp['model_type']}")

    cfg = exp.get("config", {})
    if cfg:
        print(f"  Target     : {cfg.get('target_station', '?')}")
        print(f"  Lookback   : {cfg.get('lookback_window', '?')} dias")
        print(f"  Horizonte  : {cfg.get('forecast_horizon', '?')} dias")
        print(f"  Train end  : {cfg.get('train_end', '?')}")
        print(f"  Val end    : {cfg.get('val_end', '?')}")

    if exp["training"]:
        t = exp["training"]
        print(f"  Epocas     : {t.get('epochs_trained', '?')} "
              f"(mejor: {t.get('best_epoch', '?')})")
        print(f"  Best val_loss: {t.get('best_val_loss', float('nan')):.6f}")

    print()
    print(f"  {'Metrica':<12} {'Train':>10} {'Val':>10} {'Test':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for metric in ["mae", "rmse", "nse", "kge"]:
        vals = {}
        for split in ["train", "val", "test"]:
            vals[split] = exp["metrics"].get(split, {}).get(metric, float("nan"))
        print(f"  {metric.upper():<12} "
              f"{vals['train']:>10.4f} {vals['val']:>10.4f} {vals['test']:>10.4f}")

    # Umbrales
    nse_test = exp["metrics"].get("test", {}).get("nse", float("nan"))
    kge_test = exp["metrics"].get("test", {}).get("kge", float("nan"))
    print()
    nse_ok = "OK" if nse_test >= 0.80 else ("ACEPTABLE" if nse_test >= 0.70 else "INSUFICIENTE")
    kge_ok = "OK" if kge_test >= 0.75 else "INSUFICIENTE"
    print(f"  NSE test = {nse_test:.4f} -> {nse_ok} (umbral: 0.80)")
    print(f"  KGE test = {kge_test:.4f} -> {kge_ok} (umbral: 0.75)")
    print("=" * 65)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalua un experimento individual y genera reporte completo.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", type=str,
                       help="Nombre del experimento (busca en results/experiments/{nombre}/)")
    group.add_argument("--path", type=str,
                       help="Ruta absoluta o relativa al directorio del experimento.")

    parser.add_argument("--metrics-only", action="store_true",
                        help="Solo imprime metricas, no genera graficas.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directorio de salida de figuras. Por defecto: results/figures/{nombre}/")
    args = parser.parse_args()

    # Resolver ruta
    if args.name:
        exp_path = Path("results/experiments") / args.name
    else:
        exp_path = Path(args.path)

    logger.info("Cargando experimento: %s", exp_path)
    exp = load_experiment(exp_path)

    # Reporte de metricas (siempre)
    print_metrics_report(exp)

    if args.metrics_only:
        logger.info("--metrics-only: Saliendo sin generar graficas.")
        return

    # Directorio de figuras
    out_dir = Path(args.out_dir) if args.out_dir else Path("results/figures") / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Figuras seran guardadas en: %s", out_dir)

    # ── Cargar predicciones ───────────────────────────────────
    preds_df = exp.get("predictions")
    has_preds = preds_df is not None and len(preds_df) > 0

    if not has_preds:
        logger.warning("No hay predictions_test.csv — algunas graficas seran omitidas.")

    # ── 1. Curvas de entrenamiento ────────────────────────────
    history = exp.get("history", {})
    if history.get("loss") and history.get("val_loss"):
        best_ep = int(np.argmin(history["val_loss"])) + 1
        plot_learning_curves(
            train_loss=history["loss"],
            val_loss=history["val_loss"],
            model_name=exp["name"],
            output_path=str(out_dir / "loss_curves.png"),
            early_stopping_epoch=best_ep,
        )

    if has_preds:
        # Detectar columnas y_true / y_pred
        true_col = "y_true" if "y_true" in preds_df.columns else "y_true_h7"
        pred_col = "y_pred" if "y_pred" in preds_df.columns else "y_pred_h7"

        if true_col not in preds_df.columns or pred_col not in preds_df.columns:
            logger.warning("Columnas y_true/y_pred no encontradas en predictions_test.csv.")
        else:
            dates  = pd.DatetimeIndex(preds_df["fecha"])
            y_true = preds_df[true_col].values
            y_pred = preds_df[pred_col].values

            # ── 2. Serie temporal ─────────────────────────────────
            plot_predictions_vs_actual(
                dates=dates,
                y_true=y_true,
                predictions_dict={exp["name"]: y_pred},
                output_path=str(out_dir / "predictions_vs_actual.png"),
            )

            # ── 3. Scatter por regimen ────────────────────────────
            plot_scatter_by_regime(
                y_true=y_true, y_pred=y_pred, dates=dates,
                model_name=exp["name"],
                output_path=str(out_dir / "scatter_by_regime.png"),
            )

            # ── 4. Residuos por regimen ───────────────────────────
            plot_residuals_by_regime(
                y_true=y_true, y_pred=y_pred, dates=dates,
                model_name=exp["name"],
                output_path=str(out_dir / "residuals_by_regime.png"),
            )

            # ── 5. Zoom en eventos extremos ───────────────────────
            plot_extreme_events(
                dates=dates, y_true=y_true, y_pred=y_pred,
                model_name=exp["name"],
                output_path=str(out_dir / "extreme_events.png"),
            )

            # ── 6. Tabla por regimen ──────────────────────────────
            df_regime = evaluate_model_by_regime(
                model_name=exp["name"],
                y_true=pd.Series(y_true),
                y_pred=pd.Series(y_pred),
                dates=dates,
            )
            table_path = out_dir / "metrics_table.csv"
            df_regime.to_csv(table_path)
            logger.info("Tabla de metricas por regimen: %s", table_path)
            print(df_regime.to_string())

            # ── 7. Shadow effect (bandera roja) ───────────────────
            shadow = detect_shadow_effect(pd.Series(y_true), pd.Series(y_pred))
            if shadow["shadow_effect_detected"]:
                print("\n  *** BANDERA ROJA: Shadow Effect detectado ***")
                print(f"  corr(pred, real)={shadow['corr_pred_real']:.4f} < "
                      f"corr(pred, lag-1)={shadow['corr_pred_lag1']:.4f}")
                print("  El modelo puede estar copiando el ultimo valor en lugar de predecir.")

    print(f"\nReporte completo guardado en: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
