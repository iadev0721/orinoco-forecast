"""
scripts/compare_experiments.py
================================
Compara todos los experimentos guardados en results/experiments/
y genera graficas y tablas de evaluacion para la tesis.

Genera automaticamente:
    results/figures/comparison_metrics.png    -- barras MAE/RMSE/NSE/KGE
    results/figures/predictions_test.png      -- series temporales test
    results/figures/loss_curves.png           -- curvas de entrenamiento
    results/figures/error_by_regime.png       -- error por regimen hidrologico
    results/figures/scatter_plot.png          -- predicho vs observado
    results/figures/comparison_table.csv      -- tabla resumen exportable

Uso:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --experiments-dir results/experiments
    python scripts/compare_experiments.py --highlight lstm_lookback30
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")   # no GUI, renderiza a archivo
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.experiment_tracker import load_all_experiments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paleta de colores y estilo
# ──────────────────────────────────────────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]
REGIME_COLORS = {
    "aguas_bajas": "#E8A838",
    "ascenso":     "#55A868",
    "aguas_altas": "#4C72B0",
    "descenso":    "#C44E52",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def classify_regime(month: int) -> str:
    if month in [1, 2, 3, 4]:
        return "aguas_bajas"
    elif month in [5, 6, 7]:
        return "ascenso"
    elif month in [8, 9]:
        return "aguas_altas"
    return "descenso"


# ──────────────────────────────────────────────
# 1. Tabla resumen de metricas
# ──────────────────────────────────────────────
def build_metrics_table(experiments: List[dict]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        test_m = exp.get("metrics", {}).get("test", {})
        val_m  = exp.get("metrics", {}).get("val", {})
        train_info = exp.get("training", {})
        rows.append({
            "Experimento":    exp["experiment_name"],
            "Modelo":         exp.get("model_type", "?"),
            "MAE (m)":        round(test_m.get("mae", float("nan")), 3),
            "RMSE (m)":       round(test_m.get("rmse", float("nan")), 3),
            "NSE":            round(test_m.get("nse", float("nan")), 4),
            "KGE":            round(test_m.get("kge", float("nan")), 4),
            "NSE val":        round(val_m.get("nse", float("nan")), 4),
            "Epocas":         train_info.get("epochs_trained", "-"),
            "Mejor epoca":    train_info.get("best_epoch", "-"),
            "Timestamp":      exp.get("timestamp", "")[:16],
        })
    df = pd.DataFrame(rows).sort_values("NSE", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 2. Grafica de barras comparativas de metricas
# ──────────────────────────────────────────────
def plot_metrics_comparison(df_table: pd.DataFrame, highlight: Optional[str] = None) -> None:
    metrics  = ["MAE (m)", "RMSE (m)", "NSE", "KGE"]
    n_models = len(df_table)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(4, 0.5 * n_models + 2)))
    fig.suptitle("Comparacion de Modelos — Test Set", y=1.02, fontsize=15, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        colors = []
        for name in df_table["Experimento"]:
            if highlight and name == highlight:
                colors.append("#C44E52")
            else:
                colors.append("#4C72B0")

        values = df_table[metric].values
        bars   = ax.barh(df_table["Experimento"], values, color=colors, alpha=0.85, height=0.6)

        # Umbral NSE aceptable
        if metric == "NSE":
            ax.axvline(x=0.80, color="#E8A838", linestyle="--", linewidth=1.5, label="NSE=0.80")
            ax.axvline(x=0.90, color="#55A868", linestyle="--", linewidth=1.5, label="NSE=0.90")
            ax.legend(fontsize=9)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left", fontsize=9)

        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.invert_yaxis()
        ax.margins(x=0.15)

    plt.tight_layout()
    out = FIGURES_DIR / "comparison_metrics.png"
    plt.savefig(out)
    plt.close()
    logger.info("Figura guardada: %s", out)


# ──────────────────────────────────────────────
# 3. Series temporales: predicciones vs real
# ──────────────────────────────────────────────
def plot_predictions(experiments: List[dict], max_models: int = 5) -> None:
    exps_with_preds = []
    for exp in experiments:
        pred_file = Path(exp["_path"]) / "predictions_test.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file, parse_dates=["fecha"])
            exps_with_preds.append((exp["experiment_name"], df))

    if not exps_with_preds:
        logger.warning("Ningun experimento tiene predictions_test.csv — saltando grafico de series.")
        return

    # Usar la primera para obtener y_true (es la misma para todos)
    _, df_ref = exps_with_preds[0]
    # Detectar columna y_true (puede ser y_true o y_true_h7)
    y_true_col = "y_true" if "y_true" in df_ref.columns else "y_true_h7"

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df_ref["fecha"], df_ref[y_true_col], color="black", lw=1.5,
            label="Observado", zorder=5)

    for i, (name, df) in enumerate(exps_with_preds[:max_models]):
        pred_col = "y_pred" if "y_pred" in df.columns else "y_pred_h7"
        if pred_col not in df.columns:
            continue
        ax.plot(df["fecha"], df[pred_col], color=PALETTE[i % len(PALETTE)],
                lw=1.0, alpha=0.85, label=name)

    ax.set_title("Predicciones vs Observado — Test Set (horizonte t+7)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nivel del rio (m)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    out = FIGURES_DIR / "predictions_test.png"
    plt.savefig(out)
    plt.close()
    logger.info("Figura guardada: %s", out)


# ──────────────────────────────────────────────
# 4. Curvas de entrenamiento (loss)
# ──────────────────────────────────────────────
def plot_loss_curves(experiments: List[dict]) -> None:
    exps_with_history = []
    for exp in experiments:
        hist_file = Path(exp["_path"]) / "training_history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                history = json.load(f)
            if "loss" in history and "val_loss" in history:
                exps_with_history.append((exp["experiment_name"], history))

    if not exps_with_history:
        logger.info("Ningun experimento tiene historial de entrenamiento — saltando loss curves.")
        return

    n = len(exps_with_history)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Curvas de Entrenamiento (Loss por Epoca)", fontsize=14, fontweight="bold")

    for idx, (name, history) in enumerate(exps_with_history):
        ax = axes[idx // cols][idx % cols]
        epochs = range(1, len(history["loss"]) + 1)
        ax.plot(epochs, history["loss"],     color="#4C72B0", lw=1.5, label="Train Loss")
        ax.plot(epochs, history["val_loss"], color="#DD8452", lw=1.5, label="Val Loss", linestyle="--")

        # Marcar mejor epoca
        best_ep = int(np.argmin(history["val_loss"])) + 1
        ax.axvline(x=best_ep, color="#55A868", linestyle=":", lw=1.2,
                   label=f"Mejor epoca ({best_ep})")

        ax.set_title(name)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss (MSE)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Ocultar ejes sobrantes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "loss_curves.png"
    plt.savefig(out)
    plt.close()
    logger.info("Figura guardada: %s", out)


# ──────────────────────────────────────────────
# 5. Error por regimen hidrologico (boxplot)
# ──────────────────────────────────────────────
def plot_error_by_regime(experiments: List[dict]) -> None:
    exps_data = []
    for exp in experiments:
        pred_file = Path(exp["_path"]) / "predictions_test.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file, parse_dates=["fecha"])
            exps_data.append((exp["experiment_name"], df))

    if not exps_data:
        logger.info("Sin predictions_test.csv — saltando error por regimen.")
        return

    regimes = ["aguas_bajas", "ascenso", "aguas_altas", "descenso"]
    n = len(exps_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Error Absoluto por Regimen Hidrologico", fontsize=14, fontweight="bold")

    for ax, (name, df) in zip(axes, exps_data):
        true_col = "y_true" if "y_true" in df.columns else "y_true_h7"
        pred_col = "y_pred" if "y_pred" in df.columns else "y_pred_h7"
        if true_col not in df.columns or pred_col not in df.columns:
            ax.set_visible(False)
            continue

        df = df.copy()
        df["error_abs"] = np.abs(df[true_col] - df[pred_col])
        df["regime"]    = df["fecha"].dt.month.map(classify_regime)

        data_by_regime = [df.loc[df["regime"] == r, "error_abs"].dropna().values for r in regimes]
        bp = ax.boxplot(data_by_regime, patch_artist=True, medianprops={"color": "black", "lw": 2})

        for patch, regime in zip(bp["boxes"], regimes):
            patch.set_facecolor(REGIME_COLORS[regime])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(regimes) + 1))
        ax.set_xticklabels(["Bajas", "Ascenso", "Altas", "Descenso"], rotation=15)
        ax.set_title(name)
        ax.set_ylabel("Error Absoluto (m)")
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIGURES_DIR / "error_by_regime.png"
    plt.savefig(out)
    plt.close()
    logger.info("Figura guardada: %s", out)


# ──────────────────────────────────────────────
# 6. Scatter: predicho vs observado
# ──────────────────────────────────────────────
def plot_scatter(experiments: List[dict], max_models: int = 4) -> None:
    exps_data = []
    for exp in experiments:
        pred_file = Path(exp["_path"]) / "predictions_test.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file, parse_dates=["fecha"])
            exps_data.append((exp["experiment_name"], df))

    if not exps_data:
        return

    n = min(len(exps_data), max_models)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle("Predicho vs Observado — Test Set", fontsize=14, fontweight="bold")

    for idx, (name, df) in enumerate(exps_data[:n]):
        ax = axes[idx // cols][idx % cols]
        true_col = "y_true" if "y_true" in df.columns else "y_true_h7"
        pred_col = "y_pred" if "y_pred" in df.columns else "y_pred_h7"
        if true_col not in df.columns or pred_col not in df.columns:
            ax.set_visible(False)
            continue

        y_t = df[true_col].values
        y_p = df[pred_col].values
        ax.scatter(y_t, y_p, alpha=0.25, s=10, color=PALETTE[idx % len(PALETTE)])

        # Linea 1:1 perfecta
        lim = [min(y_t.min(), y_p.min()) - 0.5, max(y_t.max(), y_p.max()) + 0.5]
        ax.plot(lim, lim, "k--", lw=1.2, label="1:1")

        # R2
        corr = np.corrcoef(y_t, y_p)[0, 1]
        ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10)
        ax.set_title(name)
        ax.set_xlabel("Observado (m)")
        ax.set_ylabel("Predicho (m)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "scatter_plot.png"
    plt.savefig(out)
    plt.close()
    logger.info("Figura guardada: %s", out)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Compara todos los experimentos del proyecto.")
    parser.add_argument("--experiments-dir", default="results/experiments",
                        help="Directorio raiz de experimentos.")
    parser.add_argument("--highlight", default=None,
                        help="Nombre del experimento a resaltar en graficos.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("COMPARACION DE EXPERIMENTOS")
    logger.info("Directorio: %s", args.experiments_dir)
    logger.info("=" * 60)

    experiments = load_all_experiments(args.experiments_dir)
    if not experiments:
        logger.error("No se encontraron experimentos en '%s'.", args.experiments_dir)
        logger.error("Ejecuta primero al menos un entrenamiento.")
        return

    # Tabla resumen
    df_table = build_metrics_table(experiments)
    table_path = FIGURES_DIR / "comparison_table.csv"
    df_table.to_csv(table_path, index=False)
    logger.info("Tabla guardada: %s", table_path)
    print("\n" + "=" * 60)
    print("TABLA DE RESULTADOS (ordenada por NSE Test)")
    print("=" * 60)
    print(df_table.to_string(index=False))
    print("=" * 60 + "\n")

    # Graficas
    plot_metrics_comparison(df_table, highlight=args.highlight)
    plot_predictions(experiments)
    plot_loss_curves(experiments)
    plot_error_by_regime(experiments)
    plot_scatter(experiments)

    logger.info("=" * 60)
    logger.info("Todas las figuras guardadas en: %s", FIGURES_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
