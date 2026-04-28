"""
src/evaluation/visualization.py
=================================
Graficas de publicacion para la tesis sobre prediccion hidrologica.

Estandar de salida: PNG 300 DPI (publicable en revista cientifica).
Directorio de salida: results/figures/{experiment_name}/

Graficas requeridas (docs/04_MANIFESTO_EVALUATION.md):
    1. Serie temporal test set: reales vs predichos
    2. Scatter plot real vs predicho por regimen hidrologico
    3. Curvas de aprendizaje (train_loss vs val_loss)
    4. Residuos por regimen hidrologico (boxplot)
    5. Zoom en eventos de crecida (picos maximos y minimos)
    6. Tabla de metricas globales y por regimen
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.evaluation.metrics import classify_regime

logger = logging.getLogger(__name__)

# ── Configuracion global de estilo ──────────────────────────────
FIGURE_DPI    = 300
FIGURE_FORMAT = "png"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
REGIME_COLORS = {
    "aguas_bajas": "#E8A838",
    "ascenso":     "#55A868",
    "aguas_altas": "#4C72B0",
    "descenso":    "#C44E52",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info("Figura guardada: %s", path)


# ── 1. Serie temporal: real vs predicho ─────────────────────────
def plot_predictions_vs_actual(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    output_path: str,
    regime_colors: bool = True,
) -> None:
    """Grafica de serie temporal: valores reales vs predichos.

    Args:
        dates: Indice temporal del test set.
        y_true: Valores reales.
        predictions_dict: Dict {nombre_modelo: predicciones}.
        output_path: Ruta de salida del PNG.
        regime_colors: Si True, colorea el fondo segun regimen hidrologico.
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    # Fondo por regimen
    if regime_colors and len(dates) > 0:
        prev_regime = classify_regime(pd.Timestamp(dates[0]))
        start_idx = 0
        for i, d in enumerate(dates[1:], 1):
            regime = classify_regime(pd.Timestamp(d))
            if regime != prev_regime or i == len(dates) - 1:
                ax.axvspan(dates[start_idx], dates[i - 1],
                           alpha=0.08, color=REGIME_COLORS[prev_regime], lw=0)
                start_idx = i
                prev_regime = regime

    ax.plot(dates, y_true, color="black", lw=1.8, label="Observado", zorder=5)
    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        ax.plot(dates, y_pred, color=PALETTE[i % len(PALETTE)],
                lw=1.2, alpha=0.85, label=name)

    # Leyenda de regimenes
    if regime_colors:
        patches = [mpatches.Patch(color=c, alpha=0.4, label=r.replace("_", " ").title())
                   for r, c in REGIME_COLORS.items()]
        legend1 = ax.legend(handles=patches, loc="upper left", fontsize=8,
                            title="Regimen", ncol=2)
        ax.add_artist(legend1)

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Predicciones vs Observado — Test Set (horizonte t+7)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nivel del rio (m)")
    ax.grid(alpha=0.25)
    _save(fig, Path(output_path))


# ── 2. Scatter coloreado por regimen ────────────────────────────
def plot_scatter_by_regime(
    y_true: pd.Series,
    y_pred: pd.Series,
    dates: pd.DatetimeIndex,
    model_name: str,
    output_path: str,
) -> None:
    """Scatter plot real vs predicho coloreado por regimen hidrologico.

    Args:
        y_true: Valores reales.
        y_pred: Predicciones del modelo.
        dates: Indice temporal.
        model_name: Nombre del modelo para el titulo.
        output_path: Ruta de salida del PNG.
    """
    y_true   = np.asarray(y_true).ravel()
    y_pred   = np.asarray(y_pred).ravel()
    regimes  = [classify_regime(pd.Timestamp(d)) for d in dates]

    fig, ax = plt.subplots(figsize=(6, 6))
    for regime, color in REGIME_COLORS.items():
        mask = np.array([r == regime for r in regimes])
        if mask.sum() == 0:
            continue
        ax.scatter(y_true[mask], y_pred[mask], color=color, alpha=0.35, s=12,
                   label=regime.replace("_", " ").title())

    # Linea 1:1
    lim = [min(y_true.min(), y_pred.min()) - 0.5,
           max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lim, lim, "k--", lw=1.5, label="1:1 (perfecto)")
    ax.set_xlim(lim); ax.set_ylim(lim)

    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    ax.text(0.05, 0.93, f"r = {r:.3f}", transform=ax.transAxes, fontsize=11)
    ax.set_title(f"Predicho vs Observado — {model_name}")
    ax.set_xlabel("Observado (m)")
    ax.set_ylabel("Predicho (m)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, Path(output_path))


# ── 3. Curvas de aprendizaje ─────────────────────────────────────
def plot_learning_curves(
    train_loss: List[float],
    val_loss: List[float],
    model_name: str,
    output_path: str,
    early_stopping_epoch: Optional[int] = None,
) -> None:
    """Curvas de aprendizaje (train vs validation loss por epoca).

    Args:
        train_loss: Loss de entrenamiento por epoca.
        val_loss: Loss de validacion por epoca.
        model_name: Nombre del modelo.
        output_path: Ruta de salida del PNG.
        early_stopping_epoch: Epoca donde se activo EarlyStopping.
    """
    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, train_loss, color="#4C72B0", lw=1.8, label="Train Loss")
    ax.plot(epochs, val_loss,   color="#DD8452", lw=1.8, linestyle="--", label="Val Loss")

    best_ep = int(np.argmin(val_loss)) + 1
    ax.axvline(x=best_ep, color="#55A868", linestyle=":", lw=1.5,
               label=f"Mejor epoca ({best_ep})")

    if early_stopping_epoch is not None and early_stopping_epoch != best_ep:
        ax.axvline(x=early_stopping_epoch, color="#C44E52", linestyle=":",
                   lw=1.5, label=f"Early Stop ({early_stopping_epoch})")

    ax.set_title(f"Curvas de Aprendizaje — {model_name}")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss (MSE)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, Path(output_path))


# ── 4. Residuos por regimen (boxplot) ───────────────────────────
def plot_residuals_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    model_name: str,
    output_path: str,
) -> None:
    """Boxplot de residuos (error) por regimen hidrologico.

    Args:
        y_true: Valores reales.
        y_pred: Predicciones.
        dates: Indice temporal.
        model_name: Nombre del modelo.
        output_path: Ruta de salida del PNG.
    """
    y_true   = np.asarray(y_true).ravel()
    y_pred   = np.asarray(y_pred).ravel()
    residuals = y_pred - y_true
    regimes  = [classify_regime(pd.Timestamp(d)) for d in dates]

    orden = ["aguas_bajas", "ascenso", "aguas_altas", "descenso"]
    labels = ["Bajas", "Ascenso", "Altas", "Descenso"]
    data   = [residuals[np.array([r == reg for r in regimes])] for reg in orden]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "lw": 2},
                    flierprops={"marker": ".", "alpha": 0.3, "markersize": 4})
    for patch, reg in zip(bp["boxes"], orden):
        patch.set_facecolor(REGIME_COLORS[reg])
        patch.set_alpha(0.75)

    ax.axhline(y=0, color="black", linestyle="--", lw=1.2)
    ax.set_xticks(range(1, len(orden) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(f"Residuos por Regimen Hidrologico — {model_name}")
    ax.set_ylabel("Residuo: Predicho - Observado (m)")
    ax.grid(alpha=0.25, axis="y")
    _save(fig, Path(output_path))


# ── 5. Zoom en eventos extremos ──────────────────────────────────
def plot_extreme_events(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str,
    n_events: int = 3,
    window_days: int = 30,
) -> None:
    """Zoom en los n_events picos mas altos del periodo de test.

    Util para evaluar si el modelo captura las crecidas extremas.

    Args:
        dates: Indice temporal.
        y_true: Valores reales.
        y_pred: Predicciones.
        model_name: Nombre del modelo.
        output_path: Ruta de salida del PNG.
        n_events: Numero de eventos a mostrar.
        window_days: Dias alrededor del pico a graficar.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Encontrar los n picos maximos (sin solapamiento de window_days)
    peak_indices = []
    remaining = list(np.argsort(y_true)[::-1])
    for idx in remaining:
        if not any(abs(idx - p) < window_days for p in peak_indices):
            peak_indices.append(idx)
        if len(peak_indices) == n_events:
            break

    fig, axes = plt.subplots(1, n_events, figsize=(6 * n_events, 4), sharey=False)
    if n_events == 1:
        axes = [axes]

    for ax, peak_idx in zip(axes, peak_indices):
        start = max(0, peak_idx - window_days // 2)
        end   = min(len(y_true), peak_idx + window_days // 2)
        d_slice = dates[start:end]
        ax.plot(d_slice, y_true[start:end], color="black", lw=2, label="Observado")
        ax.plot(d_slice, y_pred[start:end], color="#DD8452", lw=1.5,
                linestyle="--", label="Predicho")
        ax.axvline(dates[peak_idx], color="#C44E52", linestyle=":", lw=1.2, alpha=0.7)
        ax.set_title(f"Pico: {pd.Timestamp(dates[peak_idx]).strftime('%Y-%m-%d')}\n"
                     f"Real={y_true[peak_idx]:.2f}m | Pred={y_pred[peak_idx]:.2f}m")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Nivel (m)")

    fig.suptitle(f"Zoom Eventos de Crecida Extrema — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, Path(output_path))


# ── 6. Heatmap de metricas ───────────────────────────────────────
def plot_metrics_heatmap(
    comparison_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Heatmap de metricas por modelo y regimen hidrologico.

    Args:
        comparison_df: DataFrame con modelos como filas y metricas x regimen como columnas.
        output_path: Ruta de salida del PNG.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(comparison_df.columns) * 1.2),
                                    max(4, len(comparison_df) * 0.7)))
    data = comparison_df.select_dtypes(include=[float, int]).values
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(comparison_df.columns)))
    ax.set_xticklabels(comparison_df.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df.index, fontsize=10)

    for i in range(len(comparison_df)):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Valor de metrica")
    ax.set_title("Heatmap de Metricas — Todos los Modelos", fontweight="bold")
    plt.tight_layout()
    _save(fig, Path(output_path))
