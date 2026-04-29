"""
scripts/diagnose_experiment.py
================================
Diagnostica si un experimento LSTM presenta:
  - Underfitting          : el modelo no aprendió lo suficiente
  - Overfitting           : el modelo memorizó los datos de entrenamiento
  - Distribution Shift    : aprendió bien pero el período de test es diferente
  - Buena Generalización  : el modelo funciona bien en todos los splits

Uso:
    python scripts/diagnose_experiment.py --name lstm_xl
    python scripts/diagnose_experiment.py --name lstm_heavy --verbose
    python scripts/diagnose_experiment.py  # diagnostica todos los experimentos

Salida:
    Reporte impreso en consola + guardado en results/experiments/{name}/diagnosis.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# UMBRALES DE DIAGNÓSTICO (ajustados a hidrología, NSE-based)
# ──────────────────────────────────────────────────────────────
THRESHOLDS = {
    # Underfitting: el modelo no aprendió ni siquiera los datos de validación
    "underfit_nse_val_max": 0.85,

    # Overfitting: la pérdida de validación es más del doble que la de train
    # y además hay una brecha grande entre val y test
    "overfit_loss_ratio_min": 2.5,   # val_loss_final / train_loss_final
    "overfit_nse_gap_min": 0.020,    # NSE_val - NSE_test

    # Distribution Shift: val es bueno pero test cae significativamente
    "shift_nse_val_min": 0.90,       # val tiene que ser bueno para ser "shift"
    "shift_nse_gap_min": 0.015,      # brecha val-test suficiente para preocuparse

    # Buena Generalización: val y test son cercanos y ambos buenos
    "good_nse_test_min": 0.90,
    "good_nse_gap_max": 0.015,
    "good_loss_ratio_max": 2.0,

    # Información ceiling: el modelo alcanza el mejor epoch muy temprano
    "ceiling_best_epoch_ratio_max": 0.35,  # mejor epoch en el 35% inicial
}


def load_experiment(exp_dir: Path) -> dict:
    """Carga metrics.json y training_history.json de un experimento."""
    metrics_path = exp_dir / "metrics.json"
    history_path = exp_dir / "training_history.json"

    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    history = None
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    return {"metrics": metrics, "history": history}


def diagnose(data: dict, name: str, verbose: bool = False) -> dict:
    """
    Aplica las reglas de diagnóstico y devuelve un dict con el veredicto.

    Returns:
        {
            "experiment": str,
            "verdict": str,          # "UNDERFITTING" | "OVERFITTING" | "DISTRIBUTION_SHIFT" | "GOOD" | "CEILING"
            "confidence": str,       # "ALTA" | "MEDIA" | "BAJA"
            "evidence": [str],       # lista de hechos que soportan el veredicto
            "warnings": [str],       # señales secundarias
            "recommendation": str,
            "metrics_summary": dict,
        }
    """
    metrics   = data["metrics"]["metrics"]
    training  = data["metrics"]["training"]
    history   = data["history"]

    nse_val  = metrics.get("val",  {}).get("nse",  None)
    nse_test = metrics.get("test", {}).get("nse",  None)
    mae_val  = metrics.get("val",  {}).get("mae",  None)
    mae_test = metrics.get("test", {}).get("mae",  None)
    kge_test = metrics.get("test", {}).get("kge",  None)

    epochs_trained = training.get("epochs_trained", None)
    best_epoch     = training.get("best_epoch",     None)
    train_loss     = training.get("final_train_loss", None)
    val_loss_final = training.get("final_val_loss",   None)
    best_val_loss  = training.get("best_val_loss",    None)

    evidence    = []
    warnings    = []
    T           = THRESHOLDS

    # ── Métricas derivadas ──────────────────────────────────────
    nse_gap       = (nse_val - nse_test)     if (nse_val and nse_test) else None
    loss_ratio    = (val_loss_final / train_loss) if (val_loss_final and train_loss and train_loss > 0) else None
    epoch_ratio   = (best_epoch / epochs_trained) if (best_epoch and epochs_trained) else None
    mae_gap_pct   = ((mae_test - mae_val) / mae_val * 100) if (mae_val and mae_test and mae_val > 0) else None

    # ── Recopilar evidencia ─────────────────────────────────────

    # 1. Rendimiento absoluto
    if nse_val is not None:
        evidence.append(f"NSE Validación = {nse_val:.4f}")
    if nse_test is not None:
        evidence.append(f"NSE Test       = {nse_test:.4f}")
    if nse_gap is not None:
        evidence.append(f"Brecha val-test= {nse_gap:.4f} ({nse_gap*100:.2f} puntos NSE)")
    if mae_gap_pct is not None:
        evidence.append(f"MAE Test es un {mae_gap_pct:.1f}% mayor que MAE Val")

    # 2. Loss ratio
    if loss_ratio is not None:
        evidence.append(f"Ratio val_loss/train_loss (final) = {loss_ratio:.2f}")

    # 3. Epoch behavior
    if epoch_ratio is not None:
        evidence.append(f"Mejor época: {best_epoch}/{epochs_trained} (ratio={epoch_ratio:.2f})")

    # ── Aplicar reglas de diagnóstico ──────────────────────────

    # --- UNDERFITTING ---
    if nse_val is not None and nse_val < T["underfit_nse_val_max"]:
        verdict      = "UNDERFITTING"
        confidence   = "ALTA"
        recommendation = (
            "El modelo no tiene suficiente capacidad o no entrenó lo suficiente. "
            "Aumentar unidades (--units 128 64), reducir dropout, aumentar paciencia, "
            "o revisar el pipeline de features."
        )

    # --- OVERFITTING ---
    elif (
        loss_ratio is not None and loss_ratio > T["overfit_loss_ratio_min"]
        and nse_gap is not None and nse_gap > T["overfit_nse_gap_min"]
    ):
        verdict      = "OVERFITTING"
        confidence   = "ALTA" if loss_ratio > 4.0 else "MEDIA"
        recommendation = (
            "El modelo memorizó los datos de entrenamiento. "
            "Aumentar dropout (--dropout 0.4), reducir unidades, "
            "o agregar regularización L2."
        )

    # --- DISTRIBUTION SHIFT ---
    elif (
        nse_val is not None and nse_val >= T["shift_nse_val_min"]
        and nse_gap is not None and nse_gap > T["shift_nse_gap_min"]
        and (loss_ratio is None or loss_ratio < T["overfit_loss_ratio_min"])
    ):
        verdict      = "DISTRIBUTION_SHIFT"
        confidence   = "ALTA" if nse_gap > 0.025 else "MEDIA"
        recommendation = (
            "El modelo generaliza bien sobre el período de validación pero el conjunto "
            "de prueba tiene dinámicas distintas. Esto no es un problema de arquitectura: "
            "faltan variables físicas clave (ej. caudal de descarga de Guri) que capturen "
            "los cambios en el régimen del río entre 2018 y 2025."
        )

    # --- BUENA GENERALIZACIÓN ---
    elif (
        nse_test is not None and nse_test >= T["good_nse_test_min"]
        and nse_gap is not None and nse_gap <= T["good_nse_gap_max"]
    ):
        verdict      = "GOOD_GENERALIZATION"
        confidence   = "ALTA"
        recommendation = (
            "El modelo generaliza correctamente. Considera pasar a la Fase 4 (Transformer) "
            "o explorar nuevas variables de entrada para mejorar aún más."
        )

    # --- CASO INDETERMINADO ---
    else:
        verdict      = "INDETERMINATE"
        confidence   = "BAJA"
        recommendation = (
            "Los indicadores son mixtos. Revisar las curvas de pérdida manualmente "
            "y considerar más experimentos con distintos hiperparámetros."
        )

    # ── Advertencias secundarias ────────────────────────────────

    if epoch_ratio is not None and epoch_ratio < T["ceiling_best_epoch_ratio_max"]:
        warnings.append(
            f"⚠ INFORMATION CEILING: el modelo alcanzó su mejor época muy temprano "
            f"(época {best_epoch} de {epochs_trained}). Los features disponibles parecen "
            f"no contener más información predictiva. Agregar nuevas variables de entrada."
        )

    if kge_test is not None and kge_test < 0.85:
        warnings.append(
            f"⚠ KGE bajo ({kge_test:.4f}): el modelo falla en reproducir la variabilidad "
            f"o el volumen total del río, no solo el timing."
        )

    if loss_ratio is not None and 1.5 < loss_ratio <= T["overfit_loss_ratio_min"]:
        warnings.append(
            f"⚠ Leve sobreajuste en loss (ratio={loss_ratio:.2f}): señal temprana. "
            f"Monitorear si empeora con más épocas."
        )

    # ── Historial de val_loss ───────────────────────────────────
    if history and verbose:
        val_losses = history.get("val_loss", [])
        if val_losses:
            min_vl = min(val_losses)
            last_vl = val_losses[-1]
            if last_vl > min_vl * 1.5:
                warnings.append(
                    f"⚠ La val_loss final ({last_vl:.6f}) es {((last_vl/min_vl-1)*100):.1f}% "
                    f"mayor que el mínimo ({min_vl:.6f}). El modelo divergió tras la mejor época."
                )

    return {
        "experiment":   name,
        "verdict":      verdict,
        "confidence":   confidence,
        "evidence":     evidence,
        "warnings":     warnings,
        "recommendation": recommendation,
        "metrics_summary": {
            "nse_val":    nse_val,
            "nse_test":   nse_test,
            "nse_gap":    nse_gap,
            "mae_val":    mae_val,
            "mae_test":   mae_test,
            "kge_test":   kge_test,
            "loss_ratio": loss_ratio,
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "epoch_ratio": epoch_ratio,
        }
    }


VERDICT_LABELS = {
    "UNDERFITTING":       ("🔵 UNDERFITTING",       "El modelo no aprendió suficiente"),
    "OVERFITTING":        ("🔴 OVERFITTING",         "El modelo memorizó los datos"),
    "DISTRIBUTION_SHIFT": ("🟡 DISTRIBUTION SHIFT",  "El período de test tiene dinámicas distintas"),
    "GOOD_GENERALIZATION":("🟢 BUENA GENERALIZACIÓN","El modelo generaliza correctamente"),
    "INDETERMINATE":      ("⚪ INDETERMINADO",        "Señales mixtas, requiere más análisis"),
}


def print_report(result: dict) -> None:
    label, description = VERDICT_LABELS.get(result["verdict"], ("❓ DESCONOCIDO", ""))
    sep = "=" * 62

    logger.info(sep)
    logger.info(f" DIAGNÓSTICO: {result['experiment']}")
    logger.info(sep)
    logger.info(f" Veredicto   : {label}")
    logger.info(f" Descripción : {description}")
    logger.info(f" Confianza   : {result['confidence']}")
    logger.info("")
    logger.info(" Evidencia:")
    for e in result["evidence"]:
        logger.info(f"   • {e}")
    if result["warnings"]:
        logger.info("")
        logger.info(" Alertas secundarias:")
        for w in result["warnings"]:
            logger.info(f"   {w}")
    logger.info("")
    logger.info(" Recomendación:")
    # Wrap the recommendation at ~58 chars
    rec = result["recommendation"]
    words = rec.split()
    line = "   "
    for word in words:
        if len(line) + len(word) + 1 > 60:
            logger.info(line)
            line = "   " + word
        else:
            line += " " + word
    if line.strip():
        logger.info(line)
    logger.info(sep)
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico de underfitting/overfitting/distribution shift.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Nombre del experimento (ej. lstm_xl). Si se omite, diagnostica todos."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Mostrar análisis adicional de las curvas de pérdida."
    )
    args = parser.parse_args()

    experiments_dir = Path("results/experiments")
    if not experiments_dir.exists():
        logger.error("No se encontró el directorio results/experiments/")
        sys.exit(1)

    # Seleccionar experimentos a diagnosticar
    if args.name:
        targets = [experiments_dir / args.name]
    else:
        targets = sorted(
            [d for d in experiments_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name
        )

    all_results = []
    for exp_dir in targets:
        data = load_experiment(exp_dir)
        if data is None:
            logger.warning(f"Experimento '{exp_dir.name}' no tiene metrics.json. Saltando.")
            continue

        # Solo diagnosticar modelos con entrenamiento (no baseline naive)
        model_type = data["metrics"].get("model_type", "")
        if model_type == "naive":
            logger.info(f"'{exp_dir.name}' es un baseline naive — no aplica diagnóstico de fitting.")
            continue

        result = diagnose(data, exp_dir.name, verbose=args.verbose)
        print_report(result)
        all_results.append(result)

        # Guardar en el directorio del experimento
        out_path = exp_dir / "diagnosis.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Diagnóstico guardado: {out_path}\n")

    # Resumen si se diagnosticaron varios
    if len(all_results) > 1:
        logger.info("=" * 62)
        logger.info(" RESUMEN GLOBAL")
        logger.info("=" * 62)
        for r in all_results:
            label, _ = VERDICT_LABELS.get(r["verdict"], ("❓", ""))
            gap_str = f"gap={r['metrics_summary']['nse_gap']:.4f}" if r['metrics_summary']['nse_gap'] else ""
            logger.info(f"  {label:<30} {r['experiment']} ({gap_str})")
        logger.info("=" * 62)


if __name__ == "__main__":
    main()
