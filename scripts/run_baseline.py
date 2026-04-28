"""
scripts/run_baseline.py
========================
Ejecuta los modelos Naive sobre el test set y genera baseline_metrics.json.

REGLA R3: Este script DEBE ejecutarse antes de scripts/run_lstm.py.
El archivo 'results/metrics/baseline_metrics.json' es el gate de la Fase 3b.

Uso:
    python scripts/run_baseline.py
"""
import logging
import sys
from pathlib import Path

# Agregar el root del proyecto al path para imports relativos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd

from src.data.pipeline import load_config, split_data
from src.models.naive_baseline import run_baselines
from src.utils.reproducibility import set_global_seeds, log_environment_versions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("FASE 3a — BASELINE MODELS")
    logger.info("=" * 60)

    # R2: Fijar seeds
    cfg = load_config("config.yaml")
    set_global_seeds(cfg["seed"])
    log_environment_versions()

    # Cargar dataset de features
    features_path = "data/processed/dataset_orinoco_features.csv"
    logger.info("Cargando dataset: %s", features_path)
    df = pd.read_csv(features_path, parse_dates=["fecha"]).set_index("fecha").sort_index()

    # Split cronologico
    _, _, df_test = split_data(df, cfg["train_end"], cfg["val_end"])

    target_col: str = cfg["target_station"]
    horizon: int    = cfg["forecast_horizon"]
    output_path: str = cfg["results"]["baseline_metrics"]

    logger.info("Target    : %s", target_col)
    logger.info("Horizonte : %d dias", horizon)
    logger.info("Test set  : %s -> %s | %d filas",
                df_test.index.min().date(), df_test.index.max().date(), len(df_test))

    # Ejecutar baselines
    results = run_baselines(
        df=df,
        df_test=df_test,
        target_col=target_col,
        horizon=horizon,
        output_path=output_path,
    )

    # Resumen final
    logger.info("=" * 60)
    logger.info("GATE R3 DESBLOQUEADO -> %s", output_path)
    logger.info("El LSTM puede entrenarse ahora ejecutando: python scripts/run_lstm.py")
    logger.info("=" * 60)

    # Umbral de referencia segun config.yaml
    nse_naive  = results["naive_baseline"]["nse"]
    nse_season = results["seasonal_naive"]["nse"]
    nse_target = cfg["thresholds"]["nse_acceptable"]

    logger.info("Benchmark NSE aceptable: %.2f", nse_target)
    logger.info("Naive NSE       : %.4f  (%s)", nse_naive,
                "OK" if nse_naive > nse_target else "Por debajo del umbral")
    logger.info("SeasonalNaive NSE: %.4f  (%s)", nse_season,
                "OK" if nse_season > nse_target else "Por debajo del umbral (esperado)")
    logger.info("El LSTM debe superar AMBOS para ser aceptable.")


if __name__ == "__main__":
    main()
