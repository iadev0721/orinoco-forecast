"""
scripts/run_baseline_quick.py
=================================
Ejecuta las baselines (Naive y SeasonalNaive) sin cargar TensorFlow ni otras
dependencias pesadas. Útil para comprobar el flujo localmente sin consumir GPU/CPU.

Uso:
  python scripts/run_baseline_quick.py --output results/metrics/baseline_metrics.json

"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Asegurar raíz del repo en sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.pipeline import load_config, split_data  
from src.models.naive_baseline import run_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(output_path: Optional[str] = None) -> None:
    cfg = load_config()
    target = cfg.get("target_station", "palua")
    horizon = cfg.get("forecast_horizon", 7)

    features_path = Path("data/processed/dataset_orinoco_features.csv")
    if not features_path.exists():
        # Fallback a joined_legacy_nasa_cleaned si existe
        alt = Path("data/processed/joined_legacy_nasa_cleaned.csv")
        if alt.exists():
            features_path = alt
        else:
            logger.error("No se encuentra el CSV de features: %s", features_path)
            raise SystemExit(1)

    logger.info("Cargando features desde: %s", features_path)
    df = pd.read_csv(features_path, parse_dates=["fecha"]).set_index("fecha").sort_index()

    df_train, df_val, df_test = split_data(df, cfg["train_end"], cfg["val_end"])

    out = output_path or cfg.get("results", {}).get("baseline_metrics", "results/metrics/baseline_metrics.json")
    results = run_baselines(df, df_test, target, horizon, output_path=out)
    logger.info("Baselines completadas. Resultados guardados en: %s", out)


if __name__ == "__main__":
    main()
