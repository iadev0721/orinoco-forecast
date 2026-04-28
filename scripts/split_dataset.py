"""
scripts/split_dataset.py
=========================
Script auxiliar para dividir físicamente el dataset de features en 
tres archivos CSV distintos (train, val, test) según las fechas 
establecidas en config.yaml.

Esto es útil para inspeccionar los datos manualmente o cargarlos 
en otros softwares.
"""
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.pipeline import split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_path = "config.yaml"
    features_path = "data/processed/dataset_orinoco_features.csv"
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    train_end = cfg["train_end"]
    val_end   = cfg["val_end"]
    
    logger.info("Cargando dataset base desde: %s", features_path)
    if not os.path.exists(features_path):
        logger.error("El archivo %s no existe.", features_path)
        logger.error("Asegúrate de haber ejecutado src/features/build_features.py primero.")
        return

    df = pd.read_csv(features_path, parse_dates=["fecha"]).set_index("fecha").sort_index()
    
    logger.info("Dividiendo datos usando train_end=%s y val_end=%s", train_end, val_end)
    df_train, df_val, df_test = split_data(df, train_end, val_end)
    
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    
    df_train.to_csv(train_path)
    df_val.to_csv(val_path)
    df_test.to_csv(test_path)
    
    logger.info("=" * 50)
    logger.info("Archivos CSV generados con éxito:")
    logger.info("  %s -> %d filas", train_path, len(df_train))
    logger.info("  %s -> %d filas", val_path, len(df_val))
    logger.info("  %s -> %d filas", test_path, len(df_test))
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
