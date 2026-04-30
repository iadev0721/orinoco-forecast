"""
scripts/clean_target_missing.py
=================================
Elimina filas donde el target (p.ej. `palua`) es NaN.

Uso recomendado (desde la raíz del repo):
  python scripts/clean_target_missing.py 

Opciones:
  --input   Ruta CSV de entrada (por defecto lee de config.yaml -> transformer.features_path)
  --output  Ruta CSV de salida (por defecto añade sufijo _cleaned.csv)
  --target  Nombre de la columna target (por defecto de config.yaml -> target_station)
  --dry-run Muestra info sin escribir archivo

Este script NO imputa; sólo elimina filas con NaN en el target para asegurar
que la pipeline de tensores reciba un dataset sin NaNs en la columna objetivo.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Asegurar que la raíz del repositorio esté en sys.path para permitir
# `import src` cuando se ejecuta el script directamente (python scripts/...).
# Al ejecutar un script, Python inserta el directorio del script en sys.path
# (p.ej. ./scripts), por lo que las importaciones desde la raíz pueden fallar.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.pipeline import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args(argv: Optional[List[str]] = None):
    cfg = load_config()
    default_input = cfg.get("transformer", {}).get("features_path", "data/processed/joined_legacy_nasa.csv")
    default_target = cfg.get("target_station", "palua")

    parser = argparse.ArgumentParser(description="Eliminar filas con NaN en el target (no imputar).")
    parser.add_argument("--input", "-i", default=default_input, help="CSV de entrada con columna 'fecha'.")
    parser.add_argument("--output", "-o", default=None, help="CSV de salida. Si no se da, se usa input_cleaned.csv")
    parser.add_argument("--target", "-t", default=default_target, help="Columna objetivo a verificar.")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar resumen, no escribir archivo.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_cleaned" + input_path.suffix)

    if not input_path.exists():
        logger.error("Input CSV no existe: %s", input_path)
        raise SystemExit(1)

    logger.info("Cargando: %s", input_path)
    df = pd.read_csv(input_path, parse_dates=["fecha"]).sort_values("fecha")

    # Garantizar que la columna 'fecha' sea datetime; si no, convertir y avisar.
    if not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", infer_datetime_format=True)  # type: ignore[arg-type]
        n_nats = int(df["fecha"].isna().sum())
        if n_nats:
            logger.warning("Se detectaron %d filas con 'fecha' no parseable (NaT) tras la conversión.", n_nats)

    target = args.target
    if target not in df.columns:
        logger.error("Target '%s' no está en el CSV. Columnas disponibles: %s", target, list(df.columns))
        raise SystemExit(1)

  
    missing_mask = df[target].isna()
    missing_count = int(missing_mask.sum())
    missing_dates_series = df.loc[missing_mask, "fecha"]
    parsed = pd.to_datetime(missing_dates_series, errors="coerce", infer_datetime_format=True)  # type: ignore[arg-type]
    missing_dates = parsed.dropna().apply(lambda ts: ts.strftime("%Y-%m-%d")).tolist()
    if len(missing_dates) < missing_count:
        logger.warning("Algunas fechas con target NaN no pudieron parsearse y se omiten del listado.")

    logger.info("Target: %s | filas totales: %d | faltantes en target: %d", target, len(df), missing_count)
    if missing_count:
        logger.info("Fechas con NaN en target (muestra hasta 20): %s", missing_dates[:20])
    else:
        logger.info("No se detectaron NaNs en la columna target.")

    if args.dry_run:
        logger.info("Dry-run activado: no se escribe archivo. Run again without --dry-run to save cleaned CSV.")
        return

    if missing_count == 0:
        logger.info("No hay cambios. Copiando archivo de entrada al destino: %s", output_path)
        df.to_csv(output_path, index=False)
        logger.info("Archivo escrito: %s", output_path)
        return

    # Eliminar filas con NaN en target
    df_clean = df.loc[~missing_mask].copy()
    df_clean = df_clean.sort_values("fecha")
    df_clean.to_csv(output_path, index=False)
    logger.info("Archivo limpio guardado en: %s (filas: %d) — %d filas removidas", output_path, len(df_clean), missing_count)


if __name__ == "__main__":
    main()
