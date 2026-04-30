"""
build_features.py
=================
Feature engineering sobre dataset_orinoco_base.csv.
Genera dataset_orinoco_features.csv listo para entrenamiento.

Pipeline:
  1. Cargar dataset base (31 cols: río + NASA + ENSO + Guri crudo)
  2. Imputar Guri pre-1992 con media estacional + flag guri_imputado
  3. Rolling precipitation: 7d, 14d, 30d, 60d, 90d para cada radar
  4. Lags hidrológicos de estaciones upstream (Ayacucho @12d, Caicara @7d, CBolivar @3d)
  5. Interacción ENSO × precipitación acumulada (feature no lineal)
  6. Variables cíclicas: estacionalidad_seno y estacionalidad_coseno (sin/cos día del año)
  7. Quiebre estructural: regimen_post2015
"""

import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_PATH  = "data/processed/dataset_orinoco_base.csv"
OUTPUT_PATH = "data/processed/dataset_orinoco_features.csv"

# RADARES se detecta automáticamente desde las columnas del dataset base
# para no omitir ningún radar por error de nombre
RADARES: list = []
ROLLING_WINDOWS = [7, 14, 30, 60, 90]   # 60/90d capturan memoria estacional subcuenca Amazonas

def impute_guri_pre1992(df: pd.DataFrame) -> pd.DataFrame:
    if "guri_nivel_m" not in df.columns:
        logger.info("Guri: columna 'guri_nivel_m' no encontrada -- omitiendo imputacion.")
        return df
    logger.info("Imputando Guri pre-1992 con media estacional...")
    n_nan_before = df["guri_nivel_m"].isna().sum()

    df["guri_imputado"] = df["guri_nivel_m"].isna().astype(int)

    guri_mean_doy = (
        df.loc[df["guri_imputado"] == 0, "guri_nivel_m"]
        .groupby(df.loc[df["guri_imputado"] == 0].index.day_of_year)
        .mean()
    )
    df["guri_nivel_m"] = df.apply(
        lambda row: guri_mean_doy.get(row.name.day_of_year, np.nan)
        if pd.isna(row["guri_nivel_m"]) else row["guri_nivel_m"],
        axis=1,
    )

    n_nan_after = df["guri_nivel_m"].isna().sum()
    logger.info(
        "  Guri: %d NaN → %d NaN | %d filas imputadas",
        n_nan_before, n_nan_after, df["guri_imputado"].sum()
    )
    
    # Omitido temporalmente del dataset final (evaluado el 2026-04-25 como poco útil en producción).
    # Se deja la lógica documentada en caso de ser requerida para la tesis o análisis de sensibilidad.
    df.drop(columns=["guri_imputado"], inplace=True)
    return df

def impute_ayacucho_regression(df: pd.DataFrame) -> pd.DataFrame:
    # Como estamos usando el dataset Legacy de SimpleML, Ayacucho no tiene NaNs.
    # Pero dejamos la función lista en caso de que volvamos al RAW.
    if "ayacucho" not in df.columns or "caicara" not in df.columns:
        return df

    n_nan = df["ayacucho"].isna().sum()
    if n_nan == 0:
        logger.info("Ayacucho: sin NaN, no se requiere imputación.")
        return df
    # Si hubiese NaNs se haría la imputación, pero con el dataset legacy no hay.
    return df

def add_rolling_precipitation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculando rolling precipitation (%s días) para radares...", ROLLING_WINDOWS)
    added = 0
    for radar in RADARES:
        col = f"{radar}_precipitacion_mm"
        if col not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            df[f"{col}_acum_{w}d"] = (
                df[col].rolling(window=w, min_periods=1).sum()
            )
            added += 1
    logger.info("  Rolling features añadidas: %d columnas", added)
    return df

def add_guri_delta(df: pd.DataFrame) -> pd.DataFrame:
    if "guri_nivel_m" not in df.columns:
        return df
    df["guri_delta_nivel"] = df["guri_nivel_m"].rolling(window=7, min_periods=1).mean().diff()
    df["guri_delta_nivel"].fillna(0, inplace=True)
    logger.info("  guri_delta_nivel añadida (proxy descarga Macagua).")
    return df

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    dia = df.index.day_of_year
    df["estacionalidad_seno"]   = np.sin(2 * np.pi * dia / 365.25)
    df["estacionalidad_coseno"] = np.cos(2 * np.pi * dia / 365.25)
    logger.info("  Estacionalidad seno/coseno añadida.")
    return df


def add_upstream_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica los tiempos de viaje hidráulico conocidos como features explícitos.

    Tiempos de propagación de la onda de crecida (estimados por correlación cruzada):
      - Ayacucho  → Palúa: ~12 días
      - Caicara   → Palúa: ~7  días
      - Ciu. Bolívar → Palúa: ~3  días

    Esto le da al LSTM información directa del agua que ya está
    "en el río" viajando hacia la estación objetivo.
    """
    added = []
    upstream_lags = [
        ("ayacucho",       [7, 12, 20]),   # lags corto/nominal/largo
        ("caicara",        [3,  7, 14]),
        ("ciudad_bolivar", [1,  3,  7]),
    ]
    for station, lags in upstream_lags:
        if station not in df.columns:
            logger.info("  Upstream lag: columna '%s' no encontrada — omitiendo.", station)
            continue
        for lag in lags:
            col_name = f"{station}_lag{lag}d"
            df[col_name] = df[station].shift(lag)
            # Rellenar NaN iniciales con el primer valor disponible
            df[col_name] = df[col_name].bfill()
            added.append(col_name)
    logger.info("  Upstream lag features añadidas: %d columnas -> %s", len(added), added)
    return df


def add_enso_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Interacción ENSO × precipitación acumulada 90d por subcuenca.

    Durante El Niño (oni > 0.5), la misma precipitación produce MENOS caudal
    en la cuenca. Esta feature no lineal captura esa modulación.
    """
    if "enso_oni" not in df.columns:
        logger.info("  ENSO interaction: 'enso_oni' no encontrado — omitiendo.")
        return df
    added = []
    for radar in RADARES:
        precip_col = f"{radar}_precipitacion_mm_acum_90d"
        if precip_col not in df.columns:
            continue
        col_name = f"enso_x_{radar}_90d"
        df[col_name] = df["enso_oni"] * df[precip_col]
        added.append(col_name)
    logger.info("  ENSO interaction features añadidas: %d columnas", len(added))
    return df

def verify_output(df: pd.DataFrame) -> None:
    logger.info("=== VERIFICACION DEL DATASET ===")
    logger.info("  Filas     : %d", len(df))
    logger.info("  Columnas  : %d", len(df.columns))
    logger.info("  Rango     : %s -> %s", df.index.min().date(), df.index.max().date())
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.warning("  Columnas con NaN: %s", nan_cols)
    else:
        logger.info("  NaN: 0 -- Dataset limpio")

if __name__ == "__main__":
    logger.info("Cargando dataset base: %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, parse_dates=["fecha"])
    df.set_index("fecha", inplace=True)

    # Auto-detectar radares desde columnas del dataset base (evita omisiones por nombre)
    RADARES.clear()
    RADARES.extend(sorted({
        col.replace("_precipitacion_mm", "")
        for col in df.columns if col.endswith("_precipitacion_mm")
    }))
    logger.info("Radares detectados: %s", RADARES)

    df = impute_guri_pre1992(df)
    df = impute_ayacucho_regression(df)
    df = add_rolling_precipitation(df)
    df = add_upstream_lags(df)           # <-- NUEVO: lags hidráulicos explícitos
    df = add_enso_interaction(df)        # <-- NUEVO: ENSO × precipitación
    df = add_cyclic_features(df)

    # Quiebre estructural 2015 (El Niño 2015-16)
    df["regimen_post2015"] = (df.index >= "2015-01-01").astype(int)
    n_post = df["regimen_post2015"].sum()
    logger.info("  regimen_post2015: %d dias pre-quiebre | %d dias post-quiebre (desde 2015-01-01)",
                len(df) - n_post, n_post)

    verify_output(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    logger.info("Dataset de features guardado: %s", OUTPUT_PATH)
