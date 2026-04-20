"""
download_nasa_power.py — Descarga Multi-Radar de NASA POWER
============================================================
Estrategia: Multi-Source Exogenous Input.

Cada "Radar" es un punto geográfico que captura la lluvia de una sub-cuenca
específica, ANTES de que esa agua llegue a la estación de medición del río.
Esto le da al modelo información causal con ventaja de tiempo (lag).

    Radar Amazonas   → Avisa a Ayacucho  (Alto Orinoco, aguas arriba)
    Radar Apure/Meta → Avisa a Caicara   (Llanos occidentales, trib. Apure y Meta)
    Radar Ventuari   → Avisa a Caicara   (trib. Ventuari, sur del Orinoco Medio)
    Radar Caura      → Avisa a Ciudad Bolívar y aguas abajo (Escudo Guayanés)

Reglas aplicadas: R6 (código en inglés), R7 (parámetros en config), R9 (venv).
"""

import os
import time
import logging
from io import StringIO

import pandas as pd
import requests

# --- Logging (R6: nunca usar print) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURACIÓN: Radares hidrológicos
# Cada entrada define un punto de muestreo climático de NASA POWER.
# Justificación geográfica documentada para defensa de tesis.
# ==============================================================================
RADARES: dict = {
    # -----------------------------------------------------------------------
    # COBERTURA: Segmento Ayacucho (sur + cuenca alta)
    # -----------------------------------------------------------------------
    "amazonas": {
        "lat": 4.5,
        "lon": -67.6,
        "desc": (
            "Alto Orinoco, estado Amazonas. Aguas arriba de Puerto Ayacucho. "
            "Captura lluvia causal con lag ~12 días hasta Ayacucho."
        ),
    },
    # -----------------------------------------------------------------------
    # COBERTURA: Segmento Ayacucho → Caicara
    # -----------------------------------------------------------------------
    "apure_meta": {
        "lat": 7.5,
        "lon": -69.5,
        "desc": (
            "Llanos occidentales venezolanos. Cuenca de los ríos Apure y Meta "
            "(tributarios masivos desde Colombia). Avisa a Caicara con lag ~4-8 días."
        ),
    },
    "ventuari": {
        "lat": 4.0,
        "lon": -66.0,
        "desc": (
            "Río Ventuari — tributario derecho (sur) del Orinoco entre Ayacucho y Caicara. "
            "Cubre el punto ciego sur del Orinoco medio."
        ),
    },
    # -----------------------------------------------------------------------
    # COBERTURA: Segmento Caicara → Ciudad Bolívar
    # -----------------------------------------------------------------------
    "llanos_centrales": {
        "lat": 8.0,
        "lon": -66.5,
        "desc": (
            "Llanos centrales venezolanos (estado Guárico). Tributarios norte del Orinoco "
            "en el tramo Caicara–Ciudad Bolívar (ríos Pao, Zuata, Tucupido). "
            "Punto ciego entre los radares Apure/Meta y Caura."
        ),
    },
    "caura": {
        "lat": 6.0,
        "lon": -64.5,
        "desc": (
            "Escudo Guayanés occidental / Sierra de Maigualida. Cabecera del Río Caura, "
            "tributario guayanés que desemboca entre Caicara y Ciudad Bolívar. "
            "Avisa a Ciudad Bolívar con lag ~3-6 días."
        ),
    },
    # -----------------------------------------------------------------------
    # COBERTURA: Segmento Ciudad Bolívar → Palúa
    # -----------------------------------------------------------------------
    "caroni": {
        "lat": 5.5,
        "lon": -62.0,
        "desc": (
            "Gran Sabana / Escudo Guayanés oriental. Cabecera del Río Caroní, "
            "el tributario de mayor descarga del Orinoco (desemboca en Ciudad Guayana). "
            "Avisa a Ciudad Bolívar y aguas abajo. "
            "NOTA: El caudal real está parcialmente regulado por el Embalse del Guri "
            "(Corpoelec) — factor antropogénico no capturado por datos de lluvia."
        ),
    },
}

# Parámetros descargados de la API
NASA_PARAMS: str = "PRECTOTCORR,T2M,QV2M"
COMMUNITY: str = "AG"          # Agroclimatology — más preciso para lluvia/humedad
BLOCK_SIZE: int = 5            # Años por bloque para no saturar el servidor
API_PAUSE_S: float = 2.0       # Pausa entre bloques (segundos)

# Rango temporal (NASA POWER inicia en 1981)
START_YEAR: int = 1981
END_YEAR: int = 2024

# Rutas del repositorio (R7: no hardcodear fuera de esta sección)
RAW_RIVER_PATH: str = "data/raw/legacy_imputed/orinoco_dataset_legacy_simpleml.csv"
OUTPUT_PATH: str = "data/processed/dataset_orinoco_multivariado_final.csv"


# ==============================================================================
# FUNCIÓN: Descarga de un único radar
# ==============================================================================
def download_radar(name: str, lat: float, lon: float) -> pd.DataFrame:
    """Descarga datos climáticos diarios de NASA POWER para un punto geográfico.

    Args:
        name: Identificador del radar (usado como prefijo de columnas).
        lat:  Latitud del punto de muestreo.
        lon:  Longitud del punto de muestreo.

    Returns:
        DataFrame con índice 'fecha' y columnas prefijadas con el nombre del radar.
        Ej: 'amazonas_precipitacion_mm', 'amazonas_temp_media_c', etc.
    """
    logger.info("Iniciando descarga del Radar '%s' (lat=%.2f, lon=%.2f)", name, lat, lon)
    blocks = []

    for year in range(START_YEAR, END_YEAR + 1, BLOCK_SIZE):
        start_date = f"{year}0101"
        end_date = f"{min(year + BLOCK_SIZE - 1, END_YEAR)}1231"
        logger.info("  -> Bloque %s a %s ...", start_date, end_date)

        params = {
            "parameters": NASA_PARAMS,
            "community": COMMUNITY,
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "CSV",
        }

        response = requests.get(
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            params=params,
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(
                "Error en bloque %s-%s del radar '%s': HTTP %d",
                start_date, end_date, name, response.status_code
            )
            time.sleep(API_PAUSE_S)
            continue

        # Saltar el encabezado de metadatos de NASA (-BEGIN HEADER- ... -END HEADER-)
        lines = response.text.split("\n")
        skip = next(
            (i + 1 for i, line in enumerate(lines) if "-END HEADER-" in line),
            0
        )

        block_df = pd.read_csv(StringIO(response.text), skiprows=skip)

        # Construir columna 'fecha' desde YEAR + DOY (día del año)
        block_df["fecha"] = pd.to_datetime(
            block_df["YEAR"].astype(str) + block_df["DOY"].astype(str),
            format="%Y%j"
        )
        block_df.drop(columns=["YEAR", "DOY"], inplace=True)
        block_df.set_index("fecha", inplace=True)

        blocks.append(block_df)
        time.sleep(API_PAUSE_S)

    if not blocks:
        raise RuntimeError(f"No se pudo descargar ningún bloque para el radar '{name}'.")

    df = pd.concat(blocks)

    # Reemplazar valor centinela -999 (error de satélite) con NaN y rellenar
    df.replace(-999.0, pd.NA, inplace=True)
    df.ffill(inplace=True)

    # Renombrar columnas con prefijo del radar
    df.rename(columns={
        "PRECTOTCORR": f"{name}_precipitacion_mm",
        "T2M":         f"{name}_temp_media_c",
        "QV2M":        f"{name}_humedad_especifica",
    }, inplace=True)

    logger.info("Radar '%s' descargado: %d filas.", name, len(df))
    return df


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    # 1. Descargar todos los radares y unirlos
    radar_frames = []
    for radar_name, meta in RADARES.items():
        logger.info("=== RADAR: %s | %s ===", radar_name.upper(), meta["desc"])
        radar_df = download_radar(radar_name, meta["lat"], meta["lon"])
        radar_frames.append(radar_df)

    df_climate = pd.concat(radar_frames, axis=1)
    logger.info("Todos los radares unidos. Forma: %s", df_climate.shape)

    # 2. Cargar el dataset crudo del río
    logger.info("Cargando dataset crudo del rio: %s", RAW_RIVER_PATH)
    df_river = pd.read_csv(RAW_RIVER_PATH)
    df_river["fecha"] = pd.to_datetime(df_river["fecha"])
    df_river.set_index("fecha", inplace=True)

    # Interpolar NaNs de las estaciones del río (solo brechas pequeñas como 29-feb)
    # NOTA: Los NaNs estructurales de Ayacucho (700+) se tratan en Fase 1b (anti-leakage)
    for col in df_river.columns:
        df_river[col] = df_river[col].interpolate(method="time", limit=3)

    # 3. Fusión (inner join: solo fechas en ambos datasets)
    logger.info("Fusionando niveles del rio con datos climaticos de los 4 radares...")
    df_final = pd.merge(df_river, df_climate, left_index=True, right_index=True, how="inner")

    # 4. Guardar el dataset multivariado final
    df_final.to_csv(OUTPUT_PATH)

    logger.info("Dataset multivariado guardado en: %s", OUTPUT_PATH)
    logger.info("Resumen final:")
    logger.info("  Filas totales : %d", len(df_final))
    logger.info("  Columnas      : %s", list(df_final.columns))
