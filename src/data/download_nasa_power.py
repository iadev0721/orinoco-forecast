"""
download_nasa_power.py — Descarga Multi-Radar NASA POWER + ENSO ONI (NOAA)
===========================================================================
Estrategia: Multi-Source Exogenous Input — Variables Justificadas.

VARIABLES INCLUIDAS Y SU JUSTIFICACIÓN:
------------------------------------------------------------------------
Ecuación del balance hídrico que justifica la selección de variables:
    Caudal = Precipitación − Evapotranspiración − ΔAlmacenamiento_suelo

1. Niveles del río: ayacucho, caicara, ciudad_bolivar, palua
   → Señal autorregresiva. El nivel pasado es el mejor predictor del futuro.
   → Los lags empíricos (12, 4, 0 días) codifican la propagación de la onda de crecida.
   → TARGET CONFIGURADO: palua (ver config.yaml → target_station)

2. {radar}_precipitacion_mm ×6  [término P de la ecuación]
   → Causa física directa. Cada radar cubre una sub-cuenca distinta con lag causal.

3. {radar}_temp_media_c ×6
   → Incluida por solicitud del tutor académico.
   → Físicamente regula la evapotranspiración y captura señales de ENSO regional.

4. amazonas_humedad_especifica  [SOLO radar Amazonas]
   → Pre-indicador atmosférico upstream. Alta humedad → lluvia inminente en 1-3 días.
   → ¿Por qué no en todos los radares? NASA calcula EVPTRNS usando la fórmula
     Penman-Monteith, que ya usa QV2M implícitamente: ET₀ = f(T, QV2M, radiación, viento).
     Añadir QV2M a los radares 2-6 duplicaría la señal generando colinealidad garantizada.
   → Excepción Amazonas: Aquí QV2M no solo mide humedad local, sino que captura el estado
     de la masa de aire (transporte sinóptico) que ingresa al sistema desde la selva.
     Es información física anticipada y no una simple redundancia local.

5. {radar}_humedad_suelo ×6  [término ΔAlmacenamiento_suelo de la ecuación]
   → GWETROOT: fracción de saturación del suelo (0=seco, 1=saturado).
   → Determina qué porción de la lluvia se convierte en escorrentía vs. infiltración.

6. {radar}_evapotranspiracion_mm ×6  [término ET de la ecuación]
   → EVPTRNS: evapotranspiración diaria real (mm/día) calculada por NASA via Penman-Monteith.
   → Completa la ecuación hídrica. Sin ella, el modelo no sabe cuánta agua se pierde por
     evaporación antes de llegar al río (puede ser el 60-80% de la lluvia en sequía).

7. enso_oni
   → Factor interanual dominante. Años Niña = inundaciones, Niño = sequías.

TOTAL DE COLUMNAS RESULTANTES: 31
  4 (río) + 6 (precip) + 6 (temp) + 1 (hum. espec.) + 6 (hum. suelo) + 6 (ET) + 1 (ENSO) + 1 (Guri)

SALIDA: data/processed/dataset_orinoco_base.csv  (datos crudos, sin feature engineering)
FEATURE ENGINEERING: ver src/features/build_features.py
------------------------------------------------------------------------

Reglas aplicadas: R6 (código inglés), R7 (parámetros centralizados), R9 (venv).
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
# ==============================================================================
# Parámetros diferenciados por radar:
# - Todos descargan PRECTOTCORR (precipitación)
# - Solo "amazonas" descarga además QV2M (humedad específica)
RADARES: dict = {
    # COBERTURA: Upstream Ayacucho — señal causal con mayor lead time
    "amazonas": {
        "lat": 4.5,
        "lon": -67.6,
        # Precipitación + temperatura + humedad atmosférica + humedad del suelo
        "params": "PRECTOTCORR,T2M,QV2M,GWETROOT,EVPTRNS",
        "desc": "Alto Orinoco upstream de Ayacucho. Lag ~12d.",
    },
    # COBERTURA: Ayacucho → Caicara (norte — Llanos occidentales)
    "apure_meta": {
        "lat": 7.5,
        "lon": -69.5,
        "params": "PRECTOTCORR,T2M,GWETROOT,EVPTRNS",
        "desc": "Llanos occidentales / trib. Apure y Meta. Avisa a Caicara.",
    },
    # COBERTURA: Ayacucho → Caicara (sur — Río Ventuari)
    "ventuari": {
        "lat": 4.0,
        "lon": -66.0,
        "params": "PRECTOTCORR,T2M,GWETROOT,EVPTRNS",
        "desc": "Río Ventuari. Tributario sur Orinoco Medio.",
    },
    # COBERTURA: Caicara → Ciudad Bolívar (norte — Llanos de Guárico)
    "llanos_centrales": {
        "lat": 8.0,
        "lon": -66.5,
        "params": "PRECTOTCORR,T2M,GWETROOT,EVPTRNS",
        "desc": "Llanos centrales / Guárico. Avisa al tramo Caicara–Cd. Bolívar.",
    },
    # COBERTURA: Caicara → Ciudad Bolívar (sur — Río Caura / Escudo Guayanés occidental)
    "caura": {
        "lat": 6.0,
        "lon": -64.5,
        "params": "PRECTOTCORR,T2M,GWETROOT,EVPTRNS",
        "desc": "Escudo Guayanés occidental / Río Caura. Avisa a Ciudad Bolívar.",
    },
    # COBERTURA: Ciudad Bolívar → Palúa (Gran Sabana / Río Caroní)
    "caroni": {
        "lat": 5.5,
        "lon": -62.0,
        "params": "PRECTOTCORR,T2M,GWETROOT,EVPTRNS",
        "desc": (
            "Gran Sabana / Río Caroní. Mayor tributario del Orinoco. "
            "NOTA: caudal parcialmente regulado por Embalse del Guri (Corpoelec) — "
            "factor antropogénico no capturado por datos de lluvia. Limitación a documentar."
        ),
    },
}

COMMUNITY: str = "AG"       # Agroclimatology — mejor para lluvia/humedad tropical
BLOCK_SIZE: int = 5         # Años por bloque de descarga
API_PAUSE_S: float = 3.0    # Pausa entre bloques (cortesía con el servidor NASA)
RADAR_PAUSE_S: float = 10.0 # Pausa extra entre radares (evita reset de conexión)
MAX_RETRIES: int = 3        # Reintentos por bloque ante errores de red

START_YEAR: int = 1981      # Inicio de datos satelitales NASA POWER
END_YEAR: int = 2024

# Rutas del repositorio (R7: no hardcodear fuera de esta sección)
RAW_RIVER_PATH: str = "data/raw/legacy_imputed/orinoco_dataset_legacy_simpleml.csv"
ENSO_OUTPUT_PATH: str = "data/external/enso_oni_index.csv"
# Salida: dataset BASE sin feature engineering (rolling, cíclicas, imputación Guri)
# Feature engineering → src/features/build_features.py
OUTPUT_PATH: str = "data/processed/dataset_orinoco_base.csv"


# ==============================================================================
# FUNCIÓN: Descarga de un radar NASA POWER
# ==============================================================================
def download_radar(name: str, lat: float, lon: float, params: str) -> pd.DataFrame:
    """Descarga datos climáticos diarios de NASA POWER para un punto geográfico.

    Args:
        name:   Identificador del radar (usado como prefijo de columnas).
        lat:    Latitud del punto de muestreo.
        lon:    Longitud del punto de muestreo.
        params: Parámetros NASA POWER separados por coma (ej. 'PRECTOTCORR,QV2M').

    Returns:
        DataFrame con índice 'fecha' y columnas prefijadas con el radar.
    """
    logger.info("Radar '%s' (lat=%.1f, lon=%.1f) | params: %s", name, lat, lon, params)
    blocks = []

    for year in range(START_YEAR, END_YEAR + 1, BLOCK_SIZE):
        start_date = f"{year}0101"
        end_date = f"{min(year + BLOCK_SIZE - 1, END_YEAR)}1231"
        logger.info("  -> Descargando bloque %s – %s ...", start_date, end_date)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    "https://power.larc.nasa.gov/api/temporal/daily/point",
                    params={
                        "parameters": params,
                        "community": COMMUNITY,
                        "longitude": lon,
                        "latitude": lat,
                        "start": start_date,
                        "end": end_date,
                        "format": "CSV",
                    },
                    timeout=90,
                )
                if response.status_code == 200:
                    break
                logger.warning(
                    "HTTP %d en intento %d/%d (bloque %s–%s, radar '%s')",
                    response.status_code, attempt, MAX_RETRIES, start_date, end_date, name
                )
            except Exception as exc:
                logger.warning(
                    "Error de red en intento %d/%d: %s", attempt, MAX_RETRIES, exc
                )
                response = None
            # Backoff exponencial entre intentos
            time.sleep(API_PAUSE_S * (2 ** (attempt - 1)))
        else:
            logger.error(
                "Fallaron %d intentos para bloque %s–%s del radar '%s'.",
                MAX_RETRIES, start_date, end_date, name
            )
            time.sleep(API_PAUSE_S)
            continue

        if response is None or response.status_code != 200:
            time.sleep(API_PAUSE_S)
            continue

        lines = response.text.split("\n")
        skip = next(
            (i + 1 for i, line in enumerate(lines) if "-END HEADER-" in line), 0
        )
        block_df = pd.read_csv(StringIO(response.text), skiprows=skip)
        block_df["fecha"] = pd.to_datetime(
            block_df["YEAR"].astype(str) + block_df["DOY"].astype(str), format="%Y%j"
        )
        block_df.drop(columns=["YEAR", "DOY"], inplace=True)
        block_df.set_index("fecha", inplace=True)

        blocks.append(block_df)
        time.sleep(API_PAUSE_S)

    if not blocks:
        raise RuntimeError(f"No se descargó ningún bloque para el radar '{name}'.")

    df = pd.concat(blocks)
    df.replace(-999.0, pd.NA, inplace=True)
    df.ffill(inplace=True)

    # Renombrar columnas con prefijo del radar
    rename_map = {
        "PRECTOTCORR": f"{name}_precipitacion_mm",
        "T2M":         f"{name}_temp_media_c",
        "QV2M":        f"{name}_humedad_especifica",
        # GWETROOT: saturación del suelo (0=seco, 1=saturado) — término ΔS de la ecuación
        "GWETROOT":    f"{name}_humedad_suelo",
        # EVPTRNS: evapotranspiración diaria (mm/día) — término ET de la ecuación
        # Calculado por NASA via Penman-Monteith. Cierra la ecuación: Q = P - ET - ΔS
        "EVPTRNS":     f"{name}_evapotranspiracion_mm",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    logger.info("Radar '%s' listo: %d filas, columnas: %s", name, len(df), list(df.columns))
    return df


# ==============================================================================
# FUNCIÓN: Descarga del índice ENSO ONI (NOAA)
# ==============================================================================
def download_enso_oni(output_path: str) -> pd.DataFrame:
    """Descarga el índice ONI (Oceanic Niño Index) de NOAA y lo expande a resolución diaria.

    El ONI es el índice oficial de la NOAA para identificar episodios de El Niño / La Niña.
    Se calcula como la media móvil de 3 meses de la anomalía de temperatura superficial del
    mar (SST) en la región Niño-3.4 (5°N–5°S, 120°–170°W).

    Interpretación:
        ONI >= +0.5  → El Niño  (déficit de lluvias en Venezuela → caudales bajos)
        ONI <= -0.5  → La Niña  (exceso de lluvias → crecidas e inundaciones)
        -0.5 < ONI < +0.5 → Condición neutral

    Args:
        output_path: Ruta donde guardar el CSV del ENSO procesado.

    Returns:
        DataFrame con índice 'fecha' (diario) y columna 'enso_oni'.
    """
    logger.info("Descargando indice ENSO ONI desde NOAA...")
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Error descargando ENSO ONI: HTTP {response.status_code}")

    # Parsear el formato de NOAA: columnas = SEAS, YR, TOTAL, ANOM
    df_raw = pd.read_csv(
        StringIO(response.text),
        sep=r"\s+",
        header=0,
    )

    # El campo SEAS indica el trimestre centrado (ej. DJF = Dec-Jan-Feb → mes central: Enero)
    season_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }

    records = []
    for _, row in df_raw.iterrows():
        month = season_to_month.get(str(row["SEAS"]).strip())
        if month is None:
            continue
        try:
            year = int(row["YR"])
            oni = float(row["ANOM"])
            records.append({"year": year, "month": month, "enso_oni": oni})
        except (ValueError, KeyError):
            continue

    df_monthly = pd.DataFrame(records)
    df_monthly["fecha"] = pd.to_datetime(
        df_monthly[["year", "month"]].assign(day=1)
    )
    df_monthly.set_index("fecha", inplace=True)
    df_monthly = df_monthly[["enso_oni"]].sort_index()

    # Expandir a resolución diaria: cada día del mes hereda el valor mensual del ONI
    daily_index = pd.date_range(
        start=df_monthly.index.min(),
        end=pd.Timestamp(f"{END_YEAR}-12-31"),
        freq="D",
    )
    df_daily = df_monthly.reindex(daily_index, method="ffill")
    df_daily.index.name = "fecha"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_daily.to_csv(output_path)
    logger.info("ENSO ONI guardado en %s (%d dias)", output_path, len(df_daily))
    return df_daily


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    # 1. Descargar 6 radares NASA POWER
    radar_frames = []
    total = len(RADARES)
    for i, (radar_name, meta) in enumerate(RADARES.items(), start=1):
        logger.info("=== RADAR %d/%d: %s ===  %s", i, total, radar_name.upper(), meta["desc"])
        radar_df = download_radar(radar_name, meta["lat"], meta["lon"], meta["params"])
        radar_frames.append(radar_df)
        if i < total:
            logger.info("Pausa de %.0fs entre radares para no saturar el servidor...", RADAR_PAUSE_S)
            time.sleep(RADAR_PAUSE_S)

    df_climate = pd.concat(radar_frames, axis=1)
    logger.info("Todos los radares unidos. Forma: %s", df_climate.shape)

    # 2. Descargar ENSO ONI
    df_enso = download_enso_oni(ENSO_OUTPUT_PATH)

    # 3. Cargar dataset crudo del río
    logger.info("Cargando dataset crudo: %s", RAW_RIVER_PATH)
    df_river = pd.read_csv(RAW_RIVER_PATH)
    df_river["fecha"] = pd.to_datetime(df_river["fecha"])
    df_river.set_index("fecha", inplace=True)

    # Interpolación limitada: solo brechas pequeñas (max 3 días, ej. 29-feb)
    # Los NaNs estructurales de Ayacucho se tratan en Fase 1b
    for col in df_river.columns:
        df_river[col] = df_river[col].interpolate(method="time", limit=3)

    # 4. Cargar nivel del Guri ya procesado (generado por src/data/process_guri.py)
    #    Fuente original: DAHITI ID-67 (dahiti.dgfi.tum.de)
    #    Lag estimado Guri → Palúa: 1-3 días (Macagua es run-of-river)
    GURI_DAILY_PATH = "data/processed/guri_nivel_diario.csv"
    logger.info("Cargando Guri diario: %s", GURI_DAILY_PATH)
    df_guri = pd.read_csv(GURI_DAILY_PATH, parse_dates=["fecha"])
    df_guri.set_index("fecha", inplace=True)

    # 5. Fusión: río + clima + ENSO + Guri
    logger.info("Fusionando: rio + radares + ENSO + Guri...")
    df_final = df_river \
        .merge(df_climate, left_index=True, right_index=True, how="inner") \
        .merge(df_enso,    left_index=True, right_index=True, how="left") \
        .merge(df_guri,    left_index=True, right_index=True, how="left")

    # ENSO empieza en 1950 → forward fill para NaN al inicio del rango
    df_final["enso_oni"] = df_final["enso_oni"].ffill()

    # NOTA: Guri empieza en 1992. Los NaN pre-1992 se dejan intencionales.
    # La imputación (media estacional + flag guri_imputado) se hace en
    # src/features/build_features.py para mantener la separación de responsabilidades.

    # 6. Guardar base dataset (sin feature engineering)
    df_final.to_csv(OUTPUT_PATH)

    logger.info("Dataset BASE guardado: %s", OUTPUT_PATH)
    logger.info("Filas    : %d", len(df_final))
    logger.info("Columnas : %s", list(df_final.columns))
    logger.info("NaN en guri_nivel_m (pre-1992): %d filas",
                df_final["guri_nivel_m"].isna().sum())
    logger.info("Siguiente paso: python src/features/build_features.py")
