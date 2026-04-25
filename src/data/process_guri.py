"""
process_guri.py
===============
Procesa la serie temporal cruda del nivel del Embalse del Guri descargada de DAHITI
(Database for Hydrological Time Series of Inland Waters, TU Múnich, ID-67)
y la convierte en una serie diaria continua lista para fusionar con el dataset principal.

Entrada : data/external/guri_nivel_dahiti.csv   (lectura satelital irregular, ~10 días)
Salida  : data/processed/guri_nivel_diario.csv  (serie diaria continua, pchip interpolado)

Pasos:
  1. Cargar CSV crudo (separador ';', columnas: fecha, guri_nivel_m, incertidumbre)
  2. Resolver duplicados → promedio por día (puede haber 2 satélites en el mismo día)
  3. Resamplear a diario → NaN en días sin satélite
  4. Interpolar con pchip (Piecewise Cubic Hermite):
       - Preserva monotonía local → no genera picos artificiales
       - Físicamente válido: Guri (135 km³) cambia muy suavemente entre lecturas
  5. Guardar solo guri_nivel_m (la incertidumbre NO se interpola: sería ficticia)

Justificación hidrológica:
  El nivel del Guri es la variable de estado del sistema Caroní.
  Controla cuánta agua libera Macagua (run-of-river, ~100 km de Palúa).
  Lag estimado Guri → Palúa: 1-3 días.
  Referencia: DAHITI v8.0 (Schwatke et al. 2015, HESS).
"""

import logging
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_PATH  = "data/external/guri_nivel_dahiti.csv"
OUTPUT_PATH = "data/processed/guri_nivel_diario.csv"
# ---------------------------------------------------------------------------


def process_guri(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Convierte la serie cruda de altimetría satelital del Guri (DAHITI ID-67)
    en una serie temporal diaria continua usando interpolación pchip.

    Returns
    -------
    pd.DataFrame con índice diario y columna 'guri_nivel_m'.
    """
    # 1. Cargar
    logger.info("Cargando datos crudos: %s", input_path)
    df = pd.read_csv(input_path, parse_dates=["fecha"])
    logger.info("  Registros crudos: %d  |  Rango: %s → %s",
                len(df), df["fecha"].min().date(), df["fecha"].max().date())

    # 2. Resolver duplicados (2 pasos de satélites distintos en el mismo día → promedio)
    df = df.groupby(df["fecha"].dt.date)["guri_nivel_m"].mean().reset_index()
    df.columns = ["fecha", "guri_nivel_m"]
    df["fecha"] = pd.to_datetime(df["fecha"])
    df.set_index("fecha", inplace=True)
    df.sort_index(inplace=True)

    n_dup_removed = 1005 - len(df)   # 1005 = registros originales
    logger.info("  Duplicados eliminados (promediados): %d", n_dup_removed)

    # 3. Resamplear a frecuencia diaria (introduce NaN en días sin satélite)
    df_diario = df.resample("D").asfreq()
    n_huecos = df_diario["guri_nivel_m"].isna().sum()
    logger.info("  Días en el rango: %d  |  Huecos (sin satélite): %d  |  Espacio medio: %.1f días",
                len(df_diario), n_huecos, n_huecos / max(len(df) - 1, 1))

    # 4. Interpolación pchip
    #    NOTA: solo se interpola guri_nivel_m.
    #    La columna de incertidumbre (wse_u) NO se incluye: interpolar incertidumbre
    #    de una medición satelital en días sin observación no tiene significado físico.
    df_diario["guri_nivel_m"] = df_diario["guri_nivel_m"].interpolate(method="pchip")

    # Verificación de rango físico del Guri (mín operativo ~240 m, máx ~272.5 m)
    pmin = df_diario["guri_nivel_m"].min()
    pmax = df_diario["guri_nivel_m"].max()
    assert 238 < pmin < 250, f"Mínimo fuera de rango físico: {pmin:.3f} m"
    assert 268 < pmax < 275, f"Máximo fuera de rango físico: {pmax:.3f} m"
    logger.info("  Rango post-interpolación: %.3f m → %.3f m  ✓ físicamente válido", pmin, pmax)

    # 5. Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_diario[["guri_nivel_m"]].to_csv(output_path)
    logger.info("Guardado: %s  (%d filas)", output_path, len(df_diario))

    return df_diario[["guri_nivel_m"]]


if __name__ == "__main__":
    df = process_guri()
    print("\n=== Resumen del dataset procesado ===")
    print(df.describe().round(3))
    print(f"\nPrimeras 3 filas:\n{df.head(3)}")
    print(f"Últimas 3 filas:\n{df.tail(3)}")
