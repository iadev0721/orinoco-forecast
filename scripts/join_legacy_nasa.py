import os
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    legacy_path = "data/raw/legacy_imputed/orinoco_dataset_legacy_simpleml.csv"
    nasa_path = "data/processed/nasa_power_radares.csv"
    output_path = "data/processed/joined_legacy_nasa.csv"

    logger.info(f"Cargando {legacy_path}...")
    df_legacy = pd.read_csv(legacy_path)
    # Convertimos a datetime para estandarizar el formato de fecha (viene como YYYY/MM/DD)
    df_legacy['fecha'] = pd.to_datetime(df_legacy['fecha'])
    
    logger.info(f"Cargando {nasa_path}...")
    df_nasa = pd.read_csv(nasa_path)
    # Convertimos a datetime (viene como YYYY-MM-DD)
    df_nasa['fecha'] = pd.to_datetime(df_nasa['fecha'])

    logger.info("Uniendo datasets por la columna 'fecha' (Inner Join)...")
    # Usamos inner join para quedarnos solo con las fechas donde ambos tienen datos
    df_joined = pd.merge(df_legacy, df_nasa, on='fecha', how='inner')
    
    # Ordenamos cronológicamente
    df_joined = df_joined.sort_values('fecha')

    # Guardamos el archivo
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Guardando resultado en {output_path} ({len(df_joined)} filas obtenidas)")
    df_joined.to_csv(output_path, index=False)
    logger.info("¡Proceso de unión completado exitosamente!")

if __name__ == "__main__":
    main()
