import pandas as pd
import os

# Archivo origen
input_file = "POWER_Point_Daily_20200101_20201231_008d13N_063d57W_LST.csv"

# Archivo destino (adaptado a las convenciones del repositorio)
output_dir = "data/external"
output_file = os.path.join(output_dir, "nasa_power_weather.csv")

print("Procesando datos de NASA POWER...")

# Leer el CSV omitiendo las primeras 11 líneas del encabezado y la línea de metadatos (-END HEADER-)
df = pd.read_csv(input_file, skiprows=11)

# Crear una columna de fecha 'fecha' a partir de YEAR y DOY
# pd.to_datetime con formato '%Y%j' convierte Año y Día del Año directamente a fecha
df['fecha'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j')

# Eliminar las columnas YEAR y DOY ya que ahora tenemos 'fecha'
df.drop(['YEAR', 'DOY'], axis=1, inplace=True)

# Reordenar para que 'fecha' sea la primera columna, imitando el dataset_orinoco_true_raw.csv
cols = ['fecha'] + [col for col in df.columns if col != 'fecha']
df = df[cols]

# Renombrar columnas para que sean más entendibles (opcional pero conveniente)
# PRECTOTCORR -> precipitacion_mm
# T2M -> temp_media_c
# QV2M -> humedad_especifica_g_kg
df.rename(columns={
    'PRECTOTCORR': 'precipitacion_mm',
    'T2M': 'temp_media_c',
    'QV2M': 'humedad_especifica'
}, inplace=True)

# Asegurarse de que el directorio exista
os.makedirs(output_dir, exist_ok=True)

# Guardar el CSV final
df.to_csv(output_file, index=False)

print(f"✅ Archivo adaptado y guardado exitosamente en: {output_file}")
print("El archivo ahora tiene una columna 'fecha' compatible con los notebooks del proyecto.")

# Borrar el archivo original
os.remove(input_file)
print("🗑️ Archivo original eliminado de la raíz del proyecto para mantener el orden.")
