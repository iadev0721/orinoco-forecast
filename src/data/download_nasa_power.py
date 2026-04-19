import requests
import pandas as pd
import time
import os

# ==============================================================================
# CONFIGURACIÓN DEL PROYECTO
# ==============================================================================
# Coordenadas de la Cuenca del Caroní / Orinoco (Aguas arriba de Palúa)
LATITUD = 8.13   
LONGITUD = -63.57 

# Rango de años a descargar (Desde que hay datos satelitales hasta el presente)
AÑO_INICIO = 1981
AÑO_FIN = 2024 # Usamos 2024 completo

# Rutas adaptadas a la estructura oficial del repositorio Orinoco Forecast
ARCHIVO_RIO = 'data/raw/dataset_orinoco_true_raw.csv'
ARCHIVO_SALIDA = 'data/processed/dataset_orinoco_multivariado_final.csv'

# ==============================================================================
# PASO 1: FUNCIÓN PARA DESCARGAR DATOS DE LA NASA (Evitando bloqueos)
# ==============================================================================
def descargar_clima_nasa(lat, lon, start_year, end_year):
    print(f"Iniciando descarga desde la API de NASA POWER ({start_year}-{end_year})...")
    
    # Parámetros: PRECTOTCORR (Lluvia), T2M (Temp Media), QV2M (Humedad Específica)
    parametros = "PRECTOTCORR,T2M,QV2M"
    dataframes = []
    
    # Descargamos en bloques de 5 años para no saturar el servidor
    for year in range(start_year, end_year + 1, 5):
        fecha_inicio = f"{year}0101"
        fecha_fin = f"{min(year + 4, end_year)}1231"
        
        print(f" -> Descargando bloque: {fecha_inicio} a {fecha_fin}...")
        
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": parametros,
            "community": "AG", # Agroclimatology (Ideal para humedad/lluvia)
            "longitude": lon,
            "latitude": lat,
            "start": fecha_inicio,
            "end": fecha_fin,
            "format": "CSV" # NASA POWER API usa CSV/JSON. Pandas lee CSV fácilmente
        }
        
        # En solicitudes largas a veces JSON es costoso de parsear.
        # Vamos a pedir CSV a la NASA
        respuesta = requests.get(url, params=params)
        
        if respuesta.status_code == 200:
            # Encontrar dónde empiezan los datos saltando el metadata header de NASA
            lines = respuesta.text.split('\n')
            skip = 0
            for i, line in enumerate(lines):
                if "-END HEADER-" in line:
                    skip = i + 1
                    break
            
            # Usamos pd.read_csv con StringIO
            from io import StringIO
            df_bloque = pd.read_csv(StringIO(respuesta.text), skiprows=skip)
            
            # Crear columna fecha combinando YEAR y DOY
            df_bloque['fecha'] = pd.to_datetime(df_bloque['YEAR'].astype(str) + df_bloque['DOY'].astype(str), format='%Y%j')
            df_bloque.drop(['YEAR', 'DOY'], axis=1, inplace=True)
            df_bloque.set_index('fecha', inplace=True)
            
            dataframes.append(df_bloque)
        else:
            print(f"Error en bloque {year}: {respuesta.status_code}")
            
        # Pequeña pausa para ser amables con el servidor de la NASA
        time.sleep(2) 
        
    # Unir todos los bloques en una sola tabla
    df_clima = pd.concat(dataframes)
    
    # Renombrar columnas a español para que cuadre con tu proyecto
    df_clima.rename(columns={
        'PRECTOTCORR': 'precipitacion_mm',
        'T2M': 'temp_media_c',
        'QV2M': 'humedad_especifica'
    }, inplace=True)
    
    # La NASA usa -999.0 cuando hay un error en el satélite. Los cambiamos a nulos y rellenamos.
    df_clima.replace(-999.0, pd.NA, inplace=True)
    df_clima.ffill(inplace=True) # Rellena con el día anterior
    
    print("Descarga del clima completada con exito.\n")
    return df_clima

# ==============================================================================
# PASO 2: EJECUCIÓN Y FUSIÓN DE DATOS
# ==============================================================================
if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Obtenemos el clima histórico
    df_nasa = descargar_clima_nasa(LATITUD, LONGITUD, AÑO_INICIO, AÑO_FIN)
    
    # 2. Cargamos tu CSV crudo orginal del río Orinoco
    print("Cargando el dataset original del rio Orinoco...")
    df_rio = pd.read_csv(ARCHIVO_RIO)
    
    # Asegurarnos de que la columna fecha tenga formato datetime
    df_rio['fecha'] = pd.to_datetime(df_rio['fecha']) 
    df_rio.set_index('fecha', inplace=True)
    
    # 3. Interpolar los huecos de Palúa (los días 29 de febrero)
    # NOTA: Se dejan tranquilos los de Ayacucho porque eso se trata en otra fase
    df_rio['palua'] = df_rio['palua'].interpolate(method='time')
    
    # 4. FUSIÓN (El momento mágico)
    print("Fusionando niveles del rio con datos meteorologicos...")
    df_final = pd.merge(df_rio, df_nasa, left_index=True, right_index=True, how='inner')
    
    # 5. Guardar el nuevo súper-dataset
    df_final.to_csv(ARCHIVO_SALIDA)
    print(f"\nExito! El dataset final se ha guardado como '{ARCHIVO_SALIDA}'")
    print("--- Resumen del nuevo Dataset ---")
    print(f"Filas totales: {len(df_final)}")
    print(f"Columnas: {list(df_final.columns)}")
