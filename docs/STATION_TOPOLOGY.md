# 🗺️ Topología de Estaciones del Orinoco

## Grafo Hidrológico

```
                    AGUAS ARRIBA
                        │
             ┌──────────┴──────────┐
             │   Puerto Ayacucho   │  km_relative: 0 (referencia)
             │   (Predictor Lejano) │  order: 0
             └──────────┬──────────┘
                        │ ~340-350 km carretera (dist. fluvial: mayor, sin datos oficiales)
                        │ lag empírico: 12 días (cross-corr 50 años de datos, corr=0.981)
             ┌──────────┴──────────┐
             │      Caicara        │  km_relative: 500 (aprox. relativo)
             │   (Predictor Medio) │  order: 1
             └──────────┬──────────┘
                        │ ~365-370 km carretera (dist. fluvial: mayor, sin datos oficiales)
                        │ lag empírico: 4 días (cross-corr 50 años de datos, corr=0.979)
             ┌──────────┴──────────┐
             │   Ciudad Bolívar    │  km_relative: 900 (aprox. relativo)
             │  (Predictor Cercano) │  order: 2
             └──────────┬──────────┘
                        │ ~110-120 km carretera (verificado)
                        │ lag empírico: 0 días ⚠️ SOSPECHOSO — investigar en EDA PC-01-03
             ┌──────────┴──────────┐
             │       Palúa         │  km_relative: 1000  ← CONFLUENCIA CAROÍ
             │  (Estación Target)  │  order: 3    (punto ciego)
             └─────────────────────┘
                        │
                    AGUAS ABAJO
```

## Tabla de Lag Times

| Par de estaciones | Lag empírico (días) | Fuente |
|---|---|---|
| Ayacucho → Caicara | ~8 (inferido) | Lag total - lags individuales |
| Caicara → Ciudad Bolívar | ~4 (inferido) | Lag total - lag Caicara |
| Ciudad Bolívar → Palúa | 0 ⚠️ | Cross-correlación empírica — SOSPECHOSO |
| Ayacucho → Palúa (total) | 12 | Cross-corr dataset real (1974-2025, corr=0.981) |
| Caicara → Palúa (total) | 4 | Cross-corr dataset real (1974-2025, corr=0.979) |

> ⚠️ **Lag 0 en Ciudad Bolívar → Palúa es físicamente sospechoso.** La distancia por carretera
> es ~110-120 km; la distancia fluvial es mayor por los meandros. Con una celeridad típica
> de ondas de crecida en ríos de llanura tropical (30-80 km/día), el lag esperado sería
> de 1-4 días. El lag=0 empírico puede deberse a:
> 1. El Caroní impacta ambas estaciones simultáneamente (driver domínante aguas abajo)
> 2. La imputación destruyó el lag natural en los tramos imputados
> **Investigar en EDA PC-01-03 y documentar la conclusión en la tesis.**
>
> ⚠️ **Los lags por tramo (Ayacucho→Caicara, Caicara→CiudadBolívar) son INFERIDOS**, no
> medidos directamente: el lag total Ayacucho→Palúa (12 días) y Caicara→Palúa (4 días)
> son empíricos; los tramos intermedios se calculan por sustitución. El EDA debe
> calcularlos directamente (Ayacucho→Caicara, Caicara→Ciudad Bolívar) con cross-corr.

## Lógica de Selección Automática de Features

```python
STATION_ORDER = {
    "ayacucho":        {"km_relative": 0,   "order": 0},  # Verificado (referencia)
    "caicara":         {"km_relative": 339, "order": 1},  # Verificado: Google Maps 2026-04-19
    "ciudad_bolivar": {"km_relative": 698, "order": 2},  # Verificado: 339+359 km
    "palua":           {"km_relative": 791, "order": 3},  # Verificado: ruta directa A→D
    # NOTA: suma de tramos = 817 km. Ruta directa Google Maps Ayacucho→Palúa = 791 km.
    # Discrepancia de 26 km por optimización de rutas. Se usa medición directa.
}


def get_predictors(target: str, station_graph: dict) -> dict:
    """
    Dado un target, retorna las estaciones predictoras y su rol
    basado en posición relativa en el grafo hidrológico.

    Reglas:
      - Estaciones AGUAS ARRIBA del target → PREDICTORES PRIMARIOS
        (contienen información causal del futuro del target)
      - El TARGET mismo → PREDICTOR AUTOREGRESIVO
      - Estaciones AGUAS ABAJO → EXCLUIDAS
        (el agua no fluye hacia arriba: no tienen poder predictivo causal)

    Args:
        target: Nombre de la estación objetivo.
        station_graph: Diccionario con 'order' por estación.

    Returns:
        Dict con keys 'primary', 'self', 'excluded'.
    """
    target_order = station_graph[target]["order"]

    predictors = {
        "primary": [],
        "self": target,
        "excluded": [],
    }

    for station, info in station_graph.items():
        if station == target:
            continue
        if info["order"] < target_order:
            predictors["primary"].append(station)
        else:
            predictors["excluded"].append(station)

    return predictors


# Ejemplos de uso:
# target = "palua"           → primary: [ayacucho, caicara, ciudad_bolivar]
# target = "ciudad_bolivar" → primary: [ayacucho, caicara], excluded: [palua]
# target = "caicara"        → primary: [ayacucho], excluded: [ciudad_bolivar, palua]
# target = "ayacucho"       → primary: [], excluded: [caicara, ciudad_bolivar, palua] (solo autoreg.)
```

## Cobertura de Radares Climáticos (NASA POWER) — Implementado 2026-04-19

Para cubrir los puntos ciegos de la cuenca, se implementó una estrategia de
**Multi-Source Exogenous Input** con 6 radares NASA POWER, uno por sub-cuenca.
Script: `src/data/download_nasa_power.py`.
Dataset resultante: `data/processed/dataset_orinoco_multivariado_final.csv`

```
Segmento                  Radar                Lat/Lon           Avisa a
────────────────────────────────────────────────────────────────────────────────
Upstream Ayacucho         amazonas             4.5°N, -67.6°W   Ayacucho (lag ~12d)
Ayacucho→Caicara (N)     apure_meta           7.5°N, -69.5°W   Caicara
Ayacucho→Caicara (S)     ventuari             4.0°N, -66.0°W   Orinoco Medio
Caicara→Cd. Bolívar (N)  llanos_centrales     8.0°N, -66.5°W   Caicara→Cd. Bolívar
Caicara→Cd. Bolívar (S)  caura                6.0°N, -64.5°W   Ciudad Bolívar
Cd. Bolívar→Palúa        caroni               5.5°N, -62.0°W   Aguas abajo (*)
```

(*) El Caroní está parcialmente regulado por el Embalse del Guri (Corpoelec).
Los datos de lluvia capturan la tendencia estacional pero NO las decisiones
antropógenas de apertura de compuertas. Documentar como limitación en la tesis.

**Variables por radar:** `{nombre}_precipitacion_mm`, `{nombre}_temp_media_c`, `{nombre}_humedad_especifica`
**Total columnas en el dataset final:** 22 (4 estaciones + 6 radares × 3 variables)
**Rango temporal de los radares:** 1981–2024 (NASA POWER, inicio satelital)
**NOTA:** Los 7 años 1974–1980 del dataset del río NO tienen datos climáticos de apoyo.


El río Caroní es el principal afluente del Orinoco en el tramo inferior de la cuenca
y desemboca directamente en el Orinoco en la zona de Palúa/San Félix (Ciudad Guayana).
Esta confluencia ocurre **aguas abajo de Ciudad Bolívar** y explica por qué la estación
Palúa puede comportarse de forma distinta al patrón upstream.

| Característica | Valor |
|---|---|
| Descarga promedio | ~4,850 m³/s |
| Descarga máxima | ~10,000 m³/s (crecidas extremas) |
| Variabilidad | ±50% en horas (regulada por Macagua/Guri) |
| Efecto en Palúa | Puede elevar el nivel 1-3 metros en días |

**Impacto en el modelo:**
- El Caroní NO está en el dataset.
- Sus variaciones pueden dominar el nivel de Palúa en aguas altas.
- Los peores errores del modelo probablemente correlacionan con eventos del Caroní.
- Mitigación parcial: variable `caroni_proxy` definida en `02_MANIFESTO_PREPROCESSING.md`.

**Documentar en la tesis:** Esta limitación debe aparecer explícitamente en la sección
"Limitaciones del Modelo". Un jurado respeta más una limitación bien documentada que
una predicción sospechosamente perfecta.

## Información sobre Fallos de Sensor

```python
def detect_sensor_anomalies(series: pd.Series) -> pd.Series:
    """
    Detecta patrones típicos de fallos de sensor en datos hidrométricos.

    Patrones detectados:
      1. Valores constantes > 3 días (sensor atascado)
      2. Saltos > 2m en un día durante aguas bajas
      3. Valores negativos (físicamente imposible)
      4. Valores > máximo histórico + 20%

    Args:
        series: Serie temporal del nivel del río (metros).

    Returns:
        Serie de flags (0 = normal, >0 = sospechoso, suma de flags activados).
    """
    flags = pd.Series(0, index=series.index)

    # Sensor atascado: valor sin cambio por > 3 días
    flags[series.diff().abs() == 0] += 1

    # Salto abrupto durante aguas bajas
    daily_change = series.diff().abs()
    is_low_water = series < series.quantile(0.25)
    flags[(daily_change > 2) & is_low_water] += 1

    # Valores físicamente imposibles
    flags[series < 0] += 1
    flags[series > series.max() * 1.2] += 1

    return flags
```
