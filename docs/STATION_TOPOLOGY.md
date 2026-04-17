# 🗺️ Topología de Estaciones del Orinoco

## Grafo Hidrológico

```
                    AGUAS ARRIBA
                        │
             ┌──────────┴──────────┐
             │   Puerto Ayacucho   │  km 0 (referencia)
             │   (Radar Lejano)    │  order: 0
             └──────────┬──────────┘
                        │ ~500 km
                        │ lag estimado: 10-15 días
             ┌──────────┴──────────┐
             │      Caicara        │  km 500
             │   (Radar Medio)     │  order: 1
             └──────────┬──────────┘
                        │ ~400 km
                        │ lag estimado: 5-8 días
             ┌──────────┴──────────┐
             │   Ciudad Bolívar    │  km 900
             │   (Radar Cercano)   │  order: 2
             └──────────┬──────────┘
                        │ ~50 km
                        │ lag estimado: 1-2 días
             ┌──────────┴──────────┐
             │       Palúa         │  km 950  ← CONFLUENCIA CARONÍ
             │  (Target Default)   │  order: 3    (punto ciego)
             └─────────────────────┘
                        │
                    AGUAS ABAJO
```

## Tabla de Lag Times

| Par de estaciones | Lag estimado (días) | Fuente |
|---|---|---|
| Ayacucho → Caicara | 10-15 | Velocidad media del flujo en el tramo |
| Caicara → Ciudad Bolívar | 5-8 | Cruz-correlación empírica |
| Ciudad Bolívar → Palúa | 1-2 | Proximidad geográfica |
| Ayacucho → Palúa (total) | 16-25 | Suma de tramos |

> ⚠️ **Estos valores son estimados.** El EDA (PC-01-03) los debe calcular empíricamente
> y actualizar `results/metrics/eda_lag_times.json`. El lookback window se ajusta en función de estos resultados.

## Lógica de Selección Automática de Features

```python
STATION_ORDER = {
    "ayacucho": {"km_from_source": 0, "order": 0},
    "caicara": {"km_from_source": 500, "order": 1},
    "ciudad_bolivar": {"km_from_source": 900, "order": 2},
    "palua": {"km_from_source": 950, "order": 3},
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
# target = "palua"       → primary: [ayacucho, caicara, ciudad_bolivar]
# target = "ciudad_bolivar" → primary: [ayacucho, caicara], excluded: [palua]
# target = "caicara"     → primary: [ayacucho], excluded: [ciudad_bolivar, palua]
```

## El Punto Ciego: El Río Caroní

El río Caroní es el principal afluente del Orinoco aguas abajo de Ciudad Bolívar
y desemboca directamente antes de Palúa (la estación target default).

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
