# 📊 Manifiesto Fase 1: Análisis Exploratorio de Datos (EDA)

> **Prerequisito:** Fase 0 completada. `results/metrics/phase0_verdict.json` debe existir.

## Propósito

Entender la naturaleza hidrológica de los datos antes de modelar. No se trata solo de graficar:
el EDA debe producir **decisiones concretas** sobre la ingeniería de características que se
implementarán en la Fase 2.

El EDA debe responder: ¿qué sabe el río sobre su propio futuro?

---

## Preguntas de Control

El notebook `01_eda.ipynb` debe responder explícitamente a cada una:

### PC-01-01: Estacionalidad Dominante
```
¿Cuál es la estacionalidad dominante?
Descomponer cada estación con STL (Seasonal-Trend decomposition using LOESS).
Reportar:
  - Período dominante (días)
  - Amplitud del ciclo estacional (metros, promedio y rango)
  - Componente de tendencia: ¿existe drift a largo plazo?
  - Residuos: ¿son ruido blanco o contienen estructura?
```

### PC-01-02: Estacionariedad de la Serie
```
¿Es la serie estacionaria?
Aplicar test ADF (Augmented Dickey-Fuller) y KPSS a cada estación,
tanto en niveles como en primeras diferencias.
Reportar:
  - p-value ADF (H0: la serie tiene raíz unitaria)
  - p-value KPSS (H0: la serie es estacionaria)
  - Conclusión: ¿necesita diferenciación para SARIMA?
  - Implicación para el diseño del lookback window
```

### PC-01-03: Lag Times Empíricos Entre Estaciones ⭐ CRÍTICO
```
¿Cuál es el lag time empírico entre estaciones?
Calcular cross-correlation y reportar el lag (días) donde se maximiza:
  - Ayacucho → Caicara         (inferido preliminar: ~8 días)
  - Caicara → Ciudad Bolívar   (inferido preliminar: ~4 días)
  - Ciudad Bolívar → Palúa     (empírico preliminar: 0 días ⚠️ SOSPECHOSO)
  - Ayacucho → Palúa (total)   (empírico: 12 días, corr=0.981)
  - Caicara → Palúa (total)    (empírico: 4 días, corr=0.979)

INVESTIGAR OBLIGATORIAMENTE:
El lag Ciudad Bolívar → Palúa = 0 días es físicamente implausible:
  - Distancia carretera verificada: ~110-120 km
  - Distancia fluvial: mayor (por meandros — sin datos oficiales sin carta náutica INC)
  - Celeridad de onda típica en ríos de llanura tropical: 30-80 km/día
  - Lag físico esperado: 1-4 días
Hipótesis a evaluar:
  H1: El Caroní domina ambas estaciones simultáneamente (driver común aguas abajo)
  H2: La imputación destruyó el lag natural en los tramos con datos faltantes
  H3: El lag real existe pero es menor a 1 día (granularidad diaria no lo captura)
Documentar la conclusión y su implicación para el modelo en la sección de Limitaciones.

ESTOS VALORES DETERMINAN EL LOOKBACK WINDOW MÍNIMO.
Si el lag total Ayacucho→target supera 30 días, ajustar config.yaml.
```

### PC-01-04: No-Estacionariedad Climática
```
¿Existe evidencia de cambio climático en los datos?
¿Los máximos anuales muestran tendencia creciente/decreciente en 50 años?
Visualizar: serie de máximos anuales + regresión lineal + Mann-Kendall test.
Relevancia para la tesis: ¿el modelo entrenado en datos históricos será válido hoy?
```

### PC-01-05: Outliers y Distribución por Régimen
```
¿Cómo se distribuyen los errores potenciales?
Boxplot por mes y por estación.
Identificar outliers y clasificarlos:
  ¿Sensor defectuoso o evento real de crecida?
Criterio: usar función detect_sensor_anomalies() de src/data/loader.py
```

### PC-01-06: Cambios de Régimen (Regime Changes)
```
¿Existen puntos de quiebre donde el comportamiento del río cambió?
(ej: construcción de represas, deforestación masiva, cambio de uso del suelo)
Usar: Prueba de Pettitt o CUSUM para detección de cambio de media.
Si existe un quiebre significativo, documentar año y causa probable.
```

### PC-01-07: Complejidad e Impredecibilidad
```
Calcular Sample Entropy para cada estación.
¿Cuál estación es más "predecible"?
Hipótesis: si el target elegido es Palúa, debería tener entropía más alta por el
efecto del Caroní (perturbación no observable en el dataset).
Si el target es otra estación, comparar su entropía con el resto y justificar
si la elección es desafiante o conservadora — ambas son válidas, pero deben argumentarse.
Si se confirma alta entropía en el target, esto justifica la necesidad de deep learning.
```

---

## Régimen Hidrológico del Orinoco

Clasificación estándar usada en todo el proyecto:

```python
def classify_regime(month: int) -> str:
    if month in [1, 2, 3, 4]:    return "aguas_bajas"
    elif month in [5, 6, 7]:     return "ascenso"
    elif month in [8, 9]:        return "aguas_altas"
    else:                        return "descenso"  # Oct-Dic
```

El EDA debe analizar las 7 preguntas de control **desagregadas por régimen**.

---

## Entregable

- **Notebook:** `notebooks/01_eda.ipynb` (ejecutado, con outputs y conclusiones)
- **Artefacto:** `results/metrics/eda_lag_times.json`:
  ```json
  {
    "ayacucho_to_caicara_days": null,
    "caicara_to_ciudad_bolivar_days": null,
    "ciudad_bolivar_to_palua_days": null,
    "ayacucho_to_palua_total_days": 12,
    "caicara_to_palua_total_days": 4,
    "ciudad_bolivar_palua_lag0_hypothesis": "pending_eda",
    "recommended_lookback_window": 30,
    "note": "Lags totales son empíricos (cross-corr 18683 registros). Lags por tramo pendientes de EDA.",
    "executed_at": "YYYY-MM-DD"
  }
  ```
- El campo `recommended_lookback_window` actualiza `config.yaml` si difiere del valor actual.
