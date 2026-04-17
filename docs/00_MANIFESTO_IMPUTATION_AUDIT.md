# 📋 Manifiesto Fase 0: Auditoría de Imputación

> **BLOQUEANTE TOTAL.** Ninguna fase posterior puede ejecutarse hasta que este manifiesto
> sea respondido en su totalidad en `notebooks/00_imputation_audit.ipynb`.

## Propósito

Determinar si el único dataset disponible (`dataset-orinoco.xlsx`, ya imputado con Simple ML)
es confiable para entrenar un modelo de deep learning sobre un sistema hidrodinámico, o si
la imputación introdujo artefactos que degradan la señal hidrológica del Orinoco.

**Punto de partida conocido:** el dataset tiene 17 NaN en la columna `palua` (todas las fechas
29 de febrero de años bisiestos entre 1976 y 2020, más 5 días en febrero/octubre de 1993).
Esto es un artefacto de la imputación. La Fase 0 debe caracterizarlo y resolverlo.

## Contexto Crítico

**Simple ML** es un término genérico. Para un río como el Orinoco, donde los datos faltantes
pueden coincidir con eventos extremos (el sensor falla más durante crecidas por sobrecarga
eléctrica o vandalismo), un método de imputación que no entienda la física del río puede:

**Situación real del dataset:** Solo existe la versión ya imputada. El dataset original
(con brechas naturales) no está disponible para comparación directa. Esto significa que
los tests que requieren "original vs imputado" se sustituyen por tests de **plausibilidad
hidrológica** y **coherencia de la señal** sobre el único dataset disponible.

1. **Aplanar picos de crecida:** Si el modelo suavizó valores durante brechas en agosto-septiembre,
   el LSTM aprenderá que las crecidas son menos severas de lo que realmente son.

2. **Fabricar autocorrelación artificial:** La serie imputada tendrá autocorrelación más alta que la real,
   haciendo que el LSTM parezca mejor de lo que es.

3. **Destruir la correlación cruzada entre estaciones:** El desfase natural (lag) entre estaciones
   puede haberse distorsionado si las brechas se rellenaron de forma independiente.

4. **Sesgo de selección temporal:** Las brechas pueden ser más frecuentes en temporadas críticas.

---

## Preguntas de Control

El notebook `00_imputation_audit.ipynb` debe responder explícitamente a cada una:

### PC-00-01: Análisis de NaN por Estación
```
Caracterizar los 17 NaN conocidos en palua y buscar NaN ocultos en las demás estaciones:
Reportar:
  - Conteo total de NaN por estación (verificar los 17 de palua)
  - Fecha exacta de cada NaN en palua y clasificación:
      * Grupo A: 29 de febrero (años bisiestos) — artefacto de calendarización
      * Grupo B: 5 días en 1993 (feb 27-28 y oct 29-31) — causa a investigar
  - Distribución temporal: ¿se concentran en algún régimen hidrológico?
  - Estrategia de resolución: interpolación lineal (bisiestos) vs método más robusto (1993)
```

### PC-00-02: Coincidencia Temporal de Brechas
```
¿Coinciden temporalmente las brechas entre estaciones?
Si Caicara y Ciudad Bolívar tienen brechas simultáneas, el sistema de
"radares" pierde su poder predictivo en esos períodos.
Reportar:
  - Heatmap de brechas por estación y tiempo
  - Períodos con ≥ 2 estaciones con NaN simultáneos
  - Impacto estimado en el poder predictivo del modelo
```

### PC-00-03: Plausibilidad Hidrológica de la Señal
```
Sin dataset original para comparar directamente, evaluar plausibilidad de la señal:
  - Distribución mensual: ¿respeta el ciclo hidrológico típico del Orinoco?
    (aguas bajas: ene-abr, ascenso: may-jul, crecida: ago-sep, descenso: oct-dic)
  - Curtosis y asimetría por mes y estación: ¿los extremos son realistas?
  - Comparar máximos históricos contra registros documentados del INAMEH
  - Detectar zonas con varianza anormalmente baja (¿señal imputada plana?):
    ventanas de 30 días con std < percentil 10 de std móvil son sospechosas
```

### PC-00-04: ACF y PACF — Autocorrelación de la Señal
```
Calcular ACF y PACF hasta lag=60 días para cada estación.
¿Existen picos artificiales en la autocorrelación que sugieran imputación suavizada?
Buscar: autocorrelación anormalmente alta en lags 1-7 durante años bisiestos.
```

### PC-00-05: Cross-Correlation Entre Estaciones
```
Calcular correlación cruzada entre pares:
  - Ayacucho → Caicara
  - Caicara → Ciudad Bolívar
  - Ciudad Bolívar → Palúa  ← lag=0 empírico es sospechoso: investigar
  - Ayacucho → Palúa (lag total)
Esperado (físico): Ciudad Bolívar → Palúa debería tener lag de 1-4 días
(~110-120 km de distancia, celeridad de onda típica en rios de llanura: 30-80 km/día).
Si el lag empírico sigue siendo 0, documentar las hipótesis: (1) el Caroní domina
ambas estaciones simultáneamente, (2) la imputación destruyó el lag natural.
```

### PC-00-06: Visualización de las Zonas Imputadas Conocidas
```
Visualizar en detalle los 17 registros NaN de palua y su contexto (±15 días):
  - ¿Los valores adyacentes son coherentes con el ciclo esperado?
  - ¿El 29 de febrero imputado (una vez resuelto) respeta la tendencia de esa semana?
  - Para los 5 días de 1993: ¿coincide con un evento hidrológico documentado?
    Buscar en registros históricos del INAMEH o literatura de crecidas extremas del Orinoco.
```

### PC-00-07: VEREDICTO — Decisión de Datos
```
Con base en PC-00-01 a PC-00-06, emitir veredicto:

OPCIÓN A — Usar el dataset tal como está, imputando los 17 NaN de palua con
            interpolación lineal (bisiestos) y método específico para los 5 de 1993.
  Criterio: todos los tests de validación aprobados.

OPCIÓN B — Imputar los 17 NaN con método más sofisticado (MICE, KNN con estaciones
            vecinas, interpolación basada en correlación con ciudad_bolivar).
  Criterio: ≥1 test falló pero la señal es recuperable.

OPCIÓN C — Descartar las ventanas de tiempo que contengan los 17 NaN.
  Criterio: los NaN no pueden imputarse de forma confiable (bisiestos en aguas extremas).

El veredicto debe quedar registrado en results/metrics/phase0_verdict.json
```

---

## Tests de Validación de la Imputación

| ID | Test | Descripción | Criterio de Aprobación |
|----|------|-------------|----------------------|
| T1 | Distribución | Kolmogorov-Smirnov entre distribuciones original vs imputada por estación | p-value > 0.05 |
| T2 | Estacionalidad | Comparar amplitud del ciclo anual promedio (original vs imputado) | Diferencia < 5% |
| T3 | Extremos | Comparar percentil 95 y percentil 5 (original vs imputado) | Diferencia < 10% |
| T4 | Lag cruzado | Lag óptimo entre estaciones consecutivas | Diferencia ≤ 1 día |
| T5 | Varianza local | Varianza en ventanas de 30 días durante períodos imputados vs reales | Ratio ∈ [0.7, 1.3] |

---

## Entregable

- **Notebook:** `notebooks/00_imputation_audit.ipynb` (ejecutado, con outputs)
- **Artefacto:** `results/metrics/phase0_verdict.json` con:
  ```json
  {
    "verdict": "A|B|C",
    "dataset_to_use": "data/raw/... o data/processed/dataset_audited.parquet",
    "tests_passed": ["T1", "T2", "T3"],
    "tests_failed": ["T4", "T5"],
    "justification": "...",
    "executed_at": "YYYY-MM-DD"
  }
  ```

> ⚠️ **Sin este archivo, ningún agente puede proceder a Fase 1.**
