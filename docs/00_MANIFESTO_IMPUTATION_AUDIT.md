# 📋 Manifiesto Fase 0: Auditoría de Imputación

> **BLOQUEANTE TOTAL.** Ninguna fase posterior puede ejecutarse hasta que este manifiesto
> sea respondido en su totalidad en `notebooks/00_imputation_audit.ipynb`.

## Propósito

Determinar si los datos imputados con "Simple ML" son confiables para entrenar un modelo de
deep learning sobre un sistema hidrodinámico, o si la imputación introdujo artefactos que
degradan la señal hidrológica del Orinoco.

## Contexto Crítico

"Simple ML" es un término genérico. Para un río como el Orinoco, donde los datos faltantes
pueden coincidir con eventos extremos (el sensor falla más durante crecidas por sobrecarga
eléctrica o vandalismo), un método de imputación que no entienda la física del río puede:

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

### PC-00-01: Análisis de Brechas por Estación
```
¿Cuántas brechas tiene cada estación en el dataset original?
Reportar:
  - Conteo total de NaN por estación
  - Brecha más larga consecutiva (días) por estación
  - Distribución temporal: ¿se concentran en algún mes o década?
  - Porcentaje de datos faltantes sobre el total
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

### PC-00-03: Distribución Estadística Original vs Imputada
```
Comparar distribución estadística (media, varianza, curtosis, p5/p95)
entre datos originales y datos imputados, POR ESTACIÓN y POR MES.
¿La imputación aplanó los extremos?
Visualizar: boxplots comparativos por mes
```

### PC-00-04: ACF y PACF — Autocorrelación Artificial
```
Calcular ACF y PACF para ambas versiones (original e imputada) hasta lag=60 días.
¿La versión imputada tiene autocorrelación artificialmente más alta?
Visualizar: gráficos ACF/PACF lado a lado por estación
```

### PC-00-05: Cross-Correlation Entre Estaciones
```
Calcular correlación cruzada entre pares:
  - Caicara → Ciudad Bolívar
  - Ciudad Bolívar → Palúa
  - Ayacucho → Palúa (lag total)
... en ambas versiones (original e imputada).
¿El lag óptimo cambió tras la imputación?
```

### PC-00-06: Visualización Superpuesta en Brechas
```
Visualizar SUPERPUESTO: datos originales (con huecos) vs datos imputados.
ZOOM en las 3 brechas más largas de cada estación.
¿Los valores imputados son físicamente plausibles?
¿Respetan el ciclo estacional esperado?
```

### PC-00-07: VEREDICTO — Decisión de Datos
```
Con base en PC-00-01 a PC-00-06, emitir veredicto:

OPCIÓN A — Usar versión imputada tal como está.
  Criterio: todos los tests de validación aprobados.

OPCIÓN B — Re-imputar con método más sofisticado (MICE, GP interpolation).
  Criterio: ≥ 2 tests fallaron, pero las brechas son recuperables.

OPCIÓN C — Entrenar solo con datos originales, descartando ventanas con NaN.
  Criterio: la imputación introduce sesgos no recuperables, o las brechas
  son pocas y cortas (< 5% del total).

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
