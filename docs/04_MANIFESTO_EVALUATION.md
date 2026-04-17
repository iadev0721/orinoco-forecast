# 🏆 Manifiesto Fase 5: Evaluación Final Comparativa

> **Prerequisito:** Todos los modelos entrenados. `results/metrics/all_models_comparison.csv` existe.

## Propósito

Definir rigurosamente qué significa "éxito" para este modelo y producir las tablas y gráficas
que constituirán el núcleo de la defensa de tesis.

---

## Métricas Hidrológicas (No solo ML)

### Métricas estándar ML

| Métrica | Interpretación | Precaución |
|---------|---------------|------------|
| **MAE** | "El modelo se equivoca en promedio X metros." | Intuitiva, en metros |
| **RMSE** | Penaliza errores grandes. Detecta fallos catastróficos. | Sensible a outliers |
| **MAPE** | Error porcentual. | ⚠️ Problemático en aguas bajas (división por cero) |

### Métricas hidrológicas específicas

**NSE — Nash-Sutcliffe Efficiency (Nash & Sutcliffe, 1970):**

```
NSE = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
```

- NSE = 1: Predicción perfecta.
- NSE = 0: El modelo no es mejor que predecir la media histórica.
- NSE < 0: El modelo es PEOR que la media histórica.
- **Criterio de éxito:** NSE > 0.80 (aceptable), NSE > 0.90 (bueno).
- **Por qué:** EL estándar en hidrología. Todo jurado de ingeniería lo reconoce.

**KGE — Kling-Gupta Efficiency (Gupta et al., 2009):**

```
KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
```

Donde:
- r = correlación de Pearson (¿acertó el timing?)
- α = ratio de desviaciones estándar (¿acertó la variabilidad?)
- β = ratio de medias (¿acertó el volumen/sesgo?)

- **Ventaja sobre NSE:** Descompone el error en 3 componentes → diagnóstico preciso.
- **Criterio de éxito:** KGE > 0.75.

**Peak Timing Error (PTE):**
Diferencia en días entre el pico predicho y el pico real durante eventos de crecida.
Crítico para alertas tempranas. **Criterio: PTE < 3 días.**

**Volume Error Ratio:**
`Σŷ / Σy` durante eventos de crecida.
¿El modelo subestima o sobreestima el volumen de agua?

---

## Clasificación de Régimen Hidrológico

```python
def classify_regime(date: pd.Timestamp) -> str:
    """Clasifica cada fecha en su régimen hidrológico del Orinoco."""
    month = date.month
    if month in [1, 2, 3, 4]:   return "aguas_bajas"
    elif month in [5, 6, 7]:    return "ascenso"
    elif month in [8, 9]:       return "aguas_altas"
    else:                       return "descenso"  # Oct-Dic
```

---

## Preguntas de Control

### PC-04-01: Régimen con Peor Rendimiento
```
¿En qué régimen hidrológico tiene el modelo su PEOR rendimiento?
¿Es consistente con la hipótesis del "punto ciego" del Caroní?

Hipótesis de la tesis: el peor régimen debería ser "aguas_altas" (ago-sep)
porque el Caroní descarga su máximo en ese período y no está en el dataset.

Si se confirma → poderoso argumento en la defensa sobre limitaciones conocidas.
Si no se confirma → ¿qué otro factor explica los errores?
```

### PC-04-02: NSE por Régimen ≥ 0.80
```
¿El NSE del LSTM es > 0.80 en TODOS los regímenes?

Si algún régimen tiene NSE < 0.80:
  - No es necesariamente un fracaso.
  - Documentar el régimen problemático y su causa probable.
  - El modelo SIGUE siendo válido si el NSE global > 0.80.
```

### PC-04-03: Análisis de Residuos
```
¿El modelo comete errores sistemáticos?
Graficar: residuos (y - ŷ) vs tiempo.
¿Los errores se concentran en algún período específico?
Calcular: autocorrelación de los residuos.
Si los residuos tienen autocorrelación significativa → el modelo
no capturó toda la información disponible.
```

### PC-04-04: Peak Timing Error
```
¿El modelo predice correctamente el TIMING de los picos anuales?
Para cada año del test set (2021-2025):
  - Identificar el pico real de Palúa
  - Identificar el pico predicho más cercano
  - Calcular la diferencia en días

Criterio de éxito: PTE < 3 días en promedio.
Visualizar: scatter plot de picos reales vs predichos.
```

### PC-04-05: Justificación del Costo Computacional del LSTM
```
¿El LSTM supera significativamente al SARIMA?
Si la mejora en MAE es < 5%:
  - ¿Justifica la complejidad computacional?
  - ¿Justifica la pérdida de interpretabilidad?
  
Esta pregunta debe responderse honestamente en la tesis,
incluso si la respuesta es desfavorable para el LSTM.
Un hallazgo honesto es más valioso que una conclusión sesgada.
```

---

## Tabla de Resultados Final (EXIGIDA)

Replicar para **cada modelo** (Naive, Seasonal Naive, SARIMA, LSTM, Transformer):

### [Nombre del Modelo] — Set de TEST (2021-2025)

| Métrica | Global | Aguas Bajas | Ascenso | Aguas Altas | Descenso |
|---------|--------|-------------|---------|-------------|----------|
| MAE (m) | — | — | — | — | — |
| RMSE (m) | — | — | — | — | — |
| NSE | — | — | — | — | — |
| KGE | — | — | — | — | — |
| KGE-r | — | — | — | — | — |
| KGE-α | — | — | — | — | — |
| KGE-β | — | — | — | — | — |
| Peak Timing (días) | — | — | — | — | — |

La comparación visual de estas tablas es el **arma definitiva en la defensa de tesis**.

---

## Gráficas de Publicación Requeridas

1. **Serie temporal** del test set: valores reales vs predichos (todos los modelos superpuestos).
2. **Scatter plot** real vs predicho por modelo, coloreado por régimen hidrológico.
3. **Heatmap de métricas** (modelos × régimen).
4. **Curvas de aprendizaje** del LSTM y Transformer.
5. **Análisis de residuos** por régimen.
6. **Zoom en un evento de crecida:** los mejores y peores días predichos.
7. **Si se implementó atención:** heatmap de attention weights sobre la serie histórica.

---

## Entregable

- **Notebook:** `notebooks/06_evaluation_final.ipynb`
- **CSV:** `results/metrics/all_models_comparison.csv`
- **Figuras:** `results/figures/` (formato: PNG 300 DPI para publicación)
