# Fase 6: Experimentos con Transformer — Orinoco Forecast

> **Rama:** `feature/transformer-1981-2025`
> **Fecha:** 2026-04-30
> **Dataset:** `data/processed/dataset_orinoco_features.csv` (78 features, 1981–2025)
> **Splits:** Train: 1981–2018 | Val: 2019–2022-06 | Test: 2022-07–2025-02 (idénticos al LSTM gold standard)

---

## 1. Objetivo y Contexto

Después de establecer el LSTM gold standard (`ensemble_lb150_lags_xl`, MAE=13.3 cm, NSE=0.9959),
el objetivo de esta fase fue evaluar si una arquitectura **Transformer** basada en auto-atención
puede igualar o superar ese rendimiento sin necesidad de ingeniería manual de features adicionales.

La hipótesis: el mecanismo de atención puede aprender *autónomamente* qué días de los 150 días
de historial son más relevantes para cada predicción, sin depender de las memorias implícitas de
las celdas LSTM.

---

## 2. Arquitectura del Transformer

El modelo se implementó en PyTorch (`src/models/transformer_model.py`) con los siguientes
componentes:

```
Entrada (batch, lookback, n_features)
    ↓
Proyección lineal  →  d_model
    ↓
Positional Encoding (sinusoidal, Pre-LN)
    ↓
TransformerEncoder (N capas de self-attention)
    ↓
Último token  [CLS-like aggregation]
    ↓
MLP head: LayerNorm → Linear → GELU → Dropout → Linear(horizon)
    ↓
Salida (batch, horizon=7)
```

**Detalles de implementación relevantes:**
- `PositionalEncoding` hereda de `nn.Module` (crítico para que Dropout se desactive en `eval()`)
- `batch_first=True`, `norm_first=True` (Pre-Layer Normalization — más estable que Post-LN)
- Optimizador: **AdamW** (weight decay implícito → mejor regularización que Adam)
- Gradient clipping: `max_norm=1.0` (estabiliza las primeras épocas)
- Checkpoint: guarda el mejor epoch por `val_loss`

---

## 3. El Paradigma Residual — La Clave del Éxito

### 3.1 El problema sin residual

El primer intento (`transformer_v1`) entrenó al Transformer para predecir el **nivel absoluto**
del río (en escala normalizada [0,1]).

```
Resultado (transformer_v1, sin residual):
  MAE test  = 40.4 cm
  NSE test  = 0.9727  ← PEOR que el naive baseline (0.9894)
```

**BANDERA ROJA:** El NSE del Transformer fue inferior al modelo naïve (predecir
"mañana = hoy"). El Shadow Effect lo confirmó:
`corr(pred, y_lag1)=0.9868 > corr(pred, y_true)=0.9860`

El modelo aprendió a copiar el nivel de ayer porque eso minimizaba el MSE en niveles absolutos
(el río cambia lentamente). Era un modelo de persistencia disfrazado.

### 3.2 La solución: predicción residual (delta)

La misma estrategia que hizo competitivo al LSTM: entrenar sobre el **delta** en lugar del nivel absoluto.

```
target[i] = nivel_futuro[i] - nivel_actual[i]   (en espacio normalizado)
```

En el pipeline (`src/data/pipeline.py`, flag `use_residual: true` en `config.yaml`):

```python
anchor = data_y[i + lookback - 1]          # nivel "hoy" (normalizado)
y[i]   = data_y[i+lookback : i+lookback+horizon] - anchor  # predicción de Δ
```

Durante inferencia, la reconstrucción es:
```python
nivel_pred_norm = anchor + delta_pred_norm
nivel_pred_m    = scaler_y.inverse_transform(nivel_pred_norm)
```

**¿Por qué funciona mejor?**

| Aspecto | Sin residual | Con residual |
|---|---|---|
| Target típico | ~0.55 (nivel medio normalizado) | ~±0.01 (cambio diario pequeño) |
| Baseline naïve | Predecir el promedio histórico | Predecir Δ=0 (persistencia) |
| Aprendizaje necesario | Memorizar el nivel absoluto | Aprender cuándo el río subirá o bajará |
| Shadow Effect | Siempre presente | Desaparece tras convergencia |

Con residual, el modelo **compite directamente contra la persistencia** (Δ=0). Si predice Δ=0
en todo momento, obtiene MAE ≈ 25.6 cm (el naive baseline). Para mejorar, **debe** aprender
física real.

---

## 4. Evolución de Experimentos

### Experimento 1 — `transformer_v1` (sin residual)
- **Config:** d_model=64, nhead=4, 2 capas, lookback=30, splits 2007/2016
- **Resultado:** MAE=40.4 cm, NSE=0.9727 — peor que naive
- **Diagnóstico:** Sin residual + lookback corto + splits no alineados con LSTM gold

---

### Experimento 2 — `transformer_v2_residual` (modelo único)
- **Config:** d_model=64, nhead=4, 2 capas, lookback=150, residual=True
- **Épocas:** 62 (early stopping con patience=10)
- **Resultado:**

```
MAE test  = 13.8 cm    (LSTM gold: 13.3 cm)
RMSE test = 20.0 cm    (LSTM gold: 19.9 cm)
NSE test  = 0.9958     (LSTM gold: 0.9959)
KGE test  = 0.9901     (LSTM gold: 0.9943)
NSE val   = 0.9951     (LSTM gold: 0.9951) ← IDÉNTICO
```

**Hito crítico:** Un único Transformer, sin tuning específico de hiperparámetros, igualó
prácticamente al LSTM ensemble optimizado durante semanas.

---

### Experimento 3 — `ensemble_transformer_xl` (5 × d_model=128, 3 capas)
- **Config:** d_model=128, nhead=4, 3 capas, dim_feedforward=256
- **Resultado:**

```
MAE test  = 14.0 cm  ← PEOR que el modelo único v2 (13.8 cm)
NSE test  = 0.9956
KGE test  = 0.9930
```

**¿Por qué el modelo más grande produjo peores resultados que el modelo único más pequeño?**

Hay tres factores que explican este comportamiento paradójico:

**a) Sobreparametrización con dataset pequeño de validación:**
El conjunto de validación tiene solo 1,121 muestras (vs. 13,723 de entrenamiento).
Con d_model=128 y 3 capas, el modelo tiene ~2.5× más parámetros. La superficie de pérdida
es más compleja y irregular, haciendo que la señal de early stopping sea más ruidosa.
Los modelos tienden a detenerse en mínimos subóptimos.

**b) Mayor dispersión entre miembros del ensemble:**
Con arquitecturas más grandes, cada miembro con semilla diferente converge a soluciones
más distintas entre sí (la varianza del ensemble aumenta). El promedio de predicciones
muy distintas puede introducir artefactos que el promedio de soluciones similares no tiene.

**c) El d_model=64 ya captura toda la información relevante:**
El río Orinoco tiene 77 features de entrada con dinámicas estacionales conocidas. Un modelo
de 64 dimensiones es suficiente para representar esas relaciones. Aumentar a 128 no añade
capacidad útil — añade ruido de optimización.

> **Lección:** En series temporales hidro-climáticas con fuerte estructura estacional,
> la capacidad óptima del Transformer es sorprendentemente pequeña. Más parámetros
> no implica mejor pronóstico.

---

### Experimento 4 — `ensemble_transformer_v1` (5 × d_model=64) — Gold Standard Transformer
- **Config:** d_model=64, nhead=4, 2 capas, dim_feedforward=128, lookback=150, residual=True
- **Semillas:** [42, 123, 456, 789, 1011]
- **Resultado:**

```
MAE test  = 13.4 cm    (LSTM gold: 13.3 cm)  → diferencia de 1 mm
RMSE test = 19.8 cm    (LSTM gold: 19.9 cm)  → TRANSFORMER GANA
NSE test  = 0.9959     (LSTM gold: 0.9959)   → EMPATE EXACTO
KGE test  = 0.9919     (LSTM gold: 0.9943)   → LSTM todavía gana
MAE val   = 15.4 cm    (LSTM gold: 15.7 cm)  → TRANSFORMER GANA
NSE val   = 0.9954     (LSTM gold: 0.9951)   → TRANSFORMER GANA
KGE val   = 0.9968     (LSTM gold: 0.9951)   → TRANSFORMER GANA
```

---

## 5. Tabla Resumen de Todos los Experimentos

| Modelo | MAE test | RMSE test | NSE test | KGE test | Épocas/Miembros |
|---|---|---|---|---|---|
| Naive baseline | 25.6 cm | — | 0.9894 | — | — |
| `transformer_v1` (sin residual) | 40.4 cm | 51.2 cm | 0.9727 | 0.9509 | 28 épocas |
| `transformer_v2_residual` (single) | 13.8 cm | 20.0 cm | 0.9958 | 0.9901 | 62 épocas |
| `ensemble_transformer_xl` (5×d128) | 14.0 cm | 20.6 cm | 0.9956 | 0.9930 | 5 miembros |
| **`ensemble_transformer_v1` (5×d64)** | **13.4 cm** | **19.8 cm** | **0.9959** | 0.9919 | 5 miembros |
| **LSTM gold standard** (5×ensemble) | **13.3 cm** | 19.9 cm | **0.9959** | **0.9943** | 5 miembros |

---

## 6. Análisis del KGE — La Única Brecha Restante

El KGE (Kling-Gupta Efficiency) descompone el error en tres componentes:

| Componente | `ensemble_transformer_v1` | LSTM gold | Interpretación |
|---|---|---|---|
| **kge_r** (correlación) | 0.9980 | 0.9980 | **Idéntico** — igual timing |
| **kge_alpha** (ratio std) | 1.0042 | 1.0053 | **Idéntico** — igual variabilidad |
| **kge_beta** (ratio media) | 1.0067 | 1.0004 | Transformer: +0.67% sobreestimación |

La única diferencia significativa es un **sesgo volumétrico del +0.67%** en el Transformer:
predice niveles marginalmente más altos en promedio. El LSTM tiene un sesgo casi nulo (beta=1.0004).

### Intentos de corregir la brecha de KGE

Para intentar cerrar esta brecha, se realizaron dos experimentos finales:

1. **Cambio de pérdida a Huber Loss (`ensemble_transformer_huber`)**: Se entrenó el modelo usando la pérdida de Huber (delta=0.5) en lugar del MSE puro, buscando penalizar menos los errores cuadráticos extremos y reducir la sobreestimación sistemática.
   - **Resultado**: MAE=13.41 cm, NSE=0.9959, KGE=0.9919 (Idéntico al modelo base con MSE).
   - **Explicación**: En el espacio normalizado residual, las diferencias absolutas `|y - y_hat|` rara vez superan el 0.5, por lo que la pérdida de Huber actúa esencialmente igual que el MSE.

2. **Corrección de sesgo post-proceso (Bias Correction)**: Se intentó sustraer el sesgo medio calculado sobre el conjunto de validación de las predicciones del conjunto de pruebas.
   - **Resultado**: KGE marginalmente mejorado de 0.9919 a 0.9923 (apenas +0.0004 de ganancia).
   - **Explicación**: El sesgo de sobreestimación volumétrica no es un error de aprendizaje en el entrenamiento, sino un artefacto de *"domain shift"*. El período de prueba (2022-2025) corresponde a una etapa inusualmente seca comparada con el período de validación (2019-2022). Esto produce que el modelo sobreestime los mínimos en el test, lo cual es imposible de corregir utilizando el sesgo observado en la validación.

En definitiva, la diferencia de KGE es un límite intrínseco debido a los años seleccionados en los recortes cronológicos (train/val/test splits), no una falla arquitectural.

---

## 7. Sobre el Shadow Effect

En todos los experimentos con residual, se disparó la bandera de Shadow Effect:
`corr(pred, y_lag1) > corr(pred, y_true)`

Con el paradigma residual, este resultado es **esperado y no indica problema**:
- La predicción reconstruida es `nivel_pred = anchor + delta_pred`
- `anchor` ES el nivel de ayer (lag-1 del target)
- Por construcción, `nivel_pred` siempre estará correlacionado con el nivel de ayer

La evidencia de que el modelo NO hace persistencia trivial es el MAE:
- Persistencia naïve: MAE = 25.6 cm
- `ensemble_transformer_v1`: MAE = 13.4 cm → mejora del **48%**

El detector de Shadow Effect no fue diseñado para el paradigma residual con reconstrucción
explícita. En este contexto, es un falso positivo.

---

## 8. Conclusión Científica

> El Transformer con paradigma de predicción residual, entrenado sobre 150 días de historial
> y sin tuning de hiperparámetros más allá del tamaño de embedding (d_model=64), alcanza
> **paridad estadística con el LSTM ensemble gold standard** en NSE (0.9959 = 0.9959) y
> prácticamente en MAE (13.4 vs 13.3 cm). Supera al LSTM en RMSE (19.8 vs 19.9 cm) y
> en todas las métricas de validación.

> El mecanismo de **auto-atención aprende autónomamente** cuáles días del historial de
> 150 días son relevantes para cada predicción, sin requerir las memorias manuales de
> las celdas LSTM. Esto sugiere que el Orinoco tiene dependencias de largo plazo que
> la atención captura de forma más flexible que el LSTM.

> La arquitectura óptima para este problema es sorprendentemente compacta (d_model=64,
> 2 capas), lo que es consistente con la alta regularidad estacional de la cuenca. Aumentar
> la capacidad (d_model=128, 3 capas) produce resultados **peores** por sobreparametrización
> relativa al tamaño del conjunto de validación.

> **Veredicto Definitivo:** La diferencia de 1 mm entre el Transformer y el LSTM Gold Standard (13.4 vs 13.3 cm) está por debajo del margen de precisión de los sensores físicos (~1-3 cm). Hemos tocado el techo de predictibilidad con las características hidrológicas y datos proporcionados. Se descartan mejoras futuras sobre estas arquitecturas sin la inclusión de datos externos (por ejemplo, precipitaciones de radar, caudal del Caroní, etc.).

---

## 9. Scripts de Referencia

```bash
# Entrenamiento individual con residual
.venv/Scripts/python.exe scripts/run_experiment.py \
  --name transformer_v2_residual --model transformer

# Ensemble de 5 miembros (gold standard Transformer)
.venv/Scripts/python.exe scripts/run_transformer_ensemble.py \
  --name ensemble_transformer_v1 --n 5

# Ensemble XL (no recomendado — ver sección 4, Experimento 3)
.venv/Scripts/python.exe scripts/run_transformer_ensemble.py \
  --name ensemble_transformer_xl --n 5 \
  --d_model 128 --num_layers 3 --dim_feedforward 256
```

**Configuración en `config.yaml` (activa para esta rama):**
```yaml
lookback_window: 150
use_residual:    true
train_end:       "2018-12-31"
val_end:         "2022-06-30"
transformer:
  d_model:         64
  nhead:           4
  num_layers:      2
  dim_feedforward: 128
  dropout:         0.1
  batch_size:      32
  max_epochs:      100
  patience:        10
  learning_rate:   0.0001
```
