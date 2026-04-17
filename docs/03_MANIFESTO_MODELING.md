# 🧠 Manifiesto Fase 3: Modelado

> **Prerequisito:** `results/metrics/baseline_metrics.json` debe existir antes de ejecutar
> cualquier notebook de redes neuronales (04, 05).

## Propósito

Justificar y ejecutar la progresión técnica de modelado: del más simple al más complejo.
Cada nivel debe demostrar que supera al anterior. Si no lo demuestra, se documenta por qué.

---

## Progresión de Modelos (ORDEN OBLIGATORIO)

### Nivel 0 — Naive Baseline (Modelo Ingenuo)
```
ŷ(t+h) = y(t)   para h = 1,...,7
```
El nivel en 7 días será igual al nivel de hoy.
**Justificación (Hyndman, 2018):** Todo modelo debe demostrar que supera la predicción más
simple posible. Si no lo logra, el modelo es inútil independientemente de su complejidad.

### Nivel 0.5 — Seasonal Naive
```
ŷ(t+h) = y(t+h - 365)
```
El nivel será igual al del mismo día del año anterior.
Captura la estacionalidad dominante del Orinoco.

### Nivel 1 — SARIMA
Modelo estadístico clásico. Estándar en hidrología (Hyndman, 2018).
Si el LSTM no supera a SARIMA, no se justifica la complejidad computacional de deep learning.

### Nivel 2 — LSTM Multivariado (Keras) ← MODELO CORE
Justificación para este problema:
1. Capturan dependencias a largo plazo (memoria de celda retiene estado del ciclo hidrológico).
2. Procesan entrada multivariada (4 estaciones simultáneas) de forma natural.
3. La compuerta de olvido descarta información irrelevante de meses atrás.
4. Computacionalmente viable en RTX 3060 (< 1 GB VRAM).

Arquitectura recomendada:
```
Input: (batch, 30, n_features)
├── LSTM(64, return_sequences=True) → Dropout(0.2)
├── LSTM(32, return_sequences=False) → Dropout(0.2)
├── Dense(32, activation='relu')
└── Dense(7, activation='linear')   ← Multi-step: 7 días
Output: (batch, 7)
```

### Nivel 3 — Transformer (PyTorch) ← PLUS ACADÉMICO
El mecanismo de self-attention pondera directamente qué días del pasado son más relevantes.
**RIESGO:** Con 6 GB VRAM, batch_size ≤ 32 y d_model ≤ 64.

---

## Preguntas de Control

### PC-03-01: Precedencia del Baseline
```
¿El modelo Naive fue implementado y evaluado ANTES de cualquier red neuronal?
Verificar que results/metrics/baseline_metrics.json existe y contiene:
  naive_mae, naive_rmse, naive_nse
  seasonal_naive_mae, seasonal_naive_rmse, seasonal_naive_nse
  sarima_mae, sarima_rmse, sarima_nse
... por régimen hidrológico.
```

### PC-03-02: Superación del Baseline por el LSTM
```
¿El LSTM supera al baseline en TODAS las métricas (MAE, RMSE, NSE)
sobre el set de TEST?

Si NSE_LSTM > NSE_SARIMA: continuar.
Si NSE_LSTM ≤ NSE_SARIMA: DETENERSE. Diagnosticar antes de avanzar.
Posibles causas: lookback insuficiente, shadow effect, overfitting.
```

### PC-03-03: Detección del Shadow Effect (Lag-1 Copying)
```
¿El modelo predice o copia?
Calcular:
  corr_pred_lag = correlation(ŷ(t), y(t-1))
  corr_pred_real = correlation(ŷ(t), y(t))

Si corr_pred_lag > corr_pred_real:
  El modelo está copiando el último valor conocido, no prediciendo.
  ESTO ES UNA BANDERA ROJA. Ver AGENT_RULES.md R4.
```

### PC-03-04: Sensibilidad al Lookback Window
```
¿Hay un punto de rendimiento decreciente al variar el lookback?
Experimentar con lookback ∈ [14, 21, 30, 45, 60] días.
Graficar: NSE_val vs lookback_window.
Registrar el lookback óptimo y actualizar config.yaml si difiere.
```

### PC-03-05: Impacto de la Multivariabilidad
```
¿Las estaciones aguas arriba realmente ayudan?
Comparar:
  LSTM univariado: solo Palúa como input
  LSTM multivariado: Ayacucho + Caicara + Ciudad Bolívar + Palúa

Reportar: diferencia en NSE y MAE.
Si la diferencia es < 2%, las estaciones "radar" no aportan
información adicional. Esto sería un hallazgo importante para la tesis.
```

### PC-03-06: Curvas de Aprendizaje
```
¿Las curvas de aprendizaje muestran overfitting?
Graficar: train_loss vs val_loss por época.
¿En qué época se activó EarlyStopping?
¿La brecha train/val es preocupante?

Si val_loss crece mientras train_loss decrece → overfitting.
Mitigaciones: aumentar dropout, reducir unidades, más datos.
```

---

## Métricas Obligatorias por Modelo

Cada modelo debe reportar la siguiente tabla completa:

| Métrica | Global | Aguas Bajas | Ascenso | Aguas Altas | Descenso |
|---------|--------|-------------|---------|-------------|----------|
| MAE (m) | — | — | — | — | — |
| RMSE (m) | — | — | — | — | — |
| NSE | — | — | — | — | — |
| KGE | — | — | — | — | — |

**Umbrales de éxito:**
- NSE > 0.80 (aceptable), NSE > 0.90 (bueno)
- KGE > 0.75
- Peak Timing Error < 3 días

---

## Entregables

- `notebooks/03_baseline.ipynb` + `results/metrics/baseline_metrics.json`
- `notebooks/04_lstm_keras.ipynb` + `results/models/lstm_model.h5`
- `notebooks/05_transformer_pytorch.ipynb` + `results/models/transformer_model.pt`
- `results/metrics/all_models_comparison.csv` — tabla completa por modelo y régimen
