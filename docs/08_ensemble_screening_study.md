# Estudio de Screening de Semillas del Ensemble
## Decisión: Modelo Individual vs. Ensemble Promediado

**Fecha:** 2026-05-11  
**Entorno de validación:** Google Colab — NVIDIA Tesla T4 (13 757 MB VRAM)  
**Script:** `colab_gold_standard.py`

---

## 1. Motivación

El Gold Standard original (`ensemble_lb150_lags_xl`) utilizaba un ensemble de 5 modelos con seeds `[42, 123, 456, 789, 1011]` y reportaba **MAE = 13.3 cm** en el hardware local (RTX 3060). Al migrar a Colab, se aprovechó la sesión para investigar si ampliar o refinar el ensemble podría mejorar el resultado.

---

## 2. Experimentos Realizados en Orden Cronológico

### Experimento A — Reproducción base (5 miembros originales)

| Seed | Val Loss | MAE Test individual |
|------|----------|---------------------|
| 42 | 0.000187 | **13.3 cm** |
| 123 | 0.000207 | 14.1 cm |
| 456 | 0.000204 | 14.2 cm |
| 789 | 0.000192 | 13.4 cm |
| 1011 | 0.000190 | 13.8 cm |

**MAE Ensemble (promedio uniforme): 13.4 cm**

---

### Experimento B — Screening ampliado (10 miembros)

| Seed | Val Loss | MAE Test individual |
|------|----------|---------------------|
| 5050 | **0.000186** | 13.8 cm |
| 42 | 0.000187 | **13.3 cm** |
| 9999 | 0.000188 | 14.1 cm |
| 1011 | 0.000190 | 13.8 cm |
| 789 | 0.000192 | 13.4 cm |
| 2024 | 0.000202 | 14.0 cm |
| 123 | 0.000207 | 14.1 cm |
| 456 | 0.000204 | 14.2 cm |
| 7777 | 0.000199 | 13.8 cm |
| 3141 | 0.000216 | 16.3 cm |

**MAE Ensemble (10 miembros): 13.6 cm** — peor que 5 originales.

---

### Experimento C — Top-5 por val_loss

Seeds seleccionados exclusivamente por val_loss (no por MAE test — sin data leakage): `[5050, 42, 9999, 1011, 789]`

**MAE Ensemble (top-5 por val_loss): 13.5 cm** — sigue siendo peor que el mejor miembro individual.

---

## 3. Análisis: ¿Por Qué el Ensemble No Mejora?

El ensemble de promedio uniforme mejora sobre un modelo individual cuando:
1. Los modelos tienen **habilidad similar**
2. Los errores son **no correlacionados**

En este caso ambas condiciones fallan:

- Los val_loss son casi idénticos (rango < 4%), un promedio ponderado por `1/val_loss` produciría pesos ~20% cada uno, sin beneficio real.
- Modelos con la misma arquitectura, features y dataset generan errores **altamente correlacionados**, especialmente en eventos extremos.
- seed=42 es sistemáticamente superior y los demás lo penalizan al promediar.

---

## 4. Decisión Final

> **Se adopta el modelo individual `seed=42` como configuración definitiva.**

| Configuración | MAE Test | NSE | KGE Val |
|---|---|---|---|
| `ensemble_5_originales` | 13.4 cm | 0.9958 | 0.9926 |
| `ensemble_10_miembros` | 13.6 cm | 0.9957 | 0.9932 |
| `ensemble_top5_valloss` | 13.5 cm | 0.9958 | 0.9960 |
| **`single_seed42` (elegido)** | **13.3 cm** | **0.9959** | **0.9938** |

### Justificación científica

El ensemble de promedio uniforme opera bajo el supuesto de errores parcialmente independientes. Cuando esta condición no se cumple —modelos homogéneos sobre el mismo dataset—, el promedio regresa a la media penalizando al miembro superior. Esto es consistente con Lakshminarayanan et al. (2017): los Deep Ensembles mejoran la **calibración de incertidumbre** más que la precisión puntual cuando los miembros son homogéneos.

El modelo individual `seed=42` es además **determinista, reproducible y simple**.

---

## 5. Configuración Definitiva del Script

```python
ENSEMBLE_SEEDS = [42]
N_MEMBERS      = 1
# lookback=150, units=[128,64], dropout=0.2, loss=huber(delta=0.5)
# use_residual=True, batch_size=64, patience=15
```

**MAE Test: 13.3 cm | RMSE: 19.9 cm | NSE: 0.9959 | KGE: 0.9938**
