# Estándar de Salida de Experimentos — Orinoco Forecast

> **Propósito:** Este documento define el contrato que todo experimento
> (LSTM, Transformer, baseline, etc.) **debe cumplir obligatoriamente**
> para que `compare_experiments.py` y `diagnose_experiment.py` funcionen
> correctamente y los resultados sean comparables entre sesiones y autores.

---

## Estructura de Directorios

Cada experimento genera exactamente **una carpeta** bajo `results/experiments/`:

```
results/
└── experiments/
    └── <nombre_experimento>/        ← carpeta raíz del experimento
        ├── metrics.json             ← OBLIGATORIO
        ├── training_history.json    ← OBLIGATORIO (modelos entrenados)
        ├── predictions_test.csv     ← OBLIGATORIO
        ├── config_used.yaml         ← OBLIGATORIO
        └── diagnosis.json           ← GENERADO por diagnose_experiment.py
```

> **Convención de nombres:** El nombre debe ser descriptivo y único.
> Usar `snake_case`. Ejemplos: `lstm_base`, `lstm_xl`, `transformer_v1`,
> `baseline_naive`, `lstm_guri_dahiti`.

---

## Archivos Obligatorios

### 1. `metrics.json`

Contiene todas las métricas de rendimiento en los tres particionados.

**Schema exacto:**

```json
{
  "experiment_name": "lstm_xl",           // str: nombre del experimento
  "model_type": "lstm",                   // str: "lstm" | "transformer" | "naive" | otro
  "timestamp": "2026-04-28T19:42:28.553", // str: ISO 8601

  "metrics": {
    "test": {
      "mae":       0.4275,    // float: Error Absoluto Medio (metros)
      "rmse":      0.5419,    // float: Raíz del Error Cuadrático Medio (metros)
      "nse":       0.9746,    // float: Nash-Sutcliffe Efficiency [-∞, 1]
      "kge":       0.9070,    // float: Kling-Gupta Efficiency [-∞, 1]
      "kge_r":     0.9908,    // float: componente correlación de KGE
      "kge_alpha": 0.9078,    // float: componente variabilidad de KGE
      "kge_beta":  0.9924     // float: componente sesgo de KGE
    },
    "val": {
      // Mismas claves que "test"
    }
    // "train" es opcional; se omite si no se calculan métricas en train
  },

  "training": {
    "epochs_trained":      20,          // int: épocas completadas (incl. early stopping)
    "best_epoch":           5,          // int: época con mejor val_loss
    "final_train_loss": 0.000579,       // float: loss en la última época
    "final_val_loss":   0.001016,       // float: val_loss en la última época
    "best_val_loss":    0.000783        // float: mejor val_loss observado
  }
}
```

**Reglas:**
- `nse` y `kge` deben calcularse sobre predicciones **desnormalizadas** (metros reales).
- Para modelos sin entrenamiento (ej. `naive`), el bloque `"training"` puede omitirse.
- **Nunca** reportar métricas sobre datos normalizados.

---

### 2. `training_history.json`

Historial completo de métricas por época para graficar las curvas de aprendizaje.

**Schema exacto:**

```json
{
  "loss":     [0.0262, 0.0064, 0.0045, ...],  // list[float]: train loss por época
  "val_loss": [0.0024, 0.0018, 0.0022, ...],  // list[float]: val loss por época
  "mae":      [0.1067, 0.0588, 0.0499, ...],  // list[float]: train MAE (norm.) por época
  "val_mae":  [0.0374, 0.0326, 0.0376, ...],  // list[float]: val MAE (norm.) por época
  "lr":       [0.001,  0.001,  0.001,  ...]   // list[float]: learning rate por época
}
```

**Reglas:**
- Todas las listas deben tener la **misma longitud** (igual a `epochs_trained`).
- `loss` y `val_loss` deben ser en escala normalizada (la que optimiza el modelo).
- No incluir épocas que no se ejecutaron (no rellenar con ceros).
- Para modelos sin entrenamiento iterativo (ej. `naive`), este archivo **no es requerido**.

---

### 3. `predictions_test.csv`

Predicciones diarias del modelo sobre el conjunto de test, en escala real (metros).

**Schema exacto:**

```csv
fecha,y_true,y_pred
2018-10-11,10.32,10.2657175
2018-10-12,10.20,10.178893
...
2025-02-24,X.XX,X.XXXXXXX
```

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `fecha` | `YYYY-MM-DD` | Fecha de la predicción |
| `y_true` | `float` | Nivel observado real en Palúa (metros) |
| `y_pred` | `float` | Nivel predicho por el modelo (metros) |

**Reglas:**
- Cubrir **todo** el período de test sin interrupciones.
- Fechas en formato `YYYY-MM-DD`, ordenadas cronológicamente.
- Sin índice numérico adicional (la fecha es el índice lógico).
- `y_pred` ya desnormalizado al espacio físico original.
- Precisión mínima: 6 decimales.

> **Nota sobre horizontes multi-paso:** Si el modelo genera `horizon=7`
> (predicciones para 7 días), se reporta **un único valor por día**
> (la predicción del día más próximo al step correspondiente).
> La implementación actual en `run_experiment.py` maneja este aplanamiento.

---

### 4. `config_used.yaml`

Copia exacta del objeto de configuración que se usó en el experimento.
Incluye **todos los hiperparámetros**, incluyendo los overrides de CLI.

```yaml
target_station: palua
forecast_horizon: 7
lookback_window: 30
train_end: '2012-03-16'
val_end:   '2018-09-04'

lstm:
  units: [256, 128, 64]   # puede diferir del config.yaml base
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 32
  max_epochs: 200
  patience: 15

seed: 42
# ... resto de la config completa
```

**Reglas:**
- Debe ser reproducible: con este archivo + el código, el experimento
  debe poder repetirse y dar ≤ 1% de variación en NSE.
- Incluir los valores **reales usados**, no los valores por defecto.
  Si se pasó `--lookback 14`, el YAML debe decir `lookback_window: 14`.

---

### 5. `diagnosis.json`

Generado automáticamente por `python scripts/diagnose_experiment.py --name <nombre>`.
**No se genera durante el entrenamiento.**

```json
{
  "experiment":   "lstm_xl",
  "verdict":      "DISTRIBUTION_SHIFT",   // "UNDERFITTING" | "OVERFITTING" |
                                           // "DISTRIBUTION_SHIFT" | "GOOD" | "INDETERMINATE"
  "confidence":   "MEDIA",                // "ALTA" | "MEDIA" | "BAJA"
  "evidence":     ["NSE Val = 0.9923", "NSE Test = 0.9746", "..."],
  "warnings":     ["⚠ INFORMATION CEILING: ..."],
  "recommendation": "Texto libre explicando...",

  "metrics_summary": {
    "nse_val":       0.9923,
    "nse_test":      0.9746,
    "nse_gap":       0.0177,
    "mae_val":       0.222,
    "mae_test":      0.428,
    "kge_test":      0.907,
    "loss_ratio":    1.755,
    "best_epoch":    5,
    "epochs_trained": 20,
    "epoch_ratio":   0.25
  }
}
```

---

## Herramientas Automatizadas

### `python scripts/run_experiment.py`

Genera automáticamente los archivos 1–4 al finalizar el entrenamiento.
Flags de uso habitual:

```bash
# Experimento estándar
python scripts/run_experiment.py --name <nombre> --model lstm

# Sobreescribir hiperparámetros
python scripts/run_experiment.py --name lstm_xl --model lstm \
    --units 256 128 64 --batch 32 --lookback 30
```

### `python scripts/diagnose_experiment.py`

Genera el archivo 5 (`diagnosis.json`) y muestra el diagnóstico en consola.

```bash
# Un experimento específico
python scripts/diagnose_experiment.py --name lstm_xl

# Todos los experimentos a la vez
python scripts/diagnose_experiment.py
```

### `python scripts/compare_experiments.py`

Lee todos los `metrics.json` y genera:

| Archivo de salida | Descripción |
|---|---|
| `results/figures/comparison_table.csv` | Tabla maestra ordenada por NSE Test |
| `results/figures/comparison_metrics.png` | Barras comparativas MAE/RMSE/NSE/KGE |
| `results/figures/predictions_test.png` | Series temporales de predicciones |
| `results/figures/loss_curves.png` | Curvas de aprendizaje por experimento |
| `results/figures/error_by_regime.png` | Error por régimen hidrológico |
| `results/figures/scatter_plot.png` | Dispersión y_true vs y_pred |

---

## Referencia: Tabla de Experimentos Actuales

| Experimento | Tipo | NSE Test | MAE (m) | Diagnóstico |
|---|---|---|---|---|
| `baseline_naive` | naive | **0.9892** 🏆 | 0.259 | — (baseline) |
| `lstm_xl` | lstm | 0.9746 | 0.428 | Distribution Shift |
| `lstm_lookback14` | lstm | 0.9722 | 0.442 | Distribution Shift |
| `lstm_heavy` | lstm | 0.9716 | 0.440 | Distribution Shift |
| `lstm_guri_dahiti` | lstm | 0.9633 | 0.498 | Distribution Shift |
| `lstm_base` | lstm | 0.9588 | 0.517 | Distribution Shift |
| `lstm_lookback90` | lstm | 0.9563 | 0.564 | Distribution Shift |
| `lstm_gpu` | lstm | 0.9547 | 0.558 | Overfitting |

> **Todos los modelos LSTM presentan Distribution Shift** en el período de
> test (2018–2025). El modelo campeón es `lstm_xl` (NSE=0.9746), aunque
> aún no supera el baseline de persistencia (NSE=0.9892).
> La causa identificada es la ausencia de datos de caudal de descarga de
> la presa Guri (Corpoelec/MINAGUAS), que domina la dinámica del río
> en el período de test.

---

## Checklist para Nuevos Experimentos

Antes de hacer `git commit` de un nuevo experimento, verificar:

- [ ] La carpeta existe en `results/experiments/<nombre>/`
- [ ] `metrics.json` contiene métricas de **test** y **val** en metros reales
- [ ] `predictions_test.csv` cubre todo el período de test sin NaNs
- [ ] `config_used.yaml` refleja los hiperparámetros reales usados
- [ ] Se ejecutó `diagnose_experiment.py` y existe `diagnosis.json`
- [ ] El NSE Test fue comparado con el baseline (0.9892) en los logs
- [ ] El experimento aparece correctamente en `compare_experiments.py`
