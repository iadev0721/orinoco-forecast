# AGENT_RULES.md — Mandamientos para Agentes Programadores

> **LEER ANTES DE TOCAR CUALQUIER ARCHIVO.**
> Este documento es el contrato entre los agentes de IA y la integridad científica de esta tesis.
> Violar estas reglas invalida métricas, destruye la reproducibilidad y compromete la defensa.

---

## IDENTIDAD DEL CONTEXTO

Estás trabajando en una **tesis de pregrado sobre predicción hidrológica del río Orinoco**.
El código que escribas debe ser **DEFENDIBLE ante un jurado académico**.
Cada decisión debe tener una justificación técnica e hidrológica.
El modelo puede ser imperfecto; el proceso no puede serlo.

---

## ERRORES HEREDADOS (PROHIBIDO REPETIRLOS)

Estos errores fueron detectados en iteraciones anteriores del proyecto.
Son la razón de existir de estas reglas.

### ❌ Error Heredado #1: Data Leakage en el Escalado
El `MinMaxScaler` fue aplicado al dataset completo antes de particionar.
Esto invalida cualquier métrica reportada porque el scaler "vio" los min/max del test set.
**→ Ver R1 para la corrección.**

### ❌ Error Heredado #2: Imputación no auditada
Solo existe **una versión del dataset** (`dataset-orinoco.xlsx`), ya imputada con Simple ML.
El dataset original (con brechas naturales) no está disponible para comparación directa.
Se conocen 17 NaN en `palua` (años bisiestos + 5 días de 1993) que son artefactos de la
imputación. No se sabe qué método usó Simple ML internamente ni si introdujo
autocorrelaciones artificiales en los tramos imputados.
**→ La Fase 0 audita la calidad del único dataset disponible (no compara dos versiones).**

### ❌ Error Heredado #3: Inconsistencia de frameworks
El proyecto osciló entre PyTorch y TensorFlow/Keras sin decisión formal.
**→ Decisión definitiva: LSTM en Keras (core), Transformer en PyTorch (plus).**

### ❌ Error Heredado #4: Modelado sin baseline
Se intentó implementar redes neuronales sin un modelo naive evaluado.
**→ Ver R3 (Baseline Primero).**

---

## REGLAS ABSOLUTAS (NUNCA VIOLAR)

### R1: CERO DATA LEAKAGE 🔒

```
SPLIT CRONOLÓGICO → FIT SCALER (solo train) → TRANSFORM todos → VENTANAS
```

- El `MinMaxScaler` (o cualquier scaler) se ajusta **EXCLUSIVAMENTE** con datos de entrenamiento.
- Fechas de corte:
  - **Train:** 1974-01-01 → 2015-12-31
  - **Validation:** 2016-01-01 → 2020-12-31
  - **Test:** 2021-01-01 → 2025-02-24
- El scaler se guarda con `joblib` en `results/models/scaler.joblib`.
- **PROHIBIDO** usar `df.describe()` o `df.info()` sobre el dataset completo para tomar decisiones de preprocesamiento que afecten al modelo.
- **PROHIBIDO** calcular estadísticas de normalización sobre val o test.

### R2: REPRODUCIBILIDAD 🎲

Toda ejecución debe fijar seeds al inicio:

```python
import numpy as np
import tensorflow as tf
import random
import torch

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

- Documentar versiones en cada notebook: Python, TensorFlow, PyTorch, CUDA, cuDNN.
- Usar `src/utils/reproducibility.py` — no reinventar este bloque.

### R3: BASELINE PRIMERO ⚡

- **NINGÚN** notebook de modelado (04, 05) puede ejecutarse sin que exista `results/metrics/baseline_metrics.json`.
- El LSTM debe SUPERAR al baseline en MAE, RMSE y NSE sobre el set de TEST.
- Si no lo supera, el agente **DEBE diagnosticar POR QUÉ** antes de avanzar.
- Orden obligatorio: Naive → Seasonal Naive → SARIMA → LSTM → Transformer.

### R4: RESPETO FÍSICO 🌊

El río Orinoco tiene restricciones físicas que el modelo DEBE respetar:

```python
# SIEMPRE aplicar en inferencia
prediccion = max(0.0, prediccion)  # Nivel nunca negativo

# Alerta si cambio diario > 1.5 m (físicamente improbable)
if abs(pred[t+1] - pred[t]) > 1.5:
    logging.warning(f"Cambio diario implausible: {pred[t+1]-pred[t]:.2f} m/día")

# Predicciones no exceden el máximo histórico + 15%
MAX_HISTORICAL = df['palua'].max()
prediccion = min(prediccion, MAX_HISTORICAL * 1.15)
```

### R5: GESTIÓN DE MEMORIA GPU 💾

```python
# OBLIGATORIO al inicio de cualquier script con TensorFlow
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Para PyTorch (Transformer)
import torch
torch.cuda.empty_cache()
```

- `batch_size` **MÁXIMO** para Transformers: **32** (RTX 3060, 6GB VRAM)
- `batch_size` **RECOMENDADO** para LSTM: **64**
- Si hay OOM (Out of Memory): reducir batch_size a la mitad, nunca reducir d_model primero.

### R6: CONVENCIONES DE CÓDIGO 📐

- **Idioma del código:** Inglés (variables, funciones, docstrings).
- **Idioma de los comentarios:** Español (para contexto de tesis).
- **Idioma de los nombres de carpetas y archivos:** Inglés (`data/`, `src/`, `models/`, etc.).
- **Type hints** obligatorios en TODAS las funciones.
- **Docstrings** en formato Google style.
- **Logging** con el módulo `logging`, **NUNCA con `print()`**.
- **Formato:** Black + isort (configurados en `.gitignore`, ejecutar antes de commit).

```python
# ✅ CORRECTO
import logging

logger = logging.getLogger(__name__)

def compute_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Nash-Sutcliffe Efficiency.

    Args:
        y_true: Valores observados del nivel del río (metros).
        y_pred: Valores predichos por el modelo (metros).

    Returns:
        NSE score. Rango: (-∞, 1]. Ideal: 1.0.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

# ❌ INCORRECTO
def nse(a, b):
    print("calculando...")
    return 1 - sum((a-b)**2) / sum((a-a.mean())**2)
```

### R7: CONFIGURACIÓN CENTRALIZADA ⚙️

- Todos los hiperparámetros se leen de `config.yaml`.
- **PROHIBIDO hardcodear valores** dentro de los scripts.

```python
# ✅ CORRECTO
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

LOOKBACK = cfg["lookback_window"]      # No hardcodear 30
HORIZON = cfg["forecast_horizon"]      # No hardcodear 7
LSTM_UNITS = cfg["lstm"]["units"]      # No hardcodear [64, 32]

# ❌ INCORRECTO
LOOKBACK = 30
HORIZON = 7
```

### R8: NOMENCLATURA DE ESTACIONES 🗺️

```python
# Usar SIEMPRE estos identificadores. Nunca abreviar diferente.
STATION_ORDER = {
    "ayacucho": {"km_from_source": 0, "order": 0},
    "caicara": {"km_from_source": 500, "order": 1},
    "ciudad_bolivar": {"km_from_source": 900, "order": 2},
    "palua": {"km_from_source": 950, "order": 3},
}

# ✅ CORRECTO: ayacucho, caicara, ciudad_bolivar, palua
# ❌ INCORRECTO: pto_ayacucho, CB, bolivar, paloua
```

---

## PREGUNTAS DE CONTROL POR FASE

Cada notebook **DEBE incluir al final** una sección `## Preguntas de Control` donde el agente responda **explícitamente** a las preguntas definidas en el manifiesto correspondiente de `docs/`.

No omitir, no resumir, no copiar de un notebook a otro.
Respuesta real basada en los datos reales de esa ejecución.

| Fase | Notebook | Manifiesto |
|------|----------|------------|
| Fase 0 | `00_imputation_audit.ipynb` | `docs/00_MANIFESTO_IMPUTATION_AUDIT.md` |
| Fase 1a | `01_eda.ipynb` | `docs/01_MANIFESTO_EDA.md` |
| Fase 1b | `02_feature_engineering.ipynb` | `docs/02_MANIFESTO_PREPROCESSING.md` |
| Fase 3a | `03_baseline.ipynb` | `docs/03_MANIFESTO_MODELING.md` |
| Fase 3b | `04_lstm_keras.ipynb` | `docs/03_MANIFESTO_MODELING.md` |
| Fase 4 | `05_transformer_pytorch.ipynb` | `docs/03_MANIFESTO_MODELING.md` |
| Fase 5 | `06_evaluation_final.ipynb` | `docs/04_MANIFESTO_EVALUATION.md` |

---

## DECISIONES ARQUITECTÓNICAS DEFINITIVAS

| Decisión | Opción elegida | Justificación |
|----------|---------------|---------------|
| Framework LSTM | TensorFlow 2.x + Keras | Rapidez de prototipado, `model.fit()` simplifica tuning |
| Framework Transformer | PyTorch 2.x | Ecosistema nativo, papers en PyTorch |
| Scaler | MinMaxScaler | Preserva distribución, rango [0,1] |
| Split strategy | Cronológico estricto | Serie temporal — no aleatorio |
| Lookback inicial | 30 días | Verificar con cross-correlation en EDA |
| Target (configurable) | Cualquier estación | Definido en `config.yaml → target_station`. No hay "default" fijo: la elección se justifica en la tesis según el problema a resolver. |
| Loss function LSTM | MSE (base) → Pinball (cuantil) | Progresión de simple a probabilístico |

---

## BANDERAS ROJAS 🚩

Si el agente detecta cualquiera de estas situaciones, debe **DETENERSE Y REPORTAR** antes de continuar:

1. El scaler fue fit con datos que incluyen val o test.
2. Hay NaN en los tensores X_train, X_val o X_test.
3. El LSTM NO supera al Naive Baseline en NSE.
4. Las predicciones contienen valores negativos.
5. `results/metrics/baseline_metrics.json` no existe y se intenta entrenar LSTM.
6. Cualquier hiperparámetro está hardcodeado en el script.
7. El train loss sube consistentemente (lr muy alta, gradiente explosivo).
8. La correlación `corr(ŷ(t), y(t-1)) > corr(ŷ(t), y(t))` — shadow effect.
