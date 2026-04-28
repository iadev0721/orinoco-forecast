# Flujo de Trabajo de Entrenamiento y Evaluación (Orinoco Forecast)

Este documento describe la infraestructura de experimentación estandarizada para el proyecto Orinoco Forecast. Está diseñada para que múltiples miembros del equipo puedan entrenar, evaluar y comparar modelos de forma independiente y reproducible, cumpliendo todas las **AGENT_RULES**.

## Arquitectura de Experimentos

La infraestructura se basa en tres scripts principales ubicados en la carpeta `scripts/`:

1. `run_experiment.py`: Lanza un entrenamiento (Naive, LSTM, etc.) y registra todos los resultados.
2. `evaluate_experiment.py`: Analiza y grafica un experimento individual.
3. `compare_experiments.py`: Genera comparaciones entre múltiples experimentos.

Cada experimento genera una carpeta única en `results/experiments/{nombre_experimento}/` con los siguientes artefactos:
- `metrics.json`: Métricas de evaluación (MAE, RMSE, NSE, KGE).
- `config_used.yaml`: La configuración exacta con la que corrió el modelo.
- `training_history.json`: Curvas de loss (solo modelos Deep Learning).
- `predictions_test.csv`: Predicciones paso a paso para el conjunto de test.

---

## 1. Entrenar un Modelo (`run_experiment.py`)

Cualquier miembro del equipo puede iniciar un experimento. El script se encarga de:
- Fijar seeds de reproducibilidad (NumPy, TF, PyTorch).
- Configurar la memoria de la GPU (VRAM growth).
- Cargar y escalar el dataset respetando los cortes cronológicos.
- Aplicar restricciones físicas (e.g., niveles no negativos).

### Ejemplos de uso:

**A. Ejecutar los Baselines (OBLIGATORIO antes de cualquier LSTM):**
```bash
python scripts/run_experiment.py --name mi_baseline --model naive
```

**B. Entrenar LSTM con la configuración por defecto (`config.yaml`):**
```bash
python scripts/run_experiment.py --name lstm_base --model lstm
```

**C. Entrenar LSTM probando diferentes hiperparámetros (Overrides):**
Puedes modificar parámetros sin tocar el `config.yaml` usando argumentos de terminal:
```bash
python scripts/run_experiment.py --name lstm_lookback45 --model lstm --lookback 45
python scripts/run_experiment.py --name lstm_128_64 --model lstm --units 128 64
python scripts/run_experiment.py --name lstm_lr_lento --model lstm --lr 0.0005
```

---

## 2. Evaluar tu Propio Modelo (`evaluate_experiment.py`)

Si estás trabajando en tu propia rama de Git o quieres ver el rendimiento detallado de tu modelo sin compararlo con otros, usa este script.

**Genera un reporte completo en consola y 5 gráficas científicas en `results/figures/{nombre}/`:**
- `predictions_vs_actual.png`: Serie temporal real vs predicha.
- `scatter_by_regime.png`: Diagrama de dispersión coloreado por régimen hidrológico (aguas bajas, altas, ascenso, descenso).
- `residuals_by_regime.png`: Boxplot del error absoluto por régimen.
- `extreme_events.png`: Zoom detallado en los picos de crecida más extremos.
- `loss_curves.png`: Curvas de validación vs entrenamiento.

### Ejemplos de uso:

**Generar reporte y gráficas:**
```bash
python scripts/evaluate_experiment.py --name lstm_base
```

**Ver solo métricas rápidamente en consola (sin guardar imágenes):**
```bash
python scripts/evaluate_experiment.py --name lstm_base --metrics-only
```

**Evaluar un experimento en una ruta externa (ej. descargado de otro compañero):**
```bash
python scripts/evaluate_experiment.py --path C:/Descargas/experimento_juan/
```

---

## 3. Comparar el Trabajo del Equipo (`compare_experiments.py`)

Cuando se tengan varios experimentos guardados en `results/experiments/`, este script consolida los resultados y permite ver cuál modelo es superior.

**Genera comparativas en `results/figures/`:**
- `comparison_table.csv`: Tabla exportable con MAE, RMSE, NSE y KGE.
- `comparison_metrics.png`: Gráfico de barras de métricas para identificar los modelos que cruzan los umbrales (ej. NSE > 0.80).
- Gráficas conjuntas de series temporales, dispersión y curvas de loss.

### Ejemplos de uso:

**Comparar todos los experimentos:**
```bash
python scripts/compare_experiments.py
```

**Comparar todos y resaltar uno en particular en las gráficas (ej. tu mejor modelo):**
```bash
python scripts/compare_experiments.py --highlight lstm_128_64
```

---

## Estructura de Directorios Resultante

```text
orinoco-forecast/
├── results/
│   ├── experiments/
│   │   ├── mi_baseline/
│   │   │   ├── metrics.json
│   │   │   ├── config_used.yaml
│   │   │   └── predictions_test.csv
│   │   └── lstm_base/
│   │       ├── metrics.json
│   │       ├── config_used.yaml
│   │       ├── training_history.json
│   │       └── predictions_test.csv
│   │
│   ├── models/
│   │   ├── scaler.joblib           <- Escaldor ajustado solo con train
│   │   └── lstm_base_best.keras    <- Pesos del mejor epoch
│   │
│   └── figures/
│       ├── lstm_base/              <- Evaluaciones individuales
│       │   ├── extreme_events.png
│       │   └── ...
│       ├── comparison_table.csv    <- Evaluaciones conjuntas
│       └── comparison_metrics.png
```

## Beneficios para el Equipo
1. **Sin Conflictos de Git:** Cada quien puede experimentar con hiperparámetros vía CLI sin modificar `config.yaml` ni el código del modelo.
2. **Evaluación Transparente:** La métrica `shadow_effect` avisa inmediatamente (BANDERA ROJA) si el modelo hizo trampa (solo copió el valor del día anterior).
3. **Reproducibilidad:** El archivo `config_used.yaml` guardado en cada experimento permite replicar exactamente cómo se obtuvo cada resultado.
