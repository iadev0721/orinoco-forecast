# 🌊 Orinoco Forecast: Predicción del Nivel del Río Orinoco

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iadev0721/orinoco-forecast/blob/main/Orinoco_Pipeline_Final.ipynb)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)](https://mlflow.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Visión Científica

Modelado predictivo multiestación del nivel hidrométrico del río Orinoco con horizonte de **7 días (t+7)**, utilizando una arquitectura progresiva:

> **Baseline Estadístico → LSTM (Keras) → Transformer (PyTorch)**

## Hipótesis de Tesis

> Los niveles hidrométricos aguas arriba del Orinoco contienen información predictiva suficiente
> para anticipar el nivel en **la estación target elegida** con un horizonte de 7 días, superando
> significativamente un modelo naive y un baseline estadístico, particularmente durante los
> **períodos de transición hidrológica** (ascenso mayo-julio, descenso octubre-diciembre).
> La elección de la estación target y su justificación hidrológica forma parte del análisis de la tesis.

## Dataset

| Campo | Detalle |
|-------|---------|
| Rango temporal | 1974-01-01 a 2025-02-24 (~18,683 registros diarios) |
| Estaciones | Puerto Ayacucho · Caicara · Ciudad Bolívar · Palúa |
| Formato Oficial | `data/raw/dataset_orinoco_true_raw.csv` (CSV, cronológico) |
| Integridad | Caicara (0 NaNs), C. Bolívar (0 NaNs), Palúa (17 NaNs), Ayacucho (700 NaNs) |
| Target del experimento | Configurable — cualquier estación del dataset |

## Hardware

- GPU: NVIDIA RTX 3060 (6 GB VRAM)
- Restricción: `batch_size ≤ 32` para Transformers, `batch_size ≤ 64` para LSTM

## Plan de Ejecución

```
Fase 0 → Auditoría de imputación  (¿la señal sobrevivió?)          [BLOQUEANTE]
Fase 1 → EDA y Feature Engineering
Fase 2 → Preprocessing (pipeline anti-leakage)
Fase 3a → Baseline (Naive + SARIMA)                                 [BLOQUEANTE para LSTM]
Fase 3b → LSTM Multivariado Multi-Step (Keras)
Fase 4 → Transformer (PyTorch)                                      [Plus académico]
Fase 5 → Evaluación comparativa por régimen hidrológico
```

## Partición Temporal (CRONOLÓGICA — nunca aleatoria)

| Split | Rango | Propósito |
|-------|-------|-----------|
| Train | 1974-01-01 — 2015-12-31 | Ajuste de scaler + entrenamiento |
| Validation | 2016-01-01 — 2020-12-31 | Early stopping, tuning |
| Test | 2021-01-01 — 2025-02-24 | Evaluación final (intocable) |

## Estructura del Repositorio

```
orinoco-forecast/
│
├── README.md                          # Este archivo
├── AGENT_RULES.md                     # Mandamientos para agentes de IA
├── config.yaml                        # Configuración central del experimento
├── requirements.txt                   # Stack tecnológico
├── .gitignore                         # Excluye datos pesados y entornos
│
├── data/
│   ├── raw/                           # Datos crudos (INTOCABLES)
│   ├── processed/                     # Resultados de auditoría y FE
│   └── external/                      # ENSO, CHIRPS (datos auxiliares)
│
├── notebooks/
│   ├── 00_imputation_audit.ipynb      # Fase 0: Auditoría de imputación
│   ├── 01_eda.ipynb                   # Fase 1: Análisis Exploratorio
│   ├── 02_feature_engineering.ipynb   # Fase 1: Ingeniería de características
│   ├── 03_baseline.ipynb              # Fase 3a: Naive + SARIMA
│   ├── 04_lstm_keras.ipynb            # Fase 3b: LSTM
│   ├── 05_transformer_pytorch.ipynb   # Fase 4: Transformer (Plus)
│   └── 06_evaluation_final.ipynb      # Fase 5: Evaluación comparativa
│
├── src/
│   ├── data/                          # Pipeline de datos (anti-leakage)
│   ├── features/                      # Feature engineering
│   ├── models/                        # Implementaciones de modelos
│   ├── evaluation/                    # Métricas NSE, KGE, visualización
│   └── utils/                         # GPU config, reproducibilidad
│
├── docs/                              # Manifiestos con preguntas de control
│   └── 05_TRAINING_WORKFLOW.md        # Documentación de scripts de entrenamiento
├── tests/                             # Tests de integridad del pipeline
└── results/                           # Métricas, figuras, pesos (en .gitignore)
```

## Flujo de Trabajo y Entrenamiento

El repositorio cuenta con una infraestructura estandarizada para entrenar, evaluar y comparar modelos. Está diseñada para evitar conflictos entre miembros del equipo y facilitar la reproducibilidad. 

Por favor, lee la guía detallada: **[`docs/05_TRAINING_WORKFLOW.md`](./docs/05_TRAINING_WORKFLOW.md)**.

### Resumen de comandos
```bash
# Ejecutar un experimento (ej: baseline naive)
python scripts/run_experiment.py --name mi_baseline --model naive

# Entrenar un LSTM con overrides
python scripts/run_experiment.py --name lstm_128_64 --model lstm --units 128 64

# Evaluar un entrenamiento individual (genera reporte y 5 figuras científicas)
python scripts/evaluate_experiment.py --name lstm_128_64

# Comparar todos los experimentos del equipo
python scripts/compare_experiments.py
```

## Configuración Inicial del Entorno (Humanos y Agentes)

Para asegurar la reproducibilidad y evitar conflictos de dependencias (ver Regla R9), sigue estos pasos rigurosos para configurar tu entorno local y tu IDE (ej. VS Code, Cursor):

### 1. Preparar el repositorio y el entorno virtual
```bash
# Clonar el repositorio
git clone https://github.com/iadev0721/orinoco-forecast.git
cd orinoco-forecast

# Crear el entorno virtual en la carpeta raíz
python -m venv .venv

# Activar el entorno virtual
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
```

### 2. Instalar dependencias y pre-commits
Con el entorno virtual activado (`(.venv)` debe aparecer en tu terminal):
```bash
# Instalar todas las dependencias del proyecto
pip install -r requirements.txt

# Configurar los hooks de pre-commit para calidad de código
pre-commit install
```

### 3. Configuración del IDE (VS Code / Cursor)
Para trabajar correctamente con los notebooks y scripts:
1. **Extensiones necesarias:** Asegúrate de instalar las extensiones oficiales de **Python** y **Jupyter**.
2. **Selección de Intérprete Python:** Presiona `Ctrl+Shift+P` (o `Cmd+Shift+P`), busca `Python: Select Interpreter` y elige el que está dentro de la carpeta `.venv`.
3. **Selección de Kernel para Notebooks:** Al abrir un archivo `.ipynb` (como el de la Fase 0), haz clic en el selector de kernel arriba a la derecha y selecciona **"Python Environments"** -> **".venv"**. Esto asegura que Jupyter use las dependencias correctas (pandas, seaborn, statsmodels).

### 4. Datos
Colocar el dataset original en la ruta esperada:
- `data/raw/dataset_orinoco_true_raw.csv` (archivo crudo verificado, con NaNs reales)

### 5. Primer Paso de Ejecución
```bash
# Iniciar la Fase 0
jupyter notebook notebooks/00_imputation_audit.ipynb
```

## Métricas de Éxito

| Métrica | Criterio Mínimo | Criterio Bueno |
|---------|----------------|----------------|
| NSE (Nash-Sutcliffe) | > 0.80 | > 0.90 |
| KGE (Kling-Gupta) | > 0.75 | > 0.85 |
| Peak Timing Error | < 3 días | < 1 día |

## Reglas para Agentes de IA

Ver [`AGENT_RULES.md`](./AGENT_RULES.md) — contiene los mandamientos que todo agente de IA debe respetar antes de tocar cualquier archivo de este repositorio.

## Licencia

Copyright 2025 — Proyecto de Tesis de Pregrado.

Distribuido bajo la licencia **Apache 2.0**. Ver [`LICENSE`](./LICENSE) para más detalles.

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

> *"La diferencia entre un proyecto que aprueba y una tesis que marca precedente no está en la complejidad del modelo. Está en el rigor del proceso."*
