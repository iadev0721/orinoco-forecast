# 🌊 Orinoco Forecast: Predicción del Nivel del Río Orinoco

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
| Versión disponible | Única versión imputada con Simple ML (`dataset-orinoco.xlsx`) |
| Dataset original | **No disponible** para comparación directa |
| NaN conocidos | 17 en columna `palua`: todos los 29 de febrero (bisiestos 1976-2020) y 5 días en 1993 — artefacto de la imputación |
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
├── tests/                             # Tests de integridad del pipeline
└── results/                           # Métricas, figuras, pesos (en .gitignore)
```

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/iadev0721/orinoco-forecast.git
cd orinoco-forecast

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar el dataset en data/raw/
# - dataset-orinoco.xlsx  (archivo único, ya imputado con Simple ML)

# 5. Ejecutar desde Fase 0
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
