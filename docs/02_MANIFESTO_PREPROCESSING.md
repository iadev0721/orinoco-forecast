# ⚙️ Manifiesto Fase 2: Preprocessing y Feature Engineering

> **Prerequisito:** Fase 1 completada. `results/metrics/eda_lag_times.json` debe existir.

> **PRINCIPIO FUNDAMENTAL:** El pipeline de features debe construirse dinámicamente utilizando el `target_station` definido en `config.yaml`. Está estrictamente prohibido hardcodear nombres de estaciones (como 'palua') en las operaciones de feature engineering.

## Propósito

Transformar los datos crudos en tensores listos para el modelado **sin introducir sesgo**.
El pipeline debe ser reproducible, auditado y guardado para su reutilización en inferencia.

---

## El Pipeline Anti-Leakage (ORDEN OBLIGATORIO)

```
DATASET COMPLETO (1974-2025)
          │
    SPLIT CRONOLÓGICO  ← PRIMER PASO, antes de TODO
          │
    ┌─────┼─────────┐
    ▼     ▼         ▼
  TRAIN  VAL       TEST
  74-15  16-20     21-25
    │
  FIT Scaler (SOLO sobre TRAIN)
    │     │         │
    ▼     ▼         ▼
 TRANSFORM TRANSFORM TRANSFORM
    │     │         │
    ▼     ▼         ▼
 VENTANAS (X, y) por split
```

**PROHIBIDO alternar estos pasos. El scaler se fit UNA VEZ sobre TRAIN y se aplica a todos.**

---

## Preguntas de Control

### PC-02-01: Verificación de No-Leakage del Scaler
```
¿El scaler se ajustó SOLO con datos de train?
Verificar programáticamente:
  assert scaler.data_min_.min() >= df_train.min().min() * 0.99
  assert scaler.data_max_.max() <= df_train.max().max() * 1.01
  
Reportar: rango [min, max] del scaler vs rango real del train set.
```

### PC-02-02: Integridad de Bordes Train/Val/Test
```
¿Las ventanas en los bordes están correctamente manejadas?
La última ventana de TRAIN NO debe contener días de VAL como target.
La última ventana de VAL NO debe contener días de TEST como target.

Verificar:
  last_train_target_date == train_end_date  # "2015-12-31"
  first_val_input_date >= train_end_date - lookback  # solapamiento OK
```

### PC-02-03: Suficiencia del Lookback Window
```
¿La ventana de entrada (lookback) captura el lag máximo del sistema?
Si Origen→Target tiene lag de X días (de EDA), lookback debe ser ≥ X.

Verificar contra eda_lag_times.json:
  assert config['lookback_window'] >= eda_results['source_to_target_total_days']
```

### PC-02-04: Features Temporales Cíclicas
```
¿Se incluyeron features temporales correctamente codificadas?
Obligatorias:
  day_sin = sin(2π * dayofyear / 365.25)
  day_cos = cos(2π * dayofyear / 365.25)

NOTA: NO incluir el año como feature (leakage temporal de tendencia).
Verificar que las features cíclicas están incluidas en el tensor X.
```

### PC-02-05: Ausencia de NaN en Tensores Finales
```
Verificar integridad de todos los tensores:
  assert np.isnan(X_train).sum() == 0, "NaN en X_train"
  assert np.isnan(y_train).sum() == 0, "NaN en y_train"
  assert np.isnan(X_val).sum() == 0, "NaN en X_val"
  assert np.isnan(y_val).sum() == 0, "NaN en y_val"
  assert np.isnan(X_test).sum() == 0, "NaN en X_test"
  assert np.isnan(y_test).sum() == 0, "NaN en y_test"
```

---

## Features Derivadas Obligatorias

```python
# target = config['target_station']
# upstream = ... # estación adyacente anterior según topología

# 1. Codificación cíclica del día del año
df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

# 2. Tasa de cambio (velocidad del río)
df[f'{target}_delta_1d'] = df[target].diff(1)   # ¿subiendo o bajando?
df[f'{target}_delta_7d'] = df[target].diff(7)   # tendencia semanal

# 3. Estadísticas rodantes (rolling features)
df[f'{target}_rolling_mean_7'] = df[target].rolling(7).mean()
df[f'{target}_rolling_std_7'] = df[target].rolling(7).std()

# 4. Ratio entre estaciones (proxy de gradiente hidráulico)
# Asumiendo upstream_station la estación anterior en la topología
df[f'ratio_{upstream}_{target}'] = df[upstream] / (df[target] + 0.01)

# 5. Anomalía respecto al ciclo medio histórico
#    NOTA: La climatología se calcula SOLO sobre el train set.
climatology_train = df_train.groupby(df_train.index.dayofyear)[target].mean()
df[f'{target}_anomaly'] = df[target] - df.index.dayofyear.map(climatology_train)
```

> **Pregunta abierta para el agente:** ¿Estas features van dentro de la ventana (columnas
> adicionales del tensor 3D) o como vector estático concatenado al output del LSTM antes
> de la capa Dense? Documentar la decisión y su justificación.

---

## Proxy de Afluente Externo (Feature Opcional)

```python
# Si dos estaciones cercanas tienen una diferencia que fluctúa, 
# puede ser causada por un aporte externo (ej: el Caroní para Palúa).

# alpha y beta se estiman SOLO con datos de aguas bajas del train set
from sklearn.linear_model import LinearRegression

mask_low_water_train = (df_train[target] < df_train[target].quantile(0.25))
X_calib = df_train.loc[mask_low_water_train, [upstream]]
y_calib = df_train.loc[mask_low_water_train, target]

calib_model = LinearRegression().fit(X_calib, y_calib)
alpha = calib_model.coef_[0]
beta = calib_model.intercept_

df[f'tributary_proxy_{target}'] = df[target] - (df[upstream] * alpha + beta)
```

---

## Entregable

- **Notebook:** `notebooks/02_feature_engineering.ipynb` (ejecutado)
- **Artefactos guardados:**
  - `results/models/scaler.joblib` — scaler ajustado sobre train
  - `results/models/climatology_train.pkl` — media diaria del train set
  - `data/processed/dataset_final.parquet` — dataset con todas las features
- **Dimensiones reportadas:**
  ```
  X_train shape: (N_train, lookback, n_features)
  X_val shape:   (N_val, lookback, n_features)
  X_test shape:  (N_test, lookback, n_features)
  y_train shape: (N_train, horizon)
  ```
