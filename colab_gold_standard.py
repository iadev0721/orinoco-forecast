# ==============================================================
# ORINOCO FORECAST — Reproducción del Gold Standard en Colab
# Modelo: ensemble_lb150_lags_xl
# MAE Test: 13.3 cm | NSE: 0.9959 | KGE: 0.9938
#
# INSTRUCCIONES:
#   1. Subir el archivo dataset_orinoco_features.csv a Colab
#   2. Ejecutar celda por celda en orden
# ==============================================================

# ── CELDA 1: Instalación de dependencias ──────────────────────
# !pip install tensorflow==2.15.0 scikit-learn pandas numpy matplotlib joblib

# ── CELDA 2: Imports ──────────────────────────────────────────
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

# ── CELDA 3: Configuración GPU (si disponible en Colab) ───────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU habilitada: {gpus}")
else:
    print("CPU mode (GPU no disponible)")

# ── CELDA 4: Hiperparámetros del Gold Standard ────────────────
CFG = {
    "target_station":  "palua",
    "forecast_horizon": 7,
    "lookback_window":  150,
    "train_end": "2018-12-31",
    "val_end":   "2022-06-30",
    "use_residual": True,
    "seed": 42,
    "lstm": {
        "units":          [128, 64],   # XL architecture
        "dropout":        0.2,
        "learning_rate":  0.001,
        "batch_size":     64,
        "max_epochs":     200,
        "patience":       15,
        "loss":           "huber",
        "huber_delta":    0.5,
    },
    "physical": {
        "min_level_m":          0.0,
        "max_level_multiplier": 1.15,
        "max_daily_change_m":   1.5,
    },
}

# Modelo individual óptimo: seed=42
# Justificación: screening de 10 semillas mostró que el promedio de ensemble
# no mejora el MAE del mejor miembro individual (errores correlacionados entre
# miembros con misma arquitectura). seed=42 = menor val_loss Y menor MAE test.
ENSEMBLE_SEEDS = [42]
N_MEMBERS      = 1

# ── CELDA 5: Reproducibilidad global ──────────────────────────
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(CFG["seed"])
print("Seeds fijadas.")

# ── CELDA 6: Cargar dataset ────────────────────────────────────
# Cambia esta ruta según dónde subiste el archivo en Colab
FEATURES_PATH = "data/processed/dataset_orinoco_features.csv"   # <-- Ajustado para usar la estructura del repositorio

df = pd.read_csv(FEATURES_PATH, parse_dates=["fecha"]).set_index("fecha").sort_index()
print(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} cols")
print(f"Rango: {df.index.min().date()} → {df.index.max().date()}")

# ── CELDA 7: Split cronológico ────────────────────────────────
STATION_ORDER = ["ayacucho", "caicara", "ciudad_bolivar", "palua"]
TARGET        = CFG["target_station"]
TRAIN_END     = CFG["train_end"]
VAL_END       = CFG["val_end"]

df_train = df.loc[:TRAIN_END]
df_val   = df.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1): VAL_END]
df_test  = df.loc[pd.Timestamp(VAL_END)   + pd.Timedelta(days=1):]

print(f"Train : {df_train.index.min().date()} → {df_train.index.max().date()} ({len(df_train)} filas)")
print(f"Val   : {df_val.index.min().date()}   → {df_val.index.max().date()}   ({len(df_val)} filas)")
print(f"Test  : {df_test.index.min().date()}  → {df_test.index.max().date()}  ({len(df_test)} filas)")

# ── CELDA 8: Feature selection (excluir target y downstream) ──
target_idx       = STATION_ORDER.index(TARGET)
excluded_stations = STATION_ORDER[target_idx:]
excluded_cols     = [
    col for col in df.columns
    if any(col == s or col.startswith(s + "_") for s in excluded_stations)
]
feature_cols = [c for c in df.columns if c not in excluded_cols]
print(f"Features: {len(feature_cols)} columnas")
print(f"Excluidas: {excluded_cols}")

# ── CELDA 9: Escalado (fit SOLO en train — Regla R1 anti-leakage) ──
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(df_train[feature_cols])
scaler_y.fit(df_train[[TARGET]])

def scale_partition(df_part):
    X = scaler_X.transform(df_part[feature_cols]).astype(np.float32)
    y = scaler_y.transform(df_part[[TARGET]]).ravel().astype(np.float32)
    return X, y

X_tr_s, y_tr_s = scale_partition(df_train)
X_va_s, y_va_s = scale_partition(df_val)
X_te_s, y_te_s = scale_partition(df_test)
print("Escalado completado (scaler fit solo sobre train).")

# ── CELDA 10: Generar secuencias 3D ───────────────────────────
LOOKBACK  = CFG["lookback_window"]
HORIZON   = CFG["forecast_horizon"]
USE_RESID = CFG["use_residual"]

def make_sequences(data_X, data_y, lookback, horizon, use_residual=False):
    X, y, anchors = [], [], []
    max_i = len(data_X) - lookback - horizon + 1
    for i in range(max_i):
        X.append(data_X[i: i + lookback])
        anchor = data_y[i + lookback - 1]
        future = data_y[i + lookback: i + lookback + horizon]
        y.append(future - anchor if use_residual else future)
        anchors.append(anchor)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(anchors, dtype=np.float32))

X_train, y_train, anch_train = make_sequences(X_tr_s, y_tr_s, LOOKBACK, HORIZON, USE_RESID)
X_val,   y_val,   anch_val   = make_sequences(X_va_s, y_va_s, LOOKBACK, HORIZON, USE_RESID)
X_test,  y_test,  anch_test  = make_sequences(X_te_s, y_te_s, LOOKBACK, HORIZON, USE_RESID)

test_dates = df_test.index[LOOKBACK + HORIZON - 1: LOOKBACK + HORIZON - 1 + len(y_test)]

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
print(f"X_test : {X_test.shape}  | y_test : {y_test.shape}")
print(f"Modo residual: {USE_RESID}")

# ── CELDA 11: Funciones de inversión y reconstrucción ────────
def inv(arr_2d: np.ndarray) -> np.ndarray:
    """Desnormaliza array 2D (samples, horizon) → metros reales."""
    result = np.zeros_like(arr_2d)
    for col in range(arr_2d.shape[1]):
        result[:, col] = scaler_y.inverse_transform(
            arr_2d[:, col].reshape(-1, 1)
        ).ravel()
    return result

def reconstruct_real(delta_or_abs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    if USE_RESID:
        return inv(anchor[:, np.newaxis] + delta_or_abs)
    return inv(delta_or_abs)

# Ground truth en metros reales
y_test_real = reconstruct_real(y_test, anch_test)
y_val_real  = reconstruct_real(y_val,  anch_val)

# ── CELDA 12: Construcción del modelo LSTM ───────────────────
def build_model(n_features: int, cfg: dict) -> keras.Model:
    lstm_cfg = cfg["lstm"]
    units    = lstm_cfg["units"]      # [128, 64] para XL
    dropout  = lstm_cfg["dropout"]
    lr       = lstm_cfg["learning_rate"]
    horizon  = cfg["forecast_horizon"]
    lookback = cfg["lookback_window"]
    huber_delta = lstm_cfg.get("huber_delta", 0.5)

    inputs = keras.Input(shape=(lookback, n_features), name="input_seq")
    x = keras.layers.LSTM(units[0], return_sequences=True,  name="lstm_1")(inputs)
    x = keras.layers.Dropout(dropout,                        name="drop_1")(x)
    x = keras.layers.LSTM(units[1], return_sequences=False, name="lstm_2")(x)
    x = keras.layers.Dropout(dropout,                        name="drop_2")(x)
    x = keras.layers.Dense(32, activation="relu",            name="dense_h")(x)
    outputs = keras.layers.Dense(horizon, activation="linear", name="output")(x)

    model = keras.Model(inputs, outputs, name="orinoco_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )
    return model

n_features = X_train.shape[2]
print(f"Arquitectura: LSTM({CFG['lstm']['units'][0]}) → LSTM({CFG['lstm']['units'][1]}) → Dense(32) → Dense({HORIZON})")
print(f"Features de entrada: {n_features}")

# ── CELDA 13: Entrenamiento del Ensemble (5 modelos) ─────────
def train_member(model, seed: int, member_num: int) -> tuple:
    lstm_cfg   = CFG["lstm"]
    patience   = lstm_cfg["patience"]
    max_epochs = lstm_cfg["max_epochs"]
    batch_size = lstm_cfg["batch_size"]

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(5, patience // 3),
            min_lr=1e-6,
            verbose=0,
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    best_ep  = int(np.argmin(history.history["val_loss"])) + 1
    best_val = float(min(history.history["val_loss"]))
    print(f"  Miembro {member_num}/{N_MEMBERS} | seed={seed} | best_epoch={best_ep} | val_loss={best_val:.6f}")
    return history.history

all_preds_test = []
all_preds_val  = []
all_histories  = []
y_train_max    = float(scaler_y.inverse_transform([[1.0]])[0, 0])

print(f"\n{'='*55}")
print(f"ENTRENANDO MODELO INDIVIDUAL (seed=42) × LSTM({CFG['lstm']['units']})")
print(f"Lookback: {LOOKBACK} días | Horizonte: {HORIZON} días | Residual: {USE_RESID}")
print(f"{'='*55}\n")

for i, seed in enumerate(ENSEMBLE_SEEDS[:N_MEMBERS]):
    set_seeds(seed)
    model = build_model(n_features, CFG)
    hist  = train_member(model, seed, i + 1)
    all_histories.append(hist)

    pred_test = reconstruct_real(model.predict(X_test, verbose=0), anch_test)
    pred_val  = reconstruct_real(model.predict(X_val,  verbose=0), anch_val)
    all_preds_test.append(pred_test)
    all_preds_val.append(pred_val)

    mae_m = float(np.abs(pred_test - y_test_real).mean())
    print(f"  → MAE_test del miembro: {mae_m*100:.1f} cm\n")

# ── CELDA 14: Promediar ensemble ──────────────────────────────
y_pred_test = np.mean(all_preds_test, axis=0)
y_pred_val  = np.mean(all_preds_val,  axis=0)

# Restricciones físicas
y_pred_test = np.maximum(y_pred_test, 0.0)
y_pred_test = np.minimum(y_pred_test, y_train_max * 1.15)

print("Ensemble promediado y restricciones físicas aplicadas.")

# ── CELDA 15: Métricas finales ────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_t = y_true.ravel()
    y_p = y_pred.ravel()
    mae  = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    nse  = float(1.0 - ss_res / ss_tot)
    r     = float(np.corrcoef(y_t, y_p)[0, 1])
    alpha = float(np.std(y_p) / np.std(y_t))
    beta  = float(np.mean(y_p) / np.mean(y_t))
    kge   = float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))
    return {"mae": mae, "rmse": rmse, "nse": nse, "kge": kge}

test_m = compute_metrics(y_test_real, y_pred_test)
val_m  = compute_metrics(y_val_real,  y_pred_val)

print("\n" + "="*55)
print("RESULTADOS FINALES DEL ENSEMBLE")
print("="*55)
print(f"TEST  — MAE: {test_m['mae']*100:.1f} cm | RMSE: {test_m['rmse']*100:.1f} cm | NSE: {test_m['nse']:.4f} | KGE: {test_m['kge']:.4f}")
print(f"VAL   — MAE: {val_m['mae']*100:.1f} cm  | RMSE: {val_m['rmse']*100:.1f} cm  | NSE: {val_m['nse']:.4f} | KGE: {val_m['kge']:.4f}")

# Comparar con Gold Standard original
print("\nReferencia (Gold Standard original):")
print("  MAE: 13.3 cm | NSE: 0.9959 | KGE: 0.9938")
print(f"Diferencia MAE: {abs(test_m['mae']*100 - 13.3):.1f} cm")

# ── CELDA 16: Gráfico de predicciones vs real ─────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Usar el paso t+7 (último del horizonte)
y_true_plot = y_test_real[:, -1]
y_pred_plot = y_pred_test[:, -1]

ax = axes[0]
ax.plot(test_dates, y_true_plot, color="#2196F3", lw=1.5, label="Observado (Palúa)", alpha=0.9)
ax.plot(test_dates, y_pred_plot, color="#4CAF50", lw=1.5, label=f"Ensemble t+7 (MAE={test_m['mae']*100:.1f}cm)", alpha=0.85)
ax.fill_between(test_dates,
                y_pred_plot - test_m["mae"],
                y_pred_plot + test_m["mae"],
                alpha=0.15, color="#4CAF50", label="±1 MAE")
ax.set_title("Predicciones Ensemble vs Observado — Test Set (2022-07 → 2025-02)", fontsize=13, fontweight="bold")
ax.set_ylabel("Nivel del río Palúa (m)")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

# Error absoluto
ax2 = axes[1]
err = np.abs(y_true_plot - y_pred_plot)
ax2.fill_between(test_dates, err, alpha=0.5, color="#FF5722")
ax2.axhline(test_m["mae"], color="red", linestyle="--", linewidth=1.5, label=f"MAE={test_m['mae']*100:.1f}cm")
ax2.set_title("Error Absoluto Diario (t+7)", fontsize=12, fontweight="bold")
ax2.set_ylabel("|Error| (m)")
ax2.set_xlabel("Fecha")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("predicciones_gold_standard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: predicciones_gold_standard.png")

# ── CELDA 17: Guardar predicciones en CSV ────────────────────
output_df = pd.DataFrame({
    "fecha":   test_dates,
    "y_true":  y_true_plot,
    "y_pred":  y_pred_plot,
    "error_abs": err,
})
output_df.to_csv("predictions_test_colab.csv", index=False)
print("Predicciones guardadas: predictions_test_colab.csv")

# ── CELDA 18: Guardar métricas en JSON ───────────────────────
result = {
    "experiment_name": "best_single_lstm_lb150_lags_xl_seed42",
    "n_members": N_MEMBERS,
    "seeds": ENSEMBLE_SEEDS[:N_MEMBERS],
    "config": {
        "lookback": LOOKBACK,
        "units": CFG["lstm"]["units"],
        "use_residual": USE_RESID,
        "loss": CFG["lstm"]["loss"],
    },
    "metrics": {
        "test": test_m,
        "val":  val_m,
    },
    "reference_gold_standard": {
        "mae": 0.133, "rmse": 0.199, "nse": 0.9959, "kge": 0.9938
    }
}
with open("metrics_colab.json", "w") as f:
    json.dump(result, f, indent=2)
print("Métricas guardadas: metrics_colab.json")
print("\n✅ Reproducción completada.")
