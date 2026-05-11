# ==============================================================
# ORINOCO FORECAST — Reproducción del Gold Standard Transformer
# Modelo: ensemble_transformer_v1
# Config: d_model=64, nhead=4, 2 capas, lookback=150, residual=True
# Referencia: MAE=13.4 cm | RMSE=19.8 cm | NSE=0.9959 | KGE=0.9919
#
# INSTRUCCIONES:
#   1. Activar GPU en Colab: Runtime → Change runtime type → T4 GPU
#   2. Ejecutar desde la raíz del repositorio clonado
#   3. Asegurarse de que data/processed/dataset_orinoco_features.csv existe
# ==============================================================

# ── CELDA 1: Instalación (solo si falta torch) ───────────────
# torch ya viene en Colab — solo verificar:
# !pip show torch torchvision | grep Version

# ── CELDA 2: Imports ──────────────────────────────────────────
import json
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── CELDA 3: Configuración GPU ────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── CELDA 4: Hiperparámetros del Gold Standard Transformer ────
CFG = {
    "target_station":   "palua",
    "forecast_horizon": 7,
    "lookback_window":  150,
    "train_end":        "2018-12-31",
    "val_end":          "2022-06-30",
    "use_residual":     True,
    "seed":             42,
    "transformer": {
        "d_model":          64,
        "nhead":            4,
        "num_layers":       2,
        "dim_feedforward":  128,
        "dropout":          0.1,
        "batch_size":       32,
        "max_epochs":       100,
        "patience":         10,
        "learning_rate":    0.0001,
        "loss":             "mse",       # mse es el gold standard del transformer
        "grad_clip":        1.0,
    },
    "physical": {
        "min_level_m":          0.0,
        "max_level_multiplier": 1.15,
    },
}

ENSEMBLE_SEEDS = [42, 123, 456, 789, 1011]   # gold standard transformer: 5 miembros
N_MEMBERS      = 5

# ── CELDA 5: Reproducibilidad global ──────────────────────────
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seeds(CFG["seed"])
print("Seeds fijadas.")

# ── CELDA 6: Cargar dataset ────────────────────────────────────
FEATURES_PATH = "data/processed/dataset_orinoco_features.csv"

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

# ── CELDA 8: Feature selection ────────────────────────────────
target_idx        = STATION_ORDER.index(TARGET)
excluded_stations = STATION_ORDER[target_idx:]
excluded_cols     = [
    col for col in df.columns
    if any(col == s or col.startswith(s + "_") for s in excluded_stations)
]
feature_cols = [c for c in df.columns if c not in excluded_cols]
print(f"Features: {len(feature_cols)} columnas | Excluidas: {excluded_cols}")

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
print("Escalado completado.")

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

# ── CELDA 11: Inversión y reconstrucción ──────────────────────
def inv(arr_2d: np.ndarray) -> np.ndarray:
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

y_test_real = reconstruct_real(y_test, anch_test)
y_val_real  = reconstruct_real(y_val,  anch_val)
y_train_max = float(scaler_y.inverse_transform([[1.0]])[0, 0])

# ── CELDA 12: Arquitectura del Transformer ───────────────────
class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal (Pre-LN, hereda de nn.Module
    para que Dropout se desactive correctamente en eval())."""
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class OrincoTransformer(nn.Module):
    """
    Transformer encoder-only con:
    - Proyección lineal de entrada (n_features → d_model)
    - Positional Encoding sinusoidal
    - N capas de TransformerEncoderLayer (Pre-LN, batch_first=True)
    - Agregación CLS-like: último token del encoder
    - MLP head → horizon
    """
    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        tr = cfg["transformer"]
        d      = tr["d_model"]
        nhead  = tr["nhead"]
        nlays  = tr["num_layers"]
        ff     = tr["dim_feedforward"]
        drop   = tr["dropout"]
        hz     = cfg["forecast_horizon"]

        self.input_proj = nn.Linear(n_features, d)

        self.pos_enc = PositionalEncoding(d, drop)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead,
            dim_feedforward=ff,
            dropout=drop,
            batch_first=True,
            norm_first=True,        # Pre-LN — más estable
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlays)

        # MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d, hz),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        x = self.input_proj(x)          # → (batch, lookback, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)             # → (batch, lookback, d_model)
        x = x[:, -1, :]                 # último token (CLS-like aggregation)
        return self.head(x)             # → (batch, horizon)


n_features = X_train.shape[2]
print(f"Arquitectura: d_model={CFG['transformer']['d_model']}, "
      f"nhead={CFG['transformer']['nhead']}, "
      f"num_layers={CFG['transformer']['num_layers']}, "
      f"dim_feedforward={CFG['transformer']['dim_feedforward']}")
print(f"Features de entrada: {n_features}")
_demo = OrincoTransformer(n_features, CFG).to(DEVICE)
n_params = sum(p.numel() for p in _demo.parameters() if p.requires_grad)
print(f"Parámetros entrenables: {n_params:,}")
del _demo

# ── CELDA 13: Entrenamiento de un miembro ────────────────────
def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).to(DEVICE)

def train_member(seed: int, member_num: int) -> tuple[np.ndarray, np.ndarray]:
    tr_cfg     = CFG["transformer"]
    lr         = tr_cfg["learning_rate"]
    max_epochs = tr_cfg["max_epochs"]
    patience   = tr_cfg["patience"]
    batch_size = tr_cfg["batch_size"]
    grad_clip  = tr_cfg["grad_clip"]

    model = OrincoTransformer(n_features, CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(3, patience // 3), min_lr=1e-6
    )
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(to_tensor(X_train), to_tensor(y_train)),
        batch_size=batch_size, shuffle=True,
    )

    X_val_t = to_tensor(X_val)
    y_val_t = to_tensor(y_val)

    best_val  = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restaurar mejor checkpoint
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    print(f"  Miembro {member_num}/{N_MEMBERS} | seed={seed} | best_epoch={best_epoch} | val_loss={best_val:.6f}")

    with torch.no_grad():
        pred_test_raw = model(to_tensor(X_test)).cpu().numpy()
        pred_val_raw  = model(to_tensor(X_val)).cpu().numpy()

    pred_test = reconstruct_real(pred_test_raw, anch_test)
    pred_val  = reconstruct_real(pred_val_raw,  anch_val)

    mae_m = float(np.abs(pred_test - y_test_real).mean())
    print(f"  → MAE_test del miembro: {mae_m*100:.1f} cm\n")
    return pred_test, pred_val


# ── CELDA 14: Entrenamiento del Ensemble ─────────────────────
print(f"\n{'='*55}")
print(f"ENTRENANDO ENSEMBLE TRANSFORMER: {N_MEMBERS} miembros")
print(f"d_model={CFG['transformer']['d_model']} | nhead={CFG['transformer']['nhead']} | "
      f"layers={CFG['transformer']['num_layers']}")
print(f"Lookback: {LOOKBACK} días | Horizonte: {HORIZON} días | Residual: {USE_RESID}")
print(f"{'='*55}\n")

all_preds_test = []
all_preds_val  = []

for i, seed in enumerate(ENSEMBLE_SEEDS[:N_MEMBERS]):
    set_seeds(seed)
    pred_test, pred_val = train_member(seed, i + 1)
    all_preds_test.append(pred_test)
    all_preds_val.append(pred_val)

# ── CELDA 15: Promediar ensemble y restricciones físicas ──────
y_pred_test = np.mean(all_preds_test, axis=0)
y_pred_val  = np.mean(all_preds_val,  axis=0)

# Restricciones físicas
y_pred_test = np.maximum(y_pred_test, CFG["physical"]["min_level_m"])
y_pred_test = np.minimum(y_pred_test, y_train_max * CFG["physical"]["max_level_multiplier"])

# Corrección de sesgo (bias correction) usando validación
# Justificación: el Transformer con MSE loss sobreestima ligeramente.
# Se calcula el sesgo sistemático en validación y se sustrae del test.
bias_val = float(np.mean(y_pred_val - y_val_real))
y_pred_test_bc = y_pred_test - bias_val
print(f"Corrección de sesgo (val): {bias_val*100:.2f} cm → aplicada al test")
print("Ensemble promediado y restricciones físicas aplicadas.")

# ── CELDA 16: Métricas finales ────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_t  = y_true.ravel()
    y_p  = y_pred.ravel()
    mae  = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    nse  = float(1.0 - ss_res / ss_tot)
    r     = float(np.corrcoef(y_t, y_p)[0, 1])
    alpha = float(np.std(y_p) / np.std(y_t))
    beta  = float(np.mean(y_p) / np.mean(y_t))
    kge   = float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))
    return {"mae": mae, "rmse": rmse, "nse": nse, "kge": kge,
            "kge_r": r, "kge_alpha": alpha, "kge_beta": beta}

test_m    = compute_metrics(y_test_real, y_pred_test)
test_m_bc = compute_metrics(y_test_real, y_pred_test_bc)
val_m     = compute_metrics(y_val_real,  y_pred_val)

print("\n" + "="*55)
print("RESULTADOS FINALES DEL ENSEMBLE TRANSFORMER")
print("="*55)
print(f"TEST (sin BC) — MAE: {test_m['mae']*100:.1f} cm | RMSE: {test_m['rmse']*100:.1f} cm | NSE: {test_m['nse']:.4f} | KGE: {test_m['kge']:.4f}")
print(f"TEST (con BC) — MAE: {test_m_bc['mae']*100:.1f} cm | RMSE: {test_m_bc['rmse']*100:.1f} cm | NSE: {test_m_bc['nse']:.4f} | KGE: {test_m_bc['kge']:.4f}  ← OFICIAL")
print(f"VAL           — MAE: {val_m['mae']*100:.1f} cm  | RMSE: {val_m['rmse']*100:.1f} cm  | NSE: {val_m['nse']:.4f} | KGE: {val_m['kge']:.4f}")

print("\nReferencia (Gold Standard Transformer original):")
print("  MAE: 13.4 cm | RMSE: 19.8 cm | NSE: 0.9959 | KGE: 0.9919")
print(f"Diferencia MAE: {abs(test_m_bc['mae']*100 - 13.4):.1f} cm")

print("\nReferencia (LSTM Gold Standard):")
print("  MAE: 13.3 cm | RMSE: 19.9 cm | NSE: 0.9959 | KGE: 0.9938")

# ── CELDA 17: Gráfico predicciones vs real ────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

y_true_plot = y_test_real[:, -1]
y_pred_plot = y_pred_test_bc[:, -1]
err = np.abs(y_true_plot - y_pred_plot)

ax = axes[0]
ax.plot(test_dates, y_true_plot, color="#2196F3", lw=1.5, label="Observado (Palúa)", alpha=0.9)
ax.plot(test_dates, y_pred_plot, color="#FF9800", lw=1.5,
        label=f"Transformer ensemble t+7 (MAE={test_m_bc['mae']*100:.1f}cm)", alpha=0.85)
ax.fill_between(test_dates,
                y_pred_plot - test_m_bc["mae"],
                y_pred_plot + test_m_bc["mae"],
                alpha=0.15, color="#FF9800", label="±1 MAE")
ax.set_title("Predicciones Transformer Ensemble vs Observado — Test Set (2022-07 → 2025-02)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Nivel del río Palúa (m)")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

ax2 = axes[1]
ax2.fill_between(test_dates, err, alpha=0.5, color="#9C27B0")
ax2.axhline(test_m_bc["mae"], color="purple", linestyle="--", linewidth=1.5,
            label=f"MAE={test_m_bc['mae']*100:.1f}cm")
ax2.set_title("Error Absoluto Diario (t+7) — Transformer Ensemble", fontsize=12, fontweight="bold")
ax2.set_ylabel("|Error| (m)")
ax2.set_xlabel("Fecha")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("predicciones_transformer.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: predicciones_transformer.png")

# ── CELDA 18: Guardar CSV de predicciones ────────────────────
output_df = pd.DataFrame({
    "fecha":     test_dates,
    "y_true":    y_true_plot,
    "y_pred_bc": y_pred_plot,
    "error_abs": err,
})
output_df.to_csv("predictions_transformer_colab.csv", index=False)
print("Predicciones guardadas: predictions_transformer_colab.csv")

# ── CELDA 19: Guardar métricas en JSON ───────────────────────
result = {
    "experiment_name": "ensemble_transformer_v1_colab_repro",
    "n_members": N_MEMBERS,
    "seeds": ENSEMBLE_SEEDS[:N_MEMBERS],
    "config": {
        "lookback":        LOOKBACK,
        "d_model":         CFG["transformer"]["d_model"],
        "nhead":           CFG["transformer"]["nhead"],
        "num_layers":      CFG["transformer"]["num_layers"],
        "dim_feedforward": CFG["transformer"]["dim_feedforward"],
        "use_residual":    USE_RESID,
        "loss":            CFG["transformer"]["loss"],
        "bias_correction_cm": round(bias_val * 100, 4),
    },
    "metrics": {
        "test_no_bc": test_m,
        "test_bc":    test_m_bc,
        "val":        val_m,
    },
    "reference_gold_standard_transformer": {
        "mae": 0.134, "rmse": 0.198, "nse": 0.9959, "kge": 0.9919
    },
    "reference_gold_standard_lstm": {
        "mae": 0.133, "rmse": 0.199, "nse": 0.9959, "kge": 0.9938
    },
}
with open("metrics_transformer_colab.json", "w") as f:
    json.dump(result, f, indent=2)
print("Métricas guardadas: metrics_transformer_colab.json")
print("\n✅ Reproducción del Transformer completada.")
