# ==============================================================
# ORINOCO FORECAST — Transformer Ensemble Gold Standard
# MAE ref: 13.4 cm | RMSE: 19.8 cm | NSE: 0.9959 | KGE: 0.9919
# ==============================================================
import json, math, random, shutil, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 0. Baseline gate (Regla R3) ──────────────────────────────
baseline_path = Path("results/metrics/baseline_metrics.json")
if not baseline_path.exists():
    print("=" * 60)
    print("Ejecutando baseline (Requisito R3)...")
    print("=" * 60)
    subprocess.run(
        [sys.executable, "scripts/run_experiment.py",
         "--name", "baseline_naive", "--model", "naive"],
        check=True
    )

# ── 1. Config ─────────────────────────────────────────────────
EXPERIMENT_NAME = "transformer"
CFG = {
    "target_station":   "palua",
    "forecast_horizon": 7,
    "lookback_window":  150,
    "train_end":        "2018-12-31",
    "val_end":          "2022-06-30",
    "use_residual":     True,
    "seed":             42,
    "transformer": {
        "d_model": 64, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 128, "dropout": 0.1,
        "batch_size": 32, "max_epochs": 100,
        "patience": 10, "learning_rate": 0.0001,
        "loss": "mse", "grad_clip": 1.0,
    },
    "physical": {"min_level_m": 0.0, "max_level_multiplier": 1.15},
}
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1011]
N_MEMBERS      = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. Reproducibilidad ───────────────────────────────────────
def set_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seeds(CFG["seed"])
print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {vram:.1f} GB")

# ── 3. Cargar y preparar datos ────────────────────────────────
FEATURES_PATH  = "data/processed/dataset_orinoco_features.csv"
STATION_ORDER  = ["ayacucho", "caicara", "ciudad_bolivar", "palua"]
TARGET         = CFG["target_station"]
LOOKBACK       = CFG["lookback_window"]
HORIZON        = CFG["forecast_horizon"]
USE_RESID      = CFG["use_residual"]

df = pd.read_csv(FEATURES_PATH, parse_dates=["fecha"]).set_index("fecha").sort_index()
df_train = df.loc[:CFG["train_end"]]
df_val   = df.loc[pd.Timestamp(CFG["train_end"]) + pd.Timedelta(days=1): CFG["val_end"]]
df_test  = df.loc[pd.Timestamp(CFG["val_end"])   + pd.Timedelta(days=1):]

target_idx    = STATION_ORDER.index(TARGET)
excluded_cols = [c for c in df.columns
                 if any(c == s or c.startswith(s+"_") for s in STATION_ORDER[target_idx:])]
feature_cols  = [c for c in df.columns if c not in excluded_cols]

scaler_X = MinMaxScaler().fit(df_train[feature_cols])
scaler_y = MinMaxScaler().fit(df_train[[TARGET]])

def scale(df_part):
    X = scaler_X.transform(df_part[feature_cols]).astype(np.float32)
    y = scaler_y.transform(df_part[[TARGET]]).ravel().astype(np.float32)
    return X, y

X_tr_s, y_tr_s = scale(df_train)
X_va_s, y_va_s = scale(df_val)
X_te_s, y_te_s = scale(df_test)

def make_sequences(data_X, data_y, lookback, horizon, use_residual=False):
    X, y, anchors = [], [], []
    for i in range(len(data_X) - lookback - horizon + 1):
        anchor = data_y[i + lookback - 1]
        future = data_y[i + lookback: i + lookback + horizon]
        X.append(data_X[i: i + lookback])
        y.append(future - anchor if use_residual else future)
        anchors.append(anchor)
    return (np.array(X, np.float32), np.array(y, np.float32),
            np.array(anchors, np.float32))

X_train, y_train, anch_train = make_sequences(X_tr_s, y_tr_s, LOOKBACK, HORIZON, USE_RESID)
X_val,   y_val,   anch_val   = make_sequences(X_va_s, y_va_s, LOOKBACK, HORIZON, USE_RESID)
X_test,  y_test,  anch_test  = make_sequences(X_te_s, y_te_s, LOOKBACK, HORIZON, USE_RESID)
test_dates = df_test.index[LOOKBACK+HORIZON-1: LOOKBACK+HORIZON-1+len(y_test)]
y_train_max = float(scaler_y.inverse_transform([[1.0]])[0, 0])

def inv(arr_2d):
    r = np.zeros_like(arr_2d)
    for col in range(arr_2d.shape[1]):
        r[:, col] = scaler_y.inverse_transform(arr_2d[:, col].reshape(-1,1)).ravel()
    return r

def reconstruct(raw, anchor):
    return inv(anchor[:, np.newaxis] + raw) if USE_RESID else inv(raw)

y_test_real = reconstruct(y_test, anch_test)
y_val_real  = reconstruct(y_val,  anch_val)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

# ── 4. Arquitectura ───────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class OrincoTransformer(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        tr   = cfg["transformer"]
        d    = tr["d_model"]; hz = cfg["forecast_horizon"]
        self.input_proj = nn.Linear(n_features, d)
        self.pos_enc    = PositionalEncoding(d, tr["dropout"])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=tr["nhead"],
            dim_feedforward=tr["dim_feedforward"],
            dropout=tr["dropout"], batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=tr["num_layers"])
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.Dropout(tr["dropout"]), nn.Linear(d, hz),
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])

n_features = X_train.shape[2]

# ── 5. Entrenamiento de un miembro ────────────────────────────
def to_t(arr): return torch.from_numpy(arr).to(DEVICE)

def train_member(seed, member_num):
    tr_cfg     = CFG["transformer"]
    lr         = tr_cfg["learning_rate"]
    max_epochs = tr_cfg["max_epochs"]
    patience   = tr_cfg["patience"]
    batch_size = tr_cfg["batch_size"]
    grad_clip  = tr_cfg["grad_clip"]

    model     = OrincoTransformer(n_features, CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(3, patience // 3), min_lr=1e-6)
    criterion = nn.MSELoss()
    loader    = DataLoader(TensorDataset(to_t(X_train), to_t(y_train)),
                           batch_size=batch_size, shuffle=True)
    X_val_t   = to_t(X_val); y_val_t = to_t(y_val)

    best_val, no_imp, best_epoch, best_state = float("inf"), 0, 1, None
    history = {"loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            run_loss += loss.item() * xb.size(0); seen += xb.size(0)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        train_loss = run_loss / max(seen, 1)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch:>3}/{max_epochs} "
              f"-- loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss; best_epoch = epoch; no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stopping en epoca {epoch}.")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    print(f"  Miembro {member_num}/{N_MEMBERS} | seed={seed} "
          f"| best_epoch={best_epoch} | val_loss={best_val:.6f}")

    with torch.no_grad():
        pred_test_raw = model(to_t(X_test)).cpu().numpy()
        pred_val_raw  = model(to_t(X_val)).cpu().numpy()

    pred_test = reconstruct(pred_test_raw, anch_test)
    pred_val  = reconstruct(pred_val_raw,  anch_val)
    mae = float(np.abs(pred_test - y_test_real).mean())
    print(f"  -> MAE_test del miembro: {mae*100:.1f} cm\n")
    return pred_test, pred_val, history

# ── 6. Ensemble ───────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"ENTRENANDO ENSEMBLE TRANSFORMER: {N_MEMBERS} miembros")
print(f"d_model=64 | nhead=4 | layers=2 | lookback=150 | residual=True")
print(f"{'='*58}\n")

all_preds_test, all_preds_val, all_histories = [], [], []
for i, seed in enumerate(ENSEMBLE_SEEDS[:N_MEMBERS]):
    set_seeds(seed)
    pt, pv, hist = train_member(seed, i + 1)
    all_preds_test.append(pt)
    all_preds_val.append(pv)
    all_histories.append(hist)

y_pred_test = np.mean(all_preds_test, axis=0)
y_pred_val  = np.mean(all_preds_val,  axis=0)
y_pred_test = np.clip(y_pred_test,
                      CFG["physical"]["min_level_m"],
                      y_train_max * CFG["physical"]["max_level_multiplier"])

bias_val       = float(np.mean(y_pred_val - y_val_real))
y_pred_test_bc = y_pred_test - bias_val
print(f"Correccion de sesgo (val): {bias_val*100:.2f} cm")

# ── 7. Métricas ───────────────────────────────────────────────
def metrics(y_true, y_pred):
    yt, yp = y_true.ravel(), y_pred.ravel()
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp)**2)))
    nse  = float(1 - np.sum((yt-yp)**2) / np.sum((yt-np.mean(yt))**2))
    r    = float(np.corrcoef(yt, yp)[0, 1])
    a    = float(np.std(yp) / np.std(yt))
    b    = float(np.mean(yp) / np.mean(yt))
    kge  = float(1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2))
    return {"mae": mae, "rmse": rmse, "nse": nse, "kge": kge}

m_test    = metrics(y_test_real, y_pred_test)
m_test_bc = metrics(y_test_real, y_pred_test_bc)
m_val     = metrics(y_val_real,  y_pred_val)

print(f"\n{'='*58}")
print("RESULTADOS FINALES")
print(f"TEST (sin BC) MAE:{m_test['mae']*100:.1f}cm NSE:{m_test['nse']:.4f} KGE:{m_test['kge']:.4f}")
print(f"TEST (con BC) MAE:{m_test_bc['mae']*100:.1f}cm NSE:{m_test_bc['nse']:.4f} KGE:{m_test_bc['kge']:.4f} <- OFICIAL")
print(f"VAL           MAE:{m_val['mae']*100:.1f}cm NSE:{m_val['nse']:.4f} KGE:{m_val['kge']:.4f}")
print(f"Referencia Gold Standard: MAE=13.4cm NSE=0.9959 KGE=0.9919")
print(f"{'='*58}")

# ── 8. Guardar resultados ─────────────────────────────────────
exp_dir = Path(f"results/experiments/{EXPERIMENT_NAME}")
exp_dir.mkdir(parents=True, exist_ok=True)

# predicciones.csv
y_true_plot = y_test_real[:, -1]
y_pred_plot = y_pred_test_bc[:, -1]
err         = np.abs(y_true_plot - y_pred_plot)

pd.DataFrame({
    "fecha":     test_dates,
    "y_true":    y_true_plot,
    "y_pred_bc": y_pred_plot,
    "error_abs": err,
}).to_csv(exp_dir / "predictions_test.csv", index=False)

# metrics.json
with open(exp_dir / "metrics.json", "w") as f:
    json.dump({"test_bc": m_test_bc, "test_no_bc": m_test, "val": m_val,
               "bias_correction_cm": round(bias_val*100, 4)}, f, indent=2)

# config_used.yaml
import yaml
with open(exp_dir / "config_used.yaml", "w") as f:
    yaml.dump(CFG, f, allow_unicode=True)

# training_history.json
with open(exp_dir / "training_history.json", "w") as f:
    json.dump(all_histories, f)

# ── 9. Gráficos ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
ax = axes[0]
ax.plot(test_dates, y_true_plot, color="#2196F3", lw=1.5, label="Observado (Palua)", alpha=0.9)
ax.plot(test_dates, y_pred_plot, color="#4CAF50", lw=1.5,
        label=f"Transformer Ensemble t+7 (MAE={m_test_bc['mae']*100:.1f}cm)", alpha=0.85)
ax.fill_between(test_dates, y_pred_plot-m_test_bc["mae"],
                y_pred_plot+m_test_bc["mae"], alpha=0.15, color="#4CAF50")
ax.set_title("Predicciones Transformer Ensemble vs Observado -- Test Set",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Nivel del rio Palua (m)"); ax.legend(fontsize=10)
ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

ax2 = axes[1]
ax2.fill_between(test_dates, err, alpha=0.5, color="#FF5722")
ax2.axhline(m_test_bc["mae"], color="red", linestyle="--", lw=1.5,
            label=f"MAE={m_test_bc['mae']*100:.1f}cm")
ax2.set_title("Error Absoluto Diario (t+7)", fontsize=12, fontweight="bold")
ax2.set_ylabel("|Error| (m)"); ax2.set_xlabel("Fecha")
ax2.legend(fontsize=10); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(exp_dir / "predicciones.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Learning curve
fig2, ax3 = plt.subplots(figsize=(12, 5))
min_len   = min(len(h["val_loss"]) for h in all_histories)
avg_val   = np.mean([h["val_loss"][:min_len] for h in all_histories], axis=0)
avg_train = np.mean([h["loss"][:min_len]     for h in all_histories], axis=0)
for h, seed in zip(all_histories, ENSEMBLE_SEEDS[:N_MEMBERS]):
    ax3.plot(range(1, len(h["val_loss"])+1), h["val_loss"], lw=1, alpha=0.4, label=f"seed={seed}")
ax3.plot(range(1, min_len+1), avg_val,   color="black", lw=2.5, linestyle="--", label="Promedio val_loss")
ax3.plot(range(1, min_len+1), avg_train, color="gray",  lw=1.5, linestyle="-.", label="Promedio train_loss")
ax3.set_title("Curva de Aprendizaje -- Transformer Ensemble", fontsize=13, fontweight="bold")
ax3.set_xlabel("Epoca"); ax3.set_ylabel("Loss"); ax3.legend(fontsize=9)
ax3.grid(alpha=0.3); ax3.set_yscale("log")
plt.tight_layout()
plt.savefig(exp_dir / "learning_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── 10. Copiar a raíz ─────────────────────────────────────────
dest_dir = Path(EXPERIMENT_NAME)
if dest_dir.exists():
    shutil.rmtree(dest_dir)
shutil.copytree(exp_dir, dest_dir)

print(f"\nResultados guardados en: ./{EXPERIMENT_NAME}/")
print("  predicciones.png, learning_curve.png, predictions_test.csv,")
print("  metrics.json, config_used.yaml, training_history.json")
print("\nProceso completado exitosamente.")
