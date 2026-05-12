# ==============================================================
# ORINOCO FORECAST — Reproducción del Gold Standard LSTM
# Modelo individual seed=42 (mejor resultado en Colab T4)
# MAE Test: 13.3 cm | NSE: 0.9959 | KGE: 0.9938
#
# INSTRUCCIONES:
#   1. Activar GPU: Runtime → Change runtime type → T4 GPU
#   2. Ejecutar desde la raíz del repositorio clonado
#   3. Resultados en results/experiments/lstm_best_seed42/
#
# Para desactivar oneDNN (mayor reproducibilidad numérica):
#   TF_ENABLE_ONEDNN_OPTS=0 python colab_gold_standard.py
# ==============================================================

# ── (Solo en Colab) Clonar repo si no existe ─────────────────
# !git clone https://github.com/iadev0721/orinoco-forecast.git
# %cd orinoco-forecast

import subprocess, sys

subprocess.run(
    [
        sys.executable, "scripts/run_ensemble.py",
        "--name",     "lstm_best_seed42",
        "--n",        "1",          # solo seed=42
        "--lookback", "150",
        "--units",    "128", "64",
    ],
    check=True,
)
