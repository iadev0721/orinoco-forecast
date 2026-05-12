# ==============================================================
# ORINOCO FORECAST — Entrenamiento LSTM Individual en Colab
# Modelo individual (seed=42 por defecto de config.yaml)
#
# INSTRUCCIONES:
#   1. Activar GPU: Runtime → Change runtime type → T4 GPU
#   2. Clonar el repo y ejecutar desde su raíz
#   3. Resultados en results/experiments/lstm_individual/
#
# Para desactivar oneDNN (mayor reproducibilidad numérica):
#   TF_ENABLE_ONEDNN_OPTS=0 python colab_lstm_individual.py
# ==============================================================

# ── (Solo en Colab) Clonar repo si no existe ─────────────────
# !git clone https://github.com/iadev0721/orinoco-forecast.git
# %cd orinoco-forecast

import subprocess, sys

subprocess.run(
    [
        sys.executable, "scripts/run_experiment.py",
        "--name",     "lstm_individual",
        "--model",    "lstm",
        "--lookback", "150",
        "--units",    "128", "64",
    ],
    check=True,
)
