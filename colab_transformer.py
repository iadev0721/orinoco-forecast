# ==============================================================
# ORINOCO FORECAST — Reproducción del Gold Standard Transformer
# Modelo: ensemble_transformer_v1 (5 × d_model=64)
# MAE Test: 13.4 cm | RMSE: 19.8 cm | NSE: 0.9959 | KGE: 0.9919
#
# INSTRUCCIONES:
#   1. Activar GPU: Runtime → Change runtime type → T4 GPU
#   2. Ejecutar desde la raíz del repositorio clonado
#   3. Resultados en results/experiments/ensemble_transformer_v1/
# ==============================================================

# ── (Solo en Colab) Clonar repo si no existe ─────────────────
# !git clone https://github.com/iadev0721/orinoco-forecast.git
# %cd orinoco-forecast

import subprocess, sys

subprocess.run(
    [
        sys.executable, "scripts/run_transformer_ensemble.py",
        "--name", "ensemble_transformer_v1",
        "--n",    "5",
    ],
    check=True,
)
