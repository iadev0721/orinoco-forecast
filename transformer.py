# ==============================================================
# ORINOCO FORECAST — Transformer Model (Ensemble 5 miembros)
# ==============================================================

import subprocess
import sys
import shutil
from pathlib import Path

def main():
    experiment_name = "transformer"
    
    # 1. Ejecutar el pipeline de entrenamiento
    print("=" * 60)
    print(f"Iniciando entrenamiento del modelo Transformer...")
    print("=" * 60)
    
    subprocess.run(
        [
            sys.executable, "scripts/run_transformer_ensemble.py",
            "--name", experiment_name,
            "--n",    "5",
        ],
        check=True,
    )
    
    # 2. Mover la carpeta de resultados a la raíz para fácil acceso
    src_dir = Path(f"results/experiments/{experiment_name}")
    dest_dir = Path(experiment_name)
    
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
        
    shutil.copytree(src_dir, dest_dir)
    
    print("\n" + "=" * 60)
    print(f"Proceso completado exitosamente.")
    print(f"Los resultados y gráficos se encuentran en la carpeta raíz: ./{experiment_name}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
