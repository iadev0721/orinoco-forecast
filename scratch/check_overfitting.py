import json
import matplotlib.pyplot as plt
import os
import pandas as pd

def check_overfitting(exp_name):
    path = f"results/experiments/{exp_name}"
    history_file = f"{path}/training_history.json"
    metrics_file = f"{path}/metrics.json"

    if os.path.exists(history_file):
        # 1. Analizar Historia de Entrenamiento
        with open(history_file, "r") as f:
            history = json.load(f)
        
        epochs = range(1, len(history["loss"]) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.title(f"Curvas de Loss - {exp_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: MAE
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["mae"], label="Train MAE")
        plt.plot(epochs, history["val_mae"], label="Val MAE")
        plt.title(f"Curvas de MAE - {exp_name}")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{path}/overfitting_check.png"
        plt.savefig(plot_path)
        print(f"Gráfico guardado en: {plot_path}")
    else:
        print("Aviso: No se encontró history file (común en ensembles). Saltando gráficos.")

    # 2. Comparar Métricas Finales
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    val_mae = metrics["metrics"]["val"]["mae"] * 100
    test_mae = metrics["metrics"]["test"]["mae"] * 100
    
    print("\n=== DIAGNÓSTICO DE OVERFITTING ===")
    print(f"Modelo: {exp_name}")
    
    if os.path.exists(history_file):
        print(f"Final Train Loss: {history['loss'][-1]:.6f}")
        print(f"Final Val Loss:   {history['val_loss'][-1]:.6f}")
        print(f"Gap Loss:         {abs(history['loss'][-1] - history['val_loss'][-1]):.6f}")
    
    print("-" * 35)
    print(f"MAE Validación:   {val_mae:.2f} cm")
    print(f"MAE Test:         {test_mae:.2f} cm")
    print(f"Gap MAE:          {abs(val_mae - test_mae):.2f} cm")
    
    if abs(val_mae - test_mae) < 2.0:
        print("\nResultado: SALUDABLE")
        print("El modelo generaliza bien. El error en test es similar al de validación.")
    elif test_mae > val_mae + 5.0:
        print("\nResultado: OVERFITTING DETECTADO")
        print("El modelo rinde mucho mejor en validación que en datos nuevos (test).")
    else:
        print("\nResultado: LIGERO SESGO")
        print("Hay una diferencia pequeña, común en series temporales.")

if __name__ == "__main__":
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else "lstm_residual_lb90"
    check_overfitting(exp)
