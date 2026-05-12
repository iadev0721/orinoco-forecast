import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def get_regime(month):
    if 1 <= month <= 4:
        return "Aguas Bajas (Ene-Abr)"
    elif 5 <= month <= 7:
        return "Ascenso (May-Jul)"
    elif 8 <= month <= 9:
        return "Aguas Altas (Ago-Sep)"
    else:
        return "Descenso (Oct-Dic)"

REGIME_COLORS = {
    "Aguas Bajas (Ene-Abr)": "#2196F3", # Azul
    "Ascenso (May-Jul)": "#FF9800",     # Naranja
    "Aguas Altas (Ago-Sep)": "#F44336", # Rojo
    "Descenso (Oct-Dic)": "#4CAF50"     # Verde
}

REGIME_ORDER = [
    "Aguas Bajas (Ene-Abr)",
    "Ascenso (May-Jul)",
    "Aguas Altas (Ago-Sep)",
    "Descenso (Oct-Dic)"
]

def generate_plots(csv_path: str, output_dir: str, model_name: str):
    df = pd.read_csv(csv_path, parse_dates=["fecha"])
    df["month"] = df["fecha"].dt.month
    df["Regimen"] = df["month"].apply(get_regime)
    
    # Asegurar que usemos el horizonte t+7
    y_true_col = "y_true_h7" if "y_true_h7" in df.columns else "y_true"
    y_pred_col = "y_pred_h7" if "y_pred_h7" in df.columns else "y_pred_bc"
    
    # Si no existe y_pred_bc probamos con y_pred
    if y_pred_col not in df.columns and "y_pred" in df.columns:
        y_pred_col = "y_pred"
        
    df["Error Absoluto (m)"] = np.abs(df[y_true_col] - df[y_pred_col])
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Scatter Plot por Régimen ──────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, 
        x=y_true_col, 
        y=y_pred_col, 
        hue="Regimen", 
        palette=REGIME_COLORS,
        hue_order=REGIME_ORDER,
        alpha=0.7,
        edgecolor=None
    )
    
    # Linea perfecta
    min_val = min(df[y_true_col].min(), df[y_pred_col].min())
    max_val = max(df[y_true_col].max(), df[y_pred_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label="Predicción Perfecta")
    
    plt.title(f"Diagrama de Dispersión por Régimen Hidrológico\n{model_name} (Horizonte t+7)", fontsize=14, fontweight="bold")
    plt.xlabel("Nivel Real (m)", fontsize=12)
    plt.ylabel("Nivel Predicho (m)", fontsize=12)
    plt.legend(title="Régimen Hidrológico", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    scatter_path = out_dir / f"scatter_regimes_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Guardado: {scatter_path}")

    # ── 2. Box Plot de Error por Régimen ─────────────────────────────────────
    plt.figure(figsize=(10, 6))
    
    # Convertir error a cm para el gráfico
    df["Error Absoluto (cm)"] = df["Error Absoluto (m)"] * 100
    
    sns.boxplot(
        data=df, 
        x="Regimen", 
        y="Error Absoluto (cm)", 
        palette=REGIME_COLORS,
        order=REGIME_ORDER,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
    )
    
    # Imprimir medias por regimen en consola
    print(f"\nMedias de error por régimen para {model_name}:")
    for regime in REGIME_ORDER:
        mean_err = df[df["Regimen"] == regime]["Error Absoluto (cm)"].mean()
        print(f"  {regime}: {mean_err:.1f} cm")
    
    plt.title(f"Distribución del Error Absoluto por Régimen\n{model_name} (Horizonte t+7)", fontsize=14, fontweight="bold")
    plt.xlabel("Régimen Hidrológico", fontsize=12)
    plt.ylabel("Error Absoluto (cm)", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    boxplot_path = out_dir / f"boxplot_regimes_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Guardado: {boxplot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera gráficos por régimen hidrológico.")
    parser.add_argument("--csv", required=True, help="Ruta al predictions_test.csv")
    parser.add_argument("--out", default="report_plots", help="Directorio de salida")
    parser.add_argument("--model", required=True, help="Nombre del modelo para los títulos (ej. 'Ensamble LSTM')")
    
    args = parser.parse_args()
    generate_plots(args.csv, args.out, args.model)
