"""
build_base_dataset.py
=====================
Une todas las fuentes de datos en dataset_orinoco_base.csv.
Rango final: 1992-10-18 -> 2025-02-24 (solapamiento de todas las fuentes).
"""
import pandas as pd

# Cargar fuentes
df_river = pd.read_csv("data/raw/legacy_imputed/orinoco_dataset_legacy_simpleml.csv",
                       parse_dates=["fecha"]).set_index("fecha")
df_nasa  = pd.read_csv("data/processed/nasa_power_radares.csv",
                       parse_dates=["fecha"]).set_index("fecha")
df_guri  = pd.read_csv("data/processed/guri_nivel_diario.csv",
                       parse_dates=["fecha"]).set_index("fecha")
df_enso  = pd.read_csv("data/external/enso_oni_index.csv",
                       parse_dates=["fecha"]).set_index("fecha")

# Recortar ENSO al mismo cutoff de 2025-02-24
df_enso = df_enso.loc[:"2025-02-24"]

# Interpolación de huecos pequeños en el río (≤3 días, e.g. 29-feb)
for col in df_river.columns:
    df_river[col] = df_river[col].interpolate(method="time", limit=3)

# Merge jerárquico: inner con NASA define el rango base, luego left para ENSO y Guri
df = (df_river
      .merge(df_nasa,  left_index=True, right_index=True, how="inner")
      .merge(df_enso,  left_index=True, right_index=True, how="left")
      .merge(df_guri,  left_index=True, right_index=True, how="left"))

# Forward-fill ENSO (mensual → diario)
df["enso_oni"] = df["enso_oni"].ffill()

# Recortar desde primer dato del Guri
df = df.loc["1992-10-18":]

# Reporte
guri_nan = df["guri_nivel_m"].isna().sum()
print(f"Rango final : {df.index.min().date()} -> {df.index.max().date()}")
print(f"Filas       : {len(df)}")
print(f"Columnas    : {len(df.columns)}")
print(f"NaN Guri    : {guri_nan}  (esperado=0, todos los días tienen dato desde 1992)")
print(f"Columnas    : {list(df.columns)}")

df.to_csv("data/processed/dataset_orinoco_base.csv")
print("\n✓ Guardado en data/processed/dataset_orinoco_base.csv")
