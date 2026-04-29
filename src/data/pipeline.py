"""
src/data/pipeline.py
=====================
Pipeline central de preparación de datos para entrenamiento.

Responsabilidades:
    1. Cargar dataset_orinoco_features.csv
    2. Split cronológico estricto (R1: sin leakage)
    3. Fit MinMaxScaler SOLO sobre train → serializar con joblib
    4. Generar tensores 3D: (samples, lookback, n_features)
    5. Selección de features según target station (excluye downstream)

REGLA R1: El scaler NUNCA ve datos de val o test.
REGLA R7: Parámetros leídos de config.yaml.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Orden hidrológico de las estaciones (aguas arriba → aguas abajo)
STATION_ORDER: List[str] = ["ayacucho", "caicara", "ciudad_bolivar", "palua"]


def load_config(config_path: str = "config.yaml") -> dict:
    """Carga la configuración central del experimento.

    Args:
        config_path: Ruta al archivo config.yaml.

    Returns:
        Diccionario con la configuración completa.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_columns(target_station: str, df: pd.DataFrame) -> List[str]:
    """Selecciona las columnas de features excluyendo el target y estaciones downstream.

    REGLA R5 (Paluacentrismo): Ninguna estación tiene prioridad fija.
    La selección es dinámica según el target definido en config.yaml.

    Args:
        target_station: Estación objetivo (ej. 'palua').
        df: DataFrame completo con todas las columnas.

    Returns:
        Lista de nombres de columnas a usar como features de entrada.
    """
    # Índice de la estación target en el orden hidrológico
    target_idx = STATION_ORDER.index(target_station)

    # Excluir target + todas las estaciones downstream (aguas abajo del target)
    excluded_stations = STATION_ORDER[target_idx:]  # target y cualquier estación más abajo

    # Columnas a excluir: cualquier columna cuyo nombre comience con una estación excluida
    excluded_cols = [
        col for col in df.columns
        if any(col == station or col.startswith(station + "_") for station in excluded_stations)
    ]
    # También excluir la columna fecha si aparece como columna (no solo como índice)
    excluded_cols += [c for c in ["fecha"] if c in df.columns]

    feature_cols = [c for c in df.columns if c not in excluded_cols]
    logger.info(
        "Target: '%s' | Features: %d columnas | Excluidas: %s",
        target_station, len(feature_cols), excluded_cols
    )
    return feature_cols


def split_data(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el DataFrame en train/val/test de forma cronológica estricta.

    REGLA R1: No hay shuffle. El tiempo avanza hacia adelante.

    Args:
        df: DataFrame con índice DatetimeIndex ordenado.
        train_end: Fecha de fin del conjunto de entrenamiento (inclusive).
        val_end: Fecha de fin del conjunto de validación (inclusive).

    Returns:
        Tupla (df_train, df_val, df_test).
    """
    df_train = df.loc[:train_end]
    df_val   = df.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1): val_end]
    df_test  = df.loc[pd.Timestamp(val_end) + pd.Timedelta(days=1):]

    logger.info("Split cronológico:")
    logger.info("  Train : %s → %s | %d filas", df_train.index.min().date(),
                df_train.index.max().date(), len(df_train))
    logger.info("  Val   : %s → %s | %d filas", df_val.index.min().date(),
                df_val.index.max().date(), len(df_val))
    logger.info("  Test  : %s → %s | %d filas", df_test.index.min().date(),
                df_test.index.max().date(), len(df_test))
    return df_train, df_val, df_test


def fit_scaler(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    scaler_path: str = "results/models/scaler.joblib",
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Ajusta MinMaxScaler(0,1) EXCLUSIVAMENTE sobre los datos de entrenamiento.

    REGLA R1: fit() solo con train. Guarda scaler en disco para inferencia.

    Args:
        df_train: DataFrame de entrenamiento.
        feature_cols: Columnas de features de entrada.
        target_col: Columna del target (se escala por separado para invertir después).
        scaler_path: Ruta donde guardar el scaler de features.

    Returns:
        Tupla (scaler_features, scaler_target).
    """
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(df_train[feature_cols])
    scaler_y.fit(df_train[[target_col]])

    # Persistencia para inferencia futura (R1: no refitear nunca con val/test)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y,
                 "feature_cols": feature_cols, "target_col": target_col},
                scaler_path)
    logger.info("Scalers guardados en: %s", scaler_path)
    return scaler_X, scaler_y


def make_sequences(
    data_X: np.ndarray,
    data_y: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera secuencias 3D para LSTM con target multi-step.

    Formato de entrada LSTM: (samples, timesteps, features)
    Formato del target: (samples, horizon) → predice los próximos 'horizon' días

    La ventana se construye así para cada muestra i:
        X[i] = data_X[i : i+lookback]           → historia de 'lookback' días
        y[i] = data_y[i+lookback : i+lookback+horizon] → próximos 'horizon' días

    NOTA ANTI-LEAKAGE: data_X y data_y deben venir YA escalados.

    Args:
        data_X: Array 2D escalado de features. Shape: (n, n_features).
        data_y: Array 1D escalado del target. Shape: (n,).
        lookback: Días de historia como input.
        horizon: Días a predecir.

    Returns:
        Tupla (X, y):
            X shape: (samples, lookback, n_features)
            y shape: (samples, horizon)
    """
    X, y = [], []
    max_i = len(data_X) - lookback - horizon + 1
    for i in range(max_i):
        X.append(data_X[i: i + lookback])
        y.append(data_y[i + lookback: i + lookback + horizon])
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    logger.debug("Sequences: X=%s, y=%s", X_arr.shape, y_arr.shape)
    return X_arr, y_arr


def build_tensors(
    config_path: str = "config.yaml",
    features_path: str = "data/processed/dataset_orinoco_features.csv",
    scaler_path: str = "results/models/scaler.joblib",
    cfg_override: dict = None,
) -> Dict[str, np.ndarray]:
    """Función principal: carga, divide, escala y genera todos los tensores.

    Args:
        config_path: Ruta al config.yaml (se ignora si se pasa cfg_override).
        features_path: Ruta al CSV de features.
        scaler_path: Ruta donde guardar el scaler.
        cfg_override: Si se pasa, usa este dict en lugar de leer config.yaml.
                      Útil para aplicar overrides de CLI (--lookback, --units, etc.)
                      sin que build_tensors sobreescriba esos valores al releer el YAML.

    Uso:
        tensors = build_tensors()
        X_train, y_train = tensors["X_train"], tensors["y_train"]
        X_val,   y_val   = tensors["X_val"],   tensors["y_val"]
        X_test,  y_test  = tensors["X_test"],  tensors["y_test"]

    Returns:
        Diccionario con X_train, y_train, X_val, y_val, X_test, y_test
        y metadata adicional (feature_cols, target_col, dates_test).
    """
    cfg = cfg_override if cfg_override is not None else load_config(config_path)

    target_station: str = cfg["target_station"]
    target_col: str     = target_station
    lookback: int       = cfg["lookback_window"]
    horizon: int        = cfg["forecast_horizon"]
    train_end: str      = cfg["train_end"]
    val_end: str        = cfg["val_end"]

    # 1. Cargar dataset
    logger.info("Cargando dataset de features: %s", features_path)
    df = pd.read_csv(features_path, parse_dates=["fecha"]).set_index("fecha").sort_index()
    logger.info("  Rango: %s → %s | %d filas | %d cols",
                df.index.min().date(), df.index.max().date(), len(df), len(df.columns))

    # 2. Selección de features (excluye target y downstream)
    feature_cols = get_feature_columns(target_station, df)

    # 3. Split cronológico
    df_train, df_val, df_test = split_data(df, train_end, val_end)

    # 4. Escalar (fit SOLO en train)
    scaler_X, scaler_y = fit_scaler(df_train, feature_cols, target_col, scaler_path)

    def scale_partition(df_part: pd.DataFrame):
        X_scaled = scaler_X.transform(df_part[feature_cols])
        y_scaled = scaler_y.transform(df_part[[target_col]]).ravel()
        return X_scaled, y_scaled

    X_tr_s, y_tr_s = scale_partition(df_train)
    X_va_s, y_va_s = scale_partition(df_val)
    X_te_s, y_te_s = scale_partition(df_test)

    # 5. Generar ventanas 3D
    X_train, y_train = make_sequences(X_tr_s, y_tr_s, lookback, horizon)
    X_val,   y_val   = make_sequences(X_va_s, y_va_s, lookback, horizon)
    X_test,  y_test  = make_sequences(X_te_s, y_te_s, lookback, horizon)

    logger.info("Tensores generados:")
    logger.info("  X_train: %s  y_train: %s", X_train.shape, y_train.shape)
    logger.info("  X_val  : %s  y_val  : %s", X_val.shape,   y_val.shape)
    logger.info("  X_test : %s  y_test : %s", X_test.shape,  y_test.shape)

    # Índices de fechas del test (desplazados por lookback+horizon-1 para alinear con y)
    test_dates = df_test.index[lookback + horizon - 1: lookback + horizon - 1 + len(y_test)]

    return {
        "X_train": X_train,   "y_train": y_train,
        "X_val":   X_val,     "y_val":   y_val,
        "X_test":  X_test,    "y_test":  y_test,
        "feature_cols": feature_cols,
        "target_col":   target_col,
        "scaler_X":     scaler_X,
        "scaler_y":     scaler_y,
        "test_dates":   test_dates,
        # Series originales sin escalar (para baseline y métricas inversas)
        "y_test_raw":   df_test[target_col].values,
        "df_test":      df_test,
    }
