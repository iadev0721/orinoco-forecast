"""
src/models/transformer_model.py
=================================
Transformer para series temporales en PyTorch.

El modelo toma una secuencia con forma (batch, lookback, n_features) y aprende
qué días del pasado importan más para predecir un horizonte multistep.

La implementación sigue la tesis:
    - Entrada: secuencias temporales ya escaladas.
    - Bloque lineal + positional encoding.
    - TransformerEncoder con self-attention.
    - Cabeza MLP para producir los 7 pasos futuros.

REGLAS OBLIGATORIAS:
    - R2: semillas fijadas antes de entrenar.
    - R3: baseline_metrics.json debe existir antes de entrenar.
    - R4: aplicar restricciones físicas en inferencia.
    - R5: limpiar cache de PyTorch antes del experimento.
    - R7: hiperparámetros desde config.yaml.
"""
import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _validate_array(name: str, array: np.ndarray) -> None:
    """Verifica que un tensor no contenga NaNs ni infinitos.

    Args:
        name: Nombre lógico del tensor para el mensaje de error.
        array: Arreglo a validar.

    Raises:
        ValueError: Si el arreglo contiene NaN o infinito.
    """
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contiene NaN o infinitos. Revisar el pipeline antes de entrenar.")


class PositionalEncoding(torch.nn.Module):
    """Codificación posicional sinusoidal para secuencias temporales.

    Hereda de nn.Module para que su capa Dropout quede registrada
    como submódulo y se desactive correctamente en model.eval().
    """

    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        import torch
        from torch import nn
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer: el tensor se mueve a GPU con .to(device) automáticamente
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """Aplica la codificación posicional al lote de secuencias."""
        return self.dropout(x + self.pe[:, : x.size(1)])


def build_transformer_model(config: dict, n_features: int, horizon: int):
    """Construye el modelo Transformer según la configuración.

    Args:
        config: Diccionario de configuración cargado desde config.yaml.
        n_features: Número de variables de entrada por día.
        horizon: Número de pasos futuros a predecir.

    Returns:
        Instancia de nn.Module lista para entrenamiento.
    """
    from torch import nn

    transformer_cfg = config["transformer"]
    d_model = int(transformer_cfg.get("d_model", 64))
    nhead = int(transformer_cfg.get("nhead", 4))
    num_layers = int(transformer_cfg.get("num_layers", 2))
    dim_feedforward = int(transformer_cfg.get("dim_feedforward", 128))
    dropout = float(transformer_cfg.get("dropout", 0.1))
    lookback = int(config["lookback_window"])

    class TimeSeriesTransformer(nn.Module):
        """Transformer encoder para regresión multistep."""

        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(n_features, d_model)
            self.positional_encoding = PositionalEncoding(d_model, dropout, lookback)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, horizon),
            )

        def forward(self, x):
            x = self.input_projection(x)
            x = self.positional_encoding(x)
            x = self.encoder(x)
            last_token = x[:, -1, :]
            return self.head(last_token)

    model = TimeSeriesTransformer()
    # PositionalEncoding es nn.Module: queda registrado como submódulo
    # y su dropout se desactiva correctamente en model.eval()
    logger.info(
        "Modelo Transformer construido: lookback=%d, n_features=%d, d_model=%d, nhead=%d, layers=%d, horizon=%d",
        lookback,
        n_features,
        d_model,
        nhead,
        num_layers,
        horizon,
    )
    return model


def train_transformer(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    experiment_name: str = "transformer",
) -> dict:
    """Entrena el Transformer con early stopping.

    Args:
        model: Modelo PyTorch construido por build_transformer_model().
        X_train: Tensor de entrenamiento con forma (n, lookback, n_features).
        y_train: Objetivos de entrenamiento con forma (n, horizon).
        X_val: Tensor de validación.
        y_val: Objetivos de validación.
        config: Configuración del experimento.
        experiment_name: Nombre del experimento para guardar el mejor checkpoint.

    Returns:
        Historial con listas de loss y val_loss por época.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    transformer_cfg = config["transformer"]
    batch_size    = int(transformer_cfg.get("batch_size", 32))
    max_epochs    = int(transformer_cfg.get("max_epochs", 100))
    patience      = int(transformer_cfg.get("patience", 10))
    learning_rate = float(transformer_cfg.get("learning_rate", 1e-4))
    loss_name     = transformer_cfg.get("loss", "mse").lower()
    huber_delta   = float(transformer_cfg.get("huber_delta", 0.5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _validate_array("X_train", X_train)
    _validate_array("y_train", y_train)
    _validate_array("X_val", X_val)
    _validate_array("y_val", y_val)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

    model = model.to(device)
    if loss_name == "huber":
        criterion = nn.HuberLoss(delta=huber_delta)
        logger.info("Loss: HuberLoss(delta=%.2f)", huber_delta)
    else:
        criterion = nn.MSELoss()
        logger.info("Loss: MSELoss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=max(3, patience // 3),
        min_lr=1e-6,
    )

    best_path = Path(f"results/models/{experiment_name}_best.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    history: Dict[str, List[float]] = {"loss": [], "val_loss": []}

    logger.info(
        "Iniciando entrenamiento Transformer: epochs=%d, batch=%d, patience=%d, lr=%g",
        max_epochs,
        batch_size,
        patience,
        learning_rate,
    )

    n_batches = len(train_loader)
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        seen_samples = 0

        print(f"Epoch {epoch + 1}/{max_epochs}", flush=True)

        pbar = tqdm(
            total=n_batches,
            ascii=" \u2501",          # espacio = vacío, ━ = lleno
            colour="green",
            bar_format="{n_fmt}/{total_fmt} {bar:20} {elapsed} {rate_inv_fmt} - {postfix}",
            leave=True,
            dynamic_ncols=True,
        )

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            # Gradient clipping: los Transformers son susceptibles a explosiones
            # del gradiente, especialmente en las primeras épocas.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_actual = batch_x.size(0)
            running_loss += float(loss.item()) * batch_size_actual
            seen_samples += batch_size_actual
            pbar.update(1)
            pbar.set_postfix_str(
                f"loss: {running_loss / max(seen_samples, 1):.4f}"
            )

        train_loss = running_loss / max(seen_samples, 1)

        model.eval()
        val_running_loss = 0.0
        val_seen_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

                batch_size_actual = batch_x.size(0)
                val_running_loss += float(loss.item()) * batch_size_actual
                val_seen_samples += batch_size_actual

        val_loss = val_running_loss / max(val_seen_samples, 1)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        # Actualizar barra con métricas finales (igual que Keras al cerrar la época)
        pbar.set_postfix_str(
            f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
        )
        pbar.refresh()
        pbar.close()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping activado en la epoca {epoch + 1}.")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    return history


def predict_transformer(
    model,
    X: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """Genera predicciones para un lote de secuencias.

    Args:
        model: Modelo entrenado.
        X: Tensor de entrada con forma (n, lookback, n_features).
        batch_size: Tamaño de lote para inferencia.

    Returns:
        Array de predicciones con forma (n, horizon).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _validate_array("X", X)

    device = next(model.parameters()).device
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_pred = model(batch_x).cpu().numpy()
            predictions.append(batch_pred)

    return np.concatenate(predictions, axis=0) if predictions else np.empty((0, 0), dtype=np.float32)
