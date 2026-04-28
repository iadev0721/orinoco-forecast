"""
src/evaluation/experiment_tracker.py
======================================
Sistema de registro de experimentos para comparacion multi-modelo.

Cada experimento se guarda en results/experiments/{nombre}/
con estructura estandarizada para que compare_experiments.py
pueda leerlos todos y generar comparaciones automaticas.

Estructura de un experimento:
    results/experiments/{nombre}/
        config_used.yaml        <- config exacta usada en este run
        metrics.json            <- metricas estandarizadas (MAE, RMSE, NSE, KGE)
        training_history.json   <- loss por epoca (solo modelos de DL)
        predictions_test.csv    <- predicciones vs real en test
"""
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("results/experiments")


def _json_safe(obj):
    """Convierte tipos numpy a tipos nativos de Python para serializar a JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    return obj


class ExperimentTracker:
    """Gestor de un experimento individual.

    Uso tipico:
        tracker = ExperimentTracker("lstm_lookback30")
        tracker.log_config(cfg)
        tracker.log_metrics(test_metrics, split="test")
        tracker.log_predictions(dates, y_true, y_pred)
        tracker.log_training_history(history_dict)
        tracker.save()
    """

    def __init__(self, experiment_name: str, model_type: str = "unknown") -> None:
        """
        Args:
            experiment_name: Identificador unico del experimento (sin espacios).
            model_type: Tipo de modelo ('naive', 'lstm', 'transformer').
        """
        self.name        = experiment_name
        self.model_type  = model_type
        self.timestamp   = datetime.now().isoformat()
        self.exp_dir     = EXPERIMENTS_DIR / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self._metrics:    Dict = {}
        self._config:     Dict = {}
        self._history:    Dict = {}
        self._predictions: Optional[pd.DataFrame] = None

        logger.info("Experimento '%s' iniciado en: %s", experiment_name, self.exp_dir)

    def log_config(self, config: dict) -> None:
        """Registra la configuracion exacta usada en este run.

        Args:
            config: Diccionario de configuracion (de config.yaml).
        """
        self._config = config
        config_path = self.exp_dir / "config_used.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        logger.debug("Config guardada en: %s", config_path)

    def log_metrics(self, metrics: Dict[str, float], split: str = "test") -> None:
        """Registra metricas de evaluacion.

        Args:
            metrics: Dict con keys: mae, rmse, nse, kge (y opcionalmente kge_r, kge_alpha, kge_beta).
            split: 'train', 'val' o 'test'.
        """
        self._metrics[split] = _json_safe(metrics)
        logger.info("[%s] MAE=%.3f m | RMSE=%.3f m | NSE=%.4f | KGE=%.4f",
                    split.upper(),
                    metrics.get("mae", float("nan")),
                    metrics.get("rmse", float("nan")),
                    metrics.get("nse", float("nan")),
                    metrics.get("kge", float("nan")))

    def log_training_history(self, history: Dict[str, List[float]]) -> None:
        """Registra la historia de entrenamiento por epoca.

        Args:
            history: Dict con keys 'loss', 'val_loss' (listas de floats).
                     Compatible con Keras history.history.
        """
        self._history = _json_safe(history)

    def log_predictions(
        self,
        dates: pd.DatetimeIndex,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon: int = 7,
    ) -> None:
        """Registra las predicciones sobre el test set para graficado posterior.

        Args:
            dates: Indice temporal de las predicciones (alineado con y_true).
            y_true: Valores reales. Shape: (n,) o (n, horizon).
            y_pred: Predicciones. Shape: (n,) o (n, horizon).
            horizon: Numero de horizontes de prediccion.
        """
        data = {"fecha": dates}

        if y_true.ndim == 1:
            data["y_true"] = y_true
        else:
            for h in range(y_true.shape[1]):
                data[f"y_true_h{h+1}"] = y_true[:, h]

        if y_pred.ndim == 1:
            data["y_pred"] = y_pred
        else:
            for h in range(y_pred.shape[1]):
                data[f"y_pred_h{h+1}"] = y_pred[:, h]

        self._predictions = pd.DataFrame(data)

    def save(self) -> Path:
        """Persiste todos los artefactos del experimento en disco.

        Returns:
            Ruta al directorio del experimento.
        """
        # metrics.json principal
        summary = {
            "experiment_name": self.name,
            "model_type":      self.model_type,
            "timestamp":       self.timestamp,
            "metrics":         self._metrics,
        }
        if self._history:
            best_epoch = int(np.argmin(self._history.get("val_loss", [0])))
            summary["training"] = {
                "epochs_trained":    len(self._history.get("loss", [])),
                "best_epoch":        best_epoch + 1,
                "final_train_loss":  float(self._history["loss"][-1]) if self._history.get("loss") else None,
                "final_val_loss":    float(self._history["val_loss"][-1]) if self._history.get("val_loss") else None,
                "best_val_loss":     float(min(self._history["val_loss"])) if self._history.get("val_loss") else None,
            }

        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        # training_history.json
        if self._history:
            with open(self.exp_dir / "training_history.json", "w") as f:
                json.dump(self._history, f, indent=2)

        # predictions_test.csv
        if self._predictions is not None:
            self._predictions.to_csv(self.exp_dir / "predictions_test.csv", index=False)

        logger.info("Experimento '%s' guardado en: %s", self.name, self.exp_dir)
        return self.exp_dir


def load_all_experiments(experiments_dir: str = "results/experiments") -> List[Dict]:
    """Carga todos los experimentos disponibles para comparacion.

    Args:
        experiments_dir: Ruta raiz donde se almacenan los experimentos.

    Returns:
        Lista de dicts con los datos de cada experimento.
    """
    base = Path(experiments_dir)
    experiments = []
    for exp_path in sorted(base.iterdir()):
        metrics_file = exp_path / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            data = json.load(f)
        # Adjuntar ruta para graficos
        data["_path"] = str(exp_path)
        experiments.append(data)
        logger.debug("Experimento cargado: %s", data["experiment_name"])

    logger.info("Experimentos encontrados: %d", len(experiments))
    return experiments
