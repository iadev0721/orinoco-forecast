"""
src/utils/reproducibility.py
==============================
Seeds y determinismo para reproducibilidad científica.

REGLA R2: Toda ejecución DEBE llamar set_global_seeds() al inicio.
No reimplementar este bloque en cada notebook/script.

Seeds fijados:
    - NumPy
    - TensorFlow (y Keras)
    - PyTorch (CPU y CUDA)
    - Python random
    - PYTHONHASHSEED (variable de entorno)

Nota: El determinismo completo en GPU requiere configuración adicional
y puede afectar el rendimiento. Documentar en la tesis.
"""
import logging
import os
import random

logger = logging.getLogger(__name__)

GLOBAL_SEED = 42  # Fijo en todo el proyecto. Leer de config.yaml si se desea cambiar.


def set_global_seeds(seed: int = GLOBAL_SEED) -> None:
    """Fija todas las fuentes de aleatoriedad del entorno.

    REGLA R2: Llamar al inicio de cada notebook y script de entrenamiento.

    Args:
        seed: Semilla global. Por defecto: 42 (de config.yaml).
    """
    # Python
    random.seed(seed)

    # Variable de entorno para hash seeds
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    import numpy as np
    np.random.seed(seed)

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.debug(f"TensorFlow seed fijado: {seed}")
    except ImportError:
        logger.debug("TensorFlow no instalado, seed no fijado para TF.")

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Para determinismo completo en CUDA (puede afectar rendimiento):
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch seed fijado: {seed}")
    except ImportError:
        logger.debug("PyTorch no instalado, seed no fijado para PT.")

    logger.info(f"Seeds globales fijados: {seed}")


def log_environment_versions() -> dict:
    """Registra las versiones del entorno para reproducibilidad.

    REGLA R2: Documentar versiones de todas las librerías clave.

    Returns:
        Dict con versiones: Python, NumPy, Pandas, TensorFlow, PyTorch, CUDA.
    """
    import platform
    import sys

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
    }

    try:
        import numpy as np
        versions["numpy"] = np.__version__
    except ImportError:
        pass

    try:
        import pandas as pd
        versions["pandas"] = pd.__version__
    except ImportError:
        pass

    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import tensorflow as tf
        versions["tensorflow"] = tf.__version__
        try:
            import keras
            versions["keras"] = keras.__version__
        except ImportError:
            versions["keras"] = getattr(tf.keras, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import torch
        versions["torch"] = torch.__version__
        versions["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions["cuda_version"] = torch.version.cuda
            versions["cudnn_version"] = torch.backends.cudnn.version()
    except ImportError:
        pass

    try:
        import mlflow
        versions["mlflow"] = mlflow.__version__
    except ImportError:
        pass

    for lib, ver in versions.items():
        logger.info(f"  {lib}: {ver}")

    return versions
