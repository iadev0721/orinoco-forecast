"""
src/utils/gpu_config.py
========================
Configuración de GPU para RTX 3060 (6 GB VRAM).

REGLA R5: Este módulo DEBE llamarse al inicio de cualquier script
que use TensorFlow o PyTorch. Es obligatorio, no opcional.

Configuración:
    TensorFlow: memory growth activado (no reserva toda la VRAM)
    PyTorch: cache limpiado al inicio
    Batch sizes máximos:
        - LSTM: 64
        - Transformer: 32 (más agresivo en memoria)
"""
import logging

logger = logging.getLogger(__name__)


def configure_tensorflow_gpu() -> None:
    """Configura TensorFlow para no reservar toda la VRAM de la RTX 3060.

    OBLIGATORIO llamar antes de construir cualquier modelo Keras.

    Raises:
        ImportError: Si TensorFlow no está instalado.
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU configurada: {len(gpus)} dispositivo(s) con memory growth activado.")
        else:
            logger.warning("No se detectó GPU. El entrenamiento usará CPU (más lento).")
    except ImportError:
        logger.error("TensorFlow no está instalado. Instalar con: pip install tensorflow")
        raise


def configure_pytorch_gpu() -> None:
    """Limpia la cache de VRAM de PyTorch al inicio del experimento.

    OBLIGATORIO llamar antes de entrenar el Transformer.

    Raises:
        ImportError: Si PyTorch no está instalado.
    """
    try:
        import torch

        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"PyTorch GPU: {device_name} ({vram_gb:.1f} GB VRAM). Cache limpiada.")
        else:
            logger.warning("CUDA no disponible. El Transformer entrenará en CPU.")
    except ImportError:
        logger.error("PyTorch no está instalado. Instalar con: pip install torch")
        raise


def get_device() -> str:
    """Retorna el dispositivo disponible para PyTorch.

    Returns:
        'cuda' si hay GPU disponible, 'cpu' en caso contrario.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
