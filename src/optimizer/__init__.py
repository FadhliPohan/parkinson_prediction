"""Registry optimizer dengan antarmuka seragam.

Pemakaian:

    from src.optimizer import get_optimizer
    optimizer = get_optimizer("adam", learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=..., metrics=...)

Catatan desain (asumsi eksplisit):
    Pipeline training project ini berbasis Keras/TensorFlow yang memakai pola
    *define-and-compile*, sehingga optimizer dibangun dari ``learning_rate`` +
    konfigurasi, BUKAN dari daftar parameter model ala PyTorch
    (``model.parameters()``). Karena itu signature factory di sini adalah
    ``get_optimizer(name, learning_rate, **cfg)``. Untuk YOLO (Ultralytics),
    pemilihan optimizer ditangani via argumen ``optimizer=`` pada
    ``model.train`` menggunakan pemetaan di :data:`YOLO_OPTIMIZER_MAP`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .adam import build_adam
from .no_optimize import build_no_optimize

# Nama optimizer -> fungsi builder (Keras/TensorFlow).
OPTIMIZER_BUILDERS: Dict[str, Callable[..., Any]] = {
    "adam": build_adam,
    "no_optimize": build_no_optimize,
}

# Deskripsi singkat untuk ditampilkan di UI/CLI.
OPTIMIZER_DESCRIPTIONS: Dict[str, str] = {
    "adam": "Adam optimizer (adaptif, rekomendasi default).",
    "no_optimize": "Baseline SGD plain tanpa tuning (pembanding).",
}

# Pemetaan nama optimizer project -> nama optimizer Ultralytics (YOLOv8).
YOLO_OPTIMIZER_MAP: Dict[str, str] = {
    "adam": "Adam",
    "no_optimize": "SGD",
}

DEFAULT_OPTIMIZER = "adam"


def list_optimizers() -> List[str]:
    """Daftar nama optimizer yang tersedia (terurut)."""
    return sorted(OPTIMIZER_BUILDERS.keys())


def normalize_optimizer_name(name: str | None) -> str:
    """Normalisasi nama optimizer; fallback ke default bila kosong/invalid."""
    key = str(name or DEFAULT_OPTIMIZER).strip().lower()
    return key if key in OPTIMIZER_BUILDERS else DEFAULT_OPTIMIZER


def get_optimizer(name: str, learning_rate: float = 1e-3, **cfg: Any):
    """Factory optimizer seragam untuk pipeline Keras.

    Args:
        name: Nama optimizer (``"adam"`` atau ``"no_optimize"``).
        learning_rate: Learning rate awal.
        **cfg: Opsi tambahan yang diteruskan ke builder terkait.

    Returns:
        Instance ``tf.keras.optimizers.Optimizer``.

    Raises:
        KeyError: Bila ``name`` tidak terdaftar.
    """
    key = str(name or "").strip().lower()
    if key not in OPTIMIZER_BUILDERS:
        available = ", ".join(list_optimizers())
        raise KeyError(f"Optimizer '{name}' tidak dikenal. Tersedia: {available}")
    return OPTIMIZER_BUILDERS[key](learning_rate=learning_rate, **cfg)


def get_yolo_optimizer_name(name: str) -> str:
    """Konversi nama optimizer project ke nama optimizer Ultralytics/YOLOv8."""
    return YOLO_OPTIMIZER_MAP.get(normalize_optimizer_name(name), "auto")
