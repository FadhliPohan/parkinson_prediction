"""Wrapper optimizer Adam untuk pipeline training (Keras/TensorFlow).

Adam dipakai sebagai optimizer default karena adaptif dan stabil untuk
transfer learning pada citra. Konfigurasi bisa dilewatkan lewat **cfg
sehingga mudah dipanggil dari config YAML atau UI Streamlit.
"""

from __future__ import annotations

from typing import Any


def build_adam(learning_rate: float = 1e-3, **cfg: Any):
    """Bangun optimizer Adam.

    Args:
        learning_rate: Learning rate awal.
        **cfg: Opsi tambahan opsional (`beta_1`, `beta_2`, `epsilon`,
            `weight_decay`). Kunci lain diabaikan agar aman dipanggil dari
            config generik.

    Returns:
        Instance ``tf.keras.optimizers.Adam`` yang siap di-compile.
    """
    import tensorflow as tf  # import lokal agar paket ringan saat hanya introspeksi nama.

    kwargs: dict[str, Any] = {
        "learning_rate": float(learning_rate),
        "beta_1": float(cfg.get("beta_1", 0.9)),
        "beta_2": float(cfg.get("beta_2", 0.999)),
        "epsilon": float(cfg.get("epsilon", 1e-7)),
    }

    weight_decay = cfg.get("weight_decay", None)
    if weight_decay is not None:
        try:
            kwargs["weight_decay"] = float(weight_decay)
        except (TypeError, ValueError):
            pass

    return tf.keras.optimizers.Adam(**kwargs)
