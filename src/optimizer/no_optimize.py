"""Baseline optimizer "tanpa optimisasi" sebagai pembanding.

Konteksnya: di Keras setiap ``model.fit`` tetap membutuhkan sebuah optimizer
untuk melakukan update bobot. Yang dimaksud "no optimize" di sini adalah
*baseline polos*: SGD biasa tanpa momentum, tanpa Nesterov, dan tanpa
penjadwalan/tuning learning rate. Tujuannya untuk membandingkan seberapa besar
kontribusi optimizer adaptif (mis. Adam) terhadap hasil training.
"""

from __future__ import annotations

from typing import Any


def build_no_optimize(learning_rate: float = 1e-3, **cfg: Any):
    """Bangun optimizer baseline (SGD plain tanpa tuning).

    Args:
        learning_rate: Learning rate tetap.
        **cfg: Diabaikan (sengaja, agar benar-benar tanpa tuning). Tetap
            diterima supaya antarmuka seragam dengan optimizer lain.

    Returns:
        Instance ``tf.keras.optimizers.SGD`` polos (momentum=0, nesterov=False).
    """
    import tensorflow as tf  # import lokal agar paket ringan saat hanya introspeksi nama.

    return tf.keras.optimizers.SGD(
        learning_rate=float(learning_rate),
        momentum=0.0,
        nesterov=False,
    )
