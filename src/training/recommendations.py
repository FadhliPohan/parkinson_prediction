"""Rekomendasi hyperparameter default + clue per model/method.

Modul ini menjadi sumber tunggal untuk:
  * nilai default yang masuk akal (epoch, batch, fine-tune epoch, learning rate,
    image size) yang menyesuaikan karakter tiap model dan method training, dan
  * "clue" / peringatan ketika setelan user kurang pas (mis. fine-tune diaktifkan
    pada model from-scratch, batch terlalu besar untuk model berat, dst.).

Dipakai oleh dashboard Streamlit (auto-fill + peringatan) dan bisa juga dipakai
modul lain bila perlu. Sengaja TIDAK bergantung pada TensorFlow/Streamlit agar
ringan untuk diimport di mana saja.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# Profil dasar per kategori model. Dipilih berdasarkan model_id/family agar
# default menyesuaikan kebutuhan memori & konvergensi tiap arsitektur.
@dataclass(frozen=True)
class _ModelProfile:
    epochs: int
    batch_size: int
    fine_tune_epochs: int
    learning_rate: float
    image_size: int
    pretrained: bool
    supports_fine_tune: bool
    max_batch_size: int
    notes: List[str] = field(default_factory=list)


# Model CNN ringan (pretrained ImageNet, transfer learning standar).
_LIGHT_CNN = _ModelProfile(
    epochs=25,
    batch_size=16,
    fine_tune_epochs=10,
    learning_rate=1e-3,
    image_size=224,
    pretrained=True,
    supports_fine_tune=True,
    max_batch_size=32,
    notes=["CNN pretrained ImageNet: transfer learning + fine-tuning bertahap cocok."],
)

# Model CNN berat (butuh batch lebih kecil agar tidak OOM di GPU umum).
_HEAVY_CNN = _ModelProfile(
    epochs=25,
    batch_size=8,
    fine_tune_epochs=10,
    learning_rate=1e-3,
    image_size=224,
    pretrained=True,
    supports_fine_tune=True,
    max_batch_size=16,
    notes=["Model besar/berat: pakai batch kecil (<=16) agar aman dari out-of-memory GPU."],
)

# ResNeXt50 di repo ini implementasi kustom (bobot acak / from-scratch).
_RESNEXT_SCRATCH = _ModelProfile(
    epochs=40,
    batch_size=16,
    fine_tune_epochs=0,
    learning_rate=1e-3,
    image_size=224,
    pretrained=False,
    supports_fine_tune=False,
    max_batch_size=32,
    notes=[
        "ResNeXt50 di sini dibangun dari bobot acak (bukan pretrained).",
        "Butuh lebih banyak epoch; fine-tune bertahap tidak relevan (tidak ada backbone pretrained).",
    ],
)

# Transformer (ViT/Swin/DeiT) pada repo ini dilatih from-scratch.
_TRANSFORMER = _ModelProfile(
    epochs=60,
    batch_size=8,
    fine_tune_epochs=0,
    learning_rate=3e-4,
    image_size=224,
    pretrained=False,
    supports_fine_tune=False,
    max_batch_size=16,
    notes=[
        "Transformer (ViT/Swin/DeiT) dilatih from-scratch: butuh epoch jauh lebih banyak.",
        "Pakai learning rate lebih kecil (~3e-4) dan batch kecil; fine-tune bertahap tidak dipakai.",
    ],
)

# YOLOv8 classifier (dikelola ultralytics).
_YOLO = _ModelProfile(
    epochs=50,
    batch_size=16,
    fine_tune_epochs=0,
    learning_rate=1e-3,
    image_size=224,
    pretrained=True,
    supports_fine_tune=False,
    max_batch_size=32,
    notes=[
        "YOLOv8 classifier dikelola ultralytics: parameter epoch/batch dipakai langsung.",
        "Fine-tune epochs tidak dipakai untuk YOLO (gunakan epoch utama saja).",
    ],
)


_HEAVY_CNN_IDS = {"resnet152", "vgg19", "vgg16"}
_RESNEXT_IDS = {"resnext50"}


@dataclass
class HyperRecommendation:
    epochs: int
    batch_size: int
    fine_tune_epochs: int
    learning_rate: float
    image_size: int
    pretrained: bool
    supports_fine_tune: bool
    max_batch_size: int
    notes: List[str]


def _base_profile(model_id: str, family: str, framework: str) -> _ModelProfile:
    model_id = str(model_id).strip().lower()
    family = str(family).strip().lower()
    framework = str(framework).strip().lower()

    if framework == "yolo" or family == "yolo":
        return _YOLO
    if family == "transformer":
        return _TRANSFORMER
    if model_id in _RESNEXT_IDS:
        return _RESNEXT_SCRATCH
    if model_id in _HEAVY_CNN_IDS:
        return _HEAVY_CNN
    return _LIGHT_CNN


def recommend_hyperparams(
    model_id: str,
    family: str,
    framework: str,
    method_id: str = "transfer_learning",
) -> HyperRecommendation:
    """Hitung rekomendasi hyperparameter untuk kombinasi model + method."""
    profile = _base_profile(model_id, family, framework)
    method_id = str(method_id).strip().lower()

    epochs = profile.epochs
    batch_size = profile.batch_size
    fine_tune_epochs = profile.fine_tune_epochs
    learning_rate = profile.learning_rate
    notes = list(profile.notes)

    supports_fine_tune = profile.supports_fine_tune
    pretrained = profile.pretrained

    # Penyesuaian berdasarkan method training.
    if method_id == "baseline":
        # Baseline = from-scratch, tanpa fine-tuning.
        pretrained = False
        supports_fine_tune = False
        fine_tune_epochs = 0
        epochs = max(epochs, 30)
        notes.append("Method baseline: bobot acak (from-scratch), fine-tune dimatikan, epoch dinaikkan.")
    elif method_id == "full_fine_tuning":
        if supports_fine_tune:
            fine_tune_epochs = max(fine_tune_epochs, 15)
            notes.append("Method full_fine_tuning: fine-tune lebih panjang (>=15 epoch).")
    elif method_id == "transfer_learning_mixed_precision":
        notes.append("Mixed precision aktif: boleh menaikkan batch size (hemat memori GPU).")
    elif method_id == "transfer_learning":
        if supports_fine_tune:
            fine_tune_epochs = max(fine_tune_epochs, 10)

    if not supports_fine_tune:
        fine_tune_epochs = 0

    return HyperRecommendation(
        epochs=int(epochs),
        batch_size=int(batch_size),
        fine_tune_epochs=int(fine_tune_epochs),
        learning_rate=float(learning_rate),
        image_size=int(profile.image_size),
        pretrained=bool(pretrained),
        supports_fine_tune=bool(supports_fine_tune),
        max_batch_size=int(profile.max_batch_size),
        notes=notes,
    )


def evaluate_settings(
    rec: HyperRecommendation,
    *,
    epochs: int,
    batch_size: int,
    fine_tune_epochs: int,
    learning_rate: float,
) -> List[str]:
    """Bandingkan setelan user dengan rekomendasi, hasilkan daftar clue/peringatan.

    Daftar kosong berarti setelan sudah dalam rentang wajar.
    """
    clues: List[str] = []

    if epochs < max(5, int(rec.epochs * 0.5)):
        clues.append(
            f"Epoch ({epochs}) terlihat terlalu kecil untuk model ini. "
            f"Rekomendasi sekitar {rec.epochs} epoch agar konvergen."
        )

    if batch_size > rec.max_batch_size:
        clues.append(
            f"Batch size ({batch_size}) melebihi batas aman (~{rec.max_batch_size}) untuk model ini; "
            "berisiko out-of-memory. Pertimbangkan batch lebih kecil."
        )

    if not rec.supports_fine_tune and fine_tune_epochs > 0:
        clues.append(
            f"Fine-tune epochs ({fine_tune_epochs}) tidak berpengaruh: model ini dilatih from-scratch "
            "(tidak ada backbone pretrained untuk di-fine-tune). Set 0."
        )

    if rec.supports_fine_tune and fine_tune_epochs == 0:
        clues.append(
            "Fine-tune epochs = 0 pada model pretrained: Anda melewati tahap fine-tuning. "
            f"Rekomendasi {rec.fine_tune_epochs} epoch untuk hasil lebih optimal."
        )

    if learning_rate > rec.learning_rate * 5:
        clues.append(
            f"Learning rate ({learning_rate:g}) jauh lebih besar dari rekomendasi ({rec.learning_rate:g}); "
            "berisiko training tidak stabil/divergen."
        )

    return clues
