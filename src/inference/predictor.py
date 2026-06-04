from __future__ import annotations

import io
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_transformer_input(inputs: np.ndarray) -> np.ndarray:
    return (inputs.astype(np.float32) / 127.5) - 1.0


# CATATAN: peta ini TIDAK lagi dipakai untuk preprocessing di inference.
# Layer preprocessing sudah tertanam di dalam model TF saat training, jadi
# inference hanya boleh memberi piksel mentah [0,255] (lihat prepare_tf_input).
# Peta dipertahankan sebagai dokumentasi pemetaan preprocessing per-model.
# JANGAN menerapkannya lagi pada input inference (akan terjadi double preprocessing).
MODEL_PREPROCESSORS = {
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "vgg16": tf.keras.applications.vgg16.preprocess_input,
    "vgg19": tf.keras.applications.vgg19.preprocess_input,
    "resnext50": tf.keras.applications.resnet50.preprocess_input,
    "resnet152": tf.keras.applications.resnet.preprocess_input,
    "inception_googlenet": tf.keras.applications.inception_v3.preprocess_input,
    "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
    "densenet121": tf.keras.applications.densenet.preprocess_input,
    "transformer": preprocess_transformer_input,
}


def prepare_tf_input(image_bytes: bytes, target_size: Tuple[int, int], preprocess_key: str) -> np.ndarray:
    # PENTING (BLOCKER-01): model TF disimpan dengan layer preprocessing sebagai
    # bagian dari graph-nya (lihat training_common.py: `x = preprocess_fn(inputs)`),
    # sehingga model sudah melakukan normalisasi sendiri di dalam.
    # Karena itu inference WAJIB memberi piksel mentah RGB [0,255] dan TIDAK boleh
    # menerapkan preprocessing lagi di sini. Menerapkannya ulang membuat normalisasi
    # terjadi dua kali (mis. ViT: x/127.5-1 dilakukan 2x) dan merusak prediksi.
    # `preprocess_key` sengaja dipertahankan di signature untuk kompatibilitas API
    # dan tidak dipakai untuk preprocessing apa pun. Lihat MODEL_PREPROCESSORS di atas
    # yang hanya berfungsi sebagai dokumentasi pemetaan preprocessing per-model.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def prepare_yolo_input(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def normalize_probabilities(probs_raw: np.ndarray, class_names: List[str]) -> Tuple[List[str], List[float]]:
    probs_raw = np.asarray(probs_raw, dtype=np.float32).reshape(-1)

    if probs_raw.size == 1:
        positive_prob = float(max(0.0, min(1.0, probs_raw[0])))
        classes = class_names[:2] if len(class_names) >= 2 else ["class_0", "class_1"]
        probs = [1.0 - positive_prob, positive_prob]
        return classes, probs

    probs = probs_raw.astype(float).tolist()
    total = float(sum(probs))
    if total > 0:
        probs = [p / total for p in probs]

    if len(class_names) != len(probs):
        classes = [f"class_{idx}" for idx in range(len(probs))]
    else:
        classes = class_names

    return classes, probs


def predict_from_bundle(
    model_bundle: Dict[str, object],
    image_bytes: bytes,
    preprocess_key: str,
) -> Dict[str, object]:
    model = model_bundle["model"]
    framework = str(model_bundle.get("framework", "tensorflow"))
    class_names = list(model_bundle.get("class_names", []))

    if framework == "yolo":
        input_image = prepare_yolo_input(image_bytes)
        prediction = model.predict(source=input_image, verbose=False)
        if not prediction:
            raise RuntimeError("Prediksi YOLO tidak menghasilkan output")
        probs_tensor = prediction[0].probs
        if probs_tensor is None:
            raise RuntimeError("Output YOLO tidak mengandung probabilitas")
        probs_raw = np.asarray(probs_tensor.data.cpu(), dtype=np.float32).reshape(-1)
    else:
        input_shape = model.input_shape
        target_height = int(input_shape[1]) if len(input_shape) > 2 and input_shape[1] else 224
        target_width = int(input_shape[2]) if len(input_shape) > 2 and input_shape[2] else 224
        input_batch = prepare_tf_input(
            image_bytes=image_bytes,
            target_size=(target_width, target_height),
            preprocess_key=preprocess_key,
        )
        prediction = model.predict(input_batch, verbose=0)
        probs_raw = np.asarray(prediction).reshape(-1)

    classes, probabilities = normalize_probabilities(probs_raw, class_names)
    best_index = int(np.argmax(probabilities))

    return {
        "predicted_label": classes[best_index],
        "confidence": float(probabilities[best_index]),
        "class_names": classes,
        "probabilities": probabilities,
    }
