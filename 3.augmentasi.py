from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf


SPLIT_DIR = Path("dataset/split")
TRAIN_DIR = SPLIT_DIR / "train"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

TARGET_SIZE: Tuple[int, int] = (224, 224)
ROTATION_FACTOR = 20.0 / 360.0
TRANSLATION_FACTOR = 0.05
ZOOM_FACTOR = 0.08
BRIGHTNESS_FACTOR = 0.08
CONTRAST_FACTOR = 0.10
GAUSSIAN_NOISE_STDDEV_RANGE = (2.0, 6.0)
GAUSSIAN_NOISE_PROBABILITY = 0.35
RANDOM_ERASING_AREA_RANGE = (0.02, 0.06)
RANDOM_ERASING_ASPECT_RATIO_RANGE = (0.6, 1.4)
RANDOM_ERASING_PROBABILITY = 0.30


def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def count_train_images_per_class(train_dir: Path) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    if not train_dir.exists() or not train_dir.is_dir():
        return distribution
    for class_dir in sorted([path for path in train_dir.iterdir() if path.is_dir()]):
        distribution[class_dir.name] = len(list_image_files(class_dir))
    return distribution


@tf.keras.utils.register_keras_serializable(package="parkinson")
class SmallGaussianNoise(tf.keras.layers.Layer):
    def __init__(
        self,
        stddev_range: Tuple[float, float] = GAUSSIAN_NOISE_STDDEV_RANGE,
        probability: float = GAUSSIAN_NOISE_PROBABILITY,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stddev_range = (float(stddev_range[0]), float(stddev_range[1]))
        self.probability = float(probability)

    def call(self, inputs, training=None):
        if not training:
            return inputs

        apply_noise = tf.less(tf.random.uniform([], 0.0, 1.0), self.probability)

        def add_noise():
            stddev = tf.random.uniform([], self.stddev_range[0], self.stddev_range[1])
            noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=stddev, dtype=inputs.dtype)
            return tf.clip_by_value(inputs + noise, 0.0, 255.0)

        return tf.cond(apply_noise, add_noise, lambda: inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stddev_range": self.stddev_range,
                "probability": self.probability,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="parkinson")
class SmallRandomErasing(tf.keras.layers.Layer):
    def __init__(
        self,
        area_range: Tuple[float, float] = RANDOM_ERASING_AREA_RANGE,
        aspect_ratio_range: Tuple[float, float] = RANDOM_ERASING_ASPECT_RATIO_RANGE,
        probability: float = RANDOM_ERASING_PROBABILITY,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.area_range = (float(area_range[0]), float(area_range[1]))
        self.aspect_ratio_range = (float(aspect_ratio_range[0]), float(aspect_ratio_range[1]))
        self.probability = float(probability)

    def _erase_single_image(self, image: tf.Tensor) -> tf.Tensor:
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        channels = image_shape[2]

        image_area = tf.cast(height * width, tf.float32)
        erase_area = tf.random.uniform([], self.area_range[0], self.area_range[1]) * image_area
        aspect_ratio = tf.random.uniform(
            [],
            self.aspect_ratio_range[0],
            self.aspect_ratio_range[1],
        )

        erase_height = tf.cast(tf.round(tf.sqrt(erase_area * aspect_ratio)), tf.int32)
        erase_width = tf.cast(tf.round(tf.sqrt(erase_area / aspect_ratio)), tf.int32)

        erase_height = tf.clip_by_value(erase_height, 1, height)
        erase_width = tf.clip_by_value(erase_width, 1, width)

        max_top = tf.maximum(1, height - erase_height + 1)
        max_left = tf.maximum(1, width - erase_width + 1)
        top = tf.random.uniform([], 0, max_top, dtype=tf.int32)
        left = tf.random.uniform([], 0, max_left, dtype=tf.int32)

        patch = tf.random.uniform(
            shape=tf.stack([erase_height, erase_width, channels]),
            minval=0.0,
            maxval=255.0,
            dtype=image.dtype,
        )

        paddings = tf.stack(
            [
                tf.stack([top, height - top - erase_height]),
                tf.stack([left, width - left - erase_width]),
                tf.constant([0, 0], dtype=tf.int32),
            ]
        )
        erase_mask = tf.pad(
            tf.ones(
                shape=tf.stack([erase_height, erase_width, tf.constant(1, dtype=tf.int32)]),
                dtype=image.dtype,
            ),
            paddings,
        )
        erase_mask = tf.broadcast_to(erase_mask, image_shape)
        erase_canvas = tf.pad(patch, paddings)
        erased_image = (image * (1.0 - erase_mask)) + (erase_canvas * erase_mask)
        return tf.clip_by_value(erased_image, 0.0, 255.0)

    def _maybe_erase(self, image: tf.Tensor) -> tf.Tensor:
        should_apply = tf.less(tf.random.uniform([], 0.0, 1.0), self.probability)
        return tf.cond(should_apply, lambda: self._erase_single_image(image), lambda: image)

    def call(self, inputs, training=None):
        if not training:
            return inputs
        return tf.map_fn(
            self._maybe_erase,
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(None, None, None), dtype=inputs.dtype),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "area_range": self.area_range,
                "aspect_ratio_range": self.aspect_ratio_range,
                "probability": self.probability,
            }
        )
        return config


def build_training_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=ROTATION_FACTOR,
                fill_mode="reflect",
                interpolation="bilinear",
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=TRANSLATION_FACTOR,
                width_factor=TRANSLATION_FACTOR,
                fill_mode="reflect",
                interpolation="bilinear",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-ZOOM_FACTOR, ZOOM_FACTOR),
                width_factor=(-ZOOM_FACTOR, ZOOM_FACTOR),
                fill_mode="reflect",
                interpolation="bilinear",
            ),
            tf.keras.layers.RandomBrightness(
                factor=BRIGHTNESS_FACTOR,
                value_range=(0.0, 255.0),
            ),
            tf.keras.layers.RandomContrast(factor=CONTRAST_FACTOR),
            SmallGaussianNoise(),
            SmallRandomErasing(),
        ],
        name="parkinson_train_augmentation",
    )


def describe_augmentation_policy() -> List[str]:
    return [
        "Rotation kecil acak hingga sekitar +/-20 derajat.",
        "Shift/translation kecil hingga sekitar +/-5% tinggi dan lebar.",
        "Zoom ringan acak sekitar +/-8%.",
        "Brightness ringan acak sekitar +/-8%.",
        "Contrast ringan acak sekitar +/-10%.",
        "Gaussian noise ringan dengan stddev acak 2-6 piksel.",
        "Random erasing kecil pada area sekitar 2%-6% gambar.",
        "Augmentasi hanya diterapkan ke split train saat training berlangsung.",
        "File gambar di disk tidak digandakan; versi yang berubah hanya tensor yang dilihat model.",
    ]


def validate_train_split() -> Dict[str, int]:
    if not SPLIT_DIR.exists():
        raise FileNotFoundError("Folder split tidak ditemukan: {}".format(SPLIT_DIR.resolve()))
    if not TRAIN_DIR.exists():
        raise FileNotFoundError("Folder train tidak ditemukan: {}".format(TRAIN_DIR.resolve()))

    class_distribution = count_train_images_per_class(TRAIN_DIR)
    if not class_distribution:
        raise ValueError("Folder train kosong atau tidak memiliki subfolder kelas: {}".format(TRAIN_DIR.resolve()))
    return class_distribution


def main() -> None:
    class_distribution = validate_train_split()
    total_train = sum(class_distribution.values())

    print("\n=== Augmentasi Training On-the-Fly ===")
    print("Sumber train split :", TRAIN_DIR.resolve())
    print("Jumlah data train  :", total_train)
    print("Ukuran input acuan :", "{}x{}".format(TARGET_SIZE[0], TARGET_SIZE[1]))
    print("-" * 60)
    print("Distribusi train per kelas:")
    for class_name in sorted(class_distribution.keys()):
        print("- {}: {} gambar".format(class_name, class_distribution[class_name]))
    print("-" * 60)
    print("Kebijakan augmentasi:")
    for description in describe_augmentation_policy():
        print("- {}".format(description))
    print("-" * 60)
    print("Catatan: skrip ini tidak membuat file gambar baru. Augmentasi dijalankan saat training.")


if __name__ == "__main__":
    main()
