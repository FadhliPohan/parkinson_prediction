from typing import List, Tuple

import tensorflow as tf


PROFILE_NAME = "mycostum_augment"
DISPLAY_NAME = "Augmentasi kustom"
RUN_TAG = "augmentasi_kustom"
AUGMENTATION_ENABLED = True

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


@tf.keras.utils.register_keras_serializable(package="parkinson")
class SmallGaussianNoise(tf.keras.layers.Layer):
    def __init__(
        self,
        stddev_range: Tuple[float, float] = GAUSSIAN_NOISE_STDDEV_RANGE,
        probability: float = GAUSSIAN_NOISE_PROBABILITY,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stddev_range = (float(stddev_range[0]), float(stddev_range[1]))
        self.probability = float(probability)
        self.seed = None if seed is None else int(seed)
        if self.seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(self.seed)

    def call(self, inputs, training=None):
        if not training:
            return inputs

        apply_noise = tf.less(self._rng.uniform([], 0.0, 1.0), self.probability)

        def add_noise():
            stddev = self._rng.uniform([], self.stddev_range[0], self.stddev_range[1])
            noise = self._rng.normal(tf.shape(inputs), mean=0.0, stddev=stddev, dtype=inputs.dtype)
            return tf.clip_by_value(inputs + noise, 0.0, 255.0)

        return tf.cond(apply_noise, add_noise, lambda: inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stddev_range": self.stddev_range,
                "probability": self.probability,
                "seed": self.seed,
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
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.area_range = (float(area_range[0]), float(area_range[1]))
        self.aspect_ratio_range = (float(aspect_ratio_range[0]), float(aspect_ratio_range[1]))
        self.probability = float(probability)
        self.seed = None if seed is None else int(seed)
        if self.seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(self.seed)

    def _erase_single_image(self, image: tf.Tensor) -> tf.Tensor:
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        channels = image_shape[2]

        image_area = tf.cast(height * width, tf.float32)
        erase_area = self._rng.uniform([], self.area_range[0], self.area_range[1]) * image_area
        aspect_ratio = self._rng.uniform(
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
        top = self._rng.uniform([], 0, max_top, dtype=tf.int32)
        left = self._rng.uniform([], 0, max_left, dtype=tf.int32)

        patch = self._rng.uniform(
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
        should_apply = tf.less(self._rng.uniform([], 0.0, 1.0), self.probability)
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
                "seed": self.seed,
            }
        )
        return config


def build_training_augmentation(seed: int | None = None) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=ROTATION_FACTOR,
                fill_mode="reflect",
                interpolation="bilinear",
                seed=seed,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=TRANSLATION_FACTOR,
                width_factor=TRANSLATION_FACTOR,
                fill_mode="reflect",
                interpolation="bilinear",
                seed=None if seed is None else seed + 1,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-ZOOM_FACTOR, ZOOM_FACTOR),
                width_factor=(-ZOOM_FACTOR, ZOOM_FACTOR),
                fill_mode="reflect",
                interpolation="bilinear",
                seed=None if seed is None else seed + 2,
            ),
            tf.keras.layers.RandomBrightness(
                factor=BRIGHTNESS_FACTOR,
                value_range=(0.0, 255.0),
                seed=None if seed is None else seed + 3,
            ),
            tf.keras.layers.RandomContrast(
                factor=CONTRAST_FACTOR,
                seed=None if seed is None else seed + 4,
            ),
            SmallGaussianNoise(seed=None if seed is None else seed + 5),
            SmallRandomErasing(seed=None if seed is None else seed + 6),
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
        "Augmentasi hanya diterapkan ke split train.",
        "Versi augmentasi bersifat statis per run: gambar augmentasi dibangkitkan sekali lalu dipakai berulang selama training.",
    ]
