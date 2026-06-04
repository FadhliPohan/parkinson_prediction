"""ResNeXt backbone implementations untuk Parkinson classification.

Mengimplementasikan ResNeXt-50-32x4d dari scratch menggunakan grouped convolutions
(tf.keras.layers.Conv2D dengan parameter `groups`). TensorFlow tidak menyediakan
ResNeXt secara native di tf.keras.applications, sehingga arsitektur dibangun manual.

Referensi: "Aggregated Residual Transformations for Deep Neural Networks"
           Xie et al., 2017 (https://arxiv.org/abs/1611.05431)
"""

from typing import Optional, Tuple

import tensorflow as tf

# ResNeXt-50-32x4d: cardinality=32, base_width=4
_CARDINALITY = 32
_BASE_WIDTH = 4


def resnext50_preprocess_input(inputs: tf.Tensor) -> tf.Tensor:
    """Preprocessing ResNet-style (channel-wise mean subtraction, BGR order).

    ResNeXt50 termasuk keluarga ResNet sehingga menggunakan statistik ImageNet
    yang sama. Kompatibel dengan KerasTensor (Functional API) maupun numpy array.
    """
    return tf.keras.applications.resnet50.preprocess_input(inputs)


def _validate_resnext_args(
    include_top: bool,
    weights: Optional[str],
    input_shape: Optional[Tuple],
    backbone_name: str,
) -> Tuple[int, int, int]:
    if include_top:
        raise ValueError("{} hanya mendukung include_top=False.".format(backbone_name))
    if weights not in (None, "imagenet"):
        raise ValueError(
            "Argumen weights={} tidak didukung untuk {}.".format(weights, backbone_name)
        )
    if input_shape is None or len(input_shape) != 3:
        raise ValueError(
            "input_shape tidak valid untuk {}: {}".format(backbone_name, input_shape)
        )
    return (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))


def _conv_bn_relu(x: tf.Tensor, filters: int, kernel_size: int, strides: int,
                  padding: str, groups: int, use_bias: bool, name: str) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=use_bias,
        name="{}_conv".format(name),
    )(x)
    x = tf.keras.layers.BatchNormalization(name="{}_bn".format(name))(x)
    x = tf.keras.layers.Activation("relu", name="{}_relu".format(name))(x)
    return x


def _resnext_bottleneck(
    inputs: tf.Tensor,
    planes: int,
    stride: int,
    cardinality: int,
    base_width: int,
    is_projection: bool,
    prefix: str,
) -> tf.Tensor:
    """Satu bottleneck block ResNeXt dengan Aggregated Residual Transformations.

    width = int(planes * (base_width / 64.0)) * cardinality
    Untuk ResNeXt-50-32x4d pada stage pertama (planes=64):
      width = int(64 * 4/64) * 32 = 128
    """
    width = int(planes * (base_width / 64.0)) * cardinality
    out_channels = planes * 4

    # 1×1 pointwise projection → mengurangi channels ke `width`
    x = _conv_bn_relu(inputs, width, 1, 1, "valid", 1, False, "{}_pw1".format(prefix))

    # 3×3 grouped convolution (cardinality kelompok transformasi paralel)
    x = _conv_bn_relu(x, width, 3, stride, "same", cardinality, False, "{}_gconv".format(prefix))

    # 1×1 pointwise expansion → mengembalikan ke out_channels (planes * 4)
    x = tf.keras.layers.Conv2D(
        out_channels, 1, use_bias=False, name="{}_pw2_conv".format(prefix)
    )(x)
    x = tf.keras.layers.BatchNormalization(name="{}_pw2_bn".format(prefix))(x)

    # Shortcut: gunakan projection conv jika dimensi berubah
    if is_projection:
        shortcut = tf.keras.layers.Conv2D(
            out_channels, 1, strides=stride, use_bias=False,
            name="{}_proj_conv".format(prefix),
        )(inputs)
        shortcut = tf.keras.layers.BatchNormalization(
            name="{}_proj_bn".format(prefix)
        )(shortcut)
    else:
        shortcut = inputs

    x = tf.keras.layers.Add(name="{}_add".format(prefix))([x, shortcut])
    x = tf.keras.layers.Activation("relu", name="{}_out".format(prefix))(x)
    return x


def _make_resnext_stage(
    x: tf.Tensor,
    planes: int,
    num_blocks: int,
    stride: int,
    cardinality: int,
    base_width: int,
    stage_idx: int,
) -> tf.Tensor:
    out_channels = planes * 4
    in_channels = x.shape[-1]

    # Block pertama bisa melakukan downsampling (stride > 1) atau projection
    needs_projection = (int(in_channels) != out_channels) or (stride != 1)
    x = _resnext_bottleneck(
        x,
        planes=planes,
        stride=stride,
        cardinality=cardinality,
        base_width=base_width,
        is_projection=needs_projection,
        prefix="stage{}_blk0".format(stage_idx),
    )

    for blk_idx in range(1, num_blocks):
        x = _resnext_bottleneck(
            x,
            planes=planes,
            stride=1,
            cardinality=cardinality,
            base_width=base_width,
            is_projection=False,
            prefix="stage{}_blk{}".format(stage_idx, blk_idx),
        )
    return x


def build_resnext50_backbone(
    include_top: bool = False,
    weights: Optional[str] = None,
    input_shape: Optional[Tuple] = None,
) -> tf.keras.Model:
    """Membangun backbone ResNeXt-50-32x4d.

    Args:
        include_top: Harus False. Klasifikasi head ditangani oleh training_common.
        weights: None atau "imagenet". Implementasi ini tidak menyertakan bobot
                 ImageNet pretrained — parameter ini diterima agar kompatibel dengan
                 antarmuka build_model() di training_common, yang akan fallback ke
                 bobot acak secara otomatis jika terjadi error.
        input_shape: Tuple (H, W, C), misalnya (224, 224, 3).

    Returns:
        tf.keras.Model dengan output shape (batch, H/32, W/32, 2048).
    """
    input_shape = _validate_resnext_args(include_top, weights, input_shape, "ResNeXt50")

    if weights == "imagenet":
        # Bobot ImageNet tidak tersedia untuk implementasi kustom ini.
        # training_common.build_model() akan menangkap exception dan memanggil
        # ulang dengan weights=None (fallback ke random init).
        raise ValueError(
            "ResNeXt50 kustom tidak menyertakan bobot ImageNet pretrained. "
            "Akan diinisialisasi dengan bobot acak."
        )

    inputs = tf.keras.Input(shape=input_shape, name="resnext50_input")

    # Stem: 7×7 conv → BN → ReLU → MaxPool (setara dengan ResNet stem)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu", name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # Empat stage dengan konfigurasi ResNeXt-50-32x4d
    # (planes, num_blocks, stride): output_channels = planes * 4
    stage_configs = [
        (64,  3, 1),   # Stage 1 → 256 ch,  56×56 (dengan input 224)
        (128, 4, 2),   # Stage 2 → 512 ch,  28×28
        (256, 6, 2),   # Stage 3 → 1024 ch, 14×14
        (512, 3, 2),   # Stage 4 → 2048 ch,  7×7
    ]

    for stage_idx, (planes, num_blocks, stride) in enumerate(stage_configs, start=1):
        x = _make_resnext_stage(
            x,
            planes=planes,
            num_blocks=num_blocks,
            stride=stride,
            cardinality=_CARDINALITY,
            base_width=_BASE_WIDTH,
            stage_idx=stage_idx,
        )

    return tf.keras.Model(inputs=inputs, outputs=x, name="resnext50_backbone")
