from typing import Dict, Optional, Tuple

import tensorflow as tf


def transformer_preprocess_input(inputs: tf.Tensor) -> tf.Tensor:
    # Gunakan layer Keras agar kompatibel dengan KerasTensor pada Functional API.
    return tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)


def _validate_builder_args(
    include_top: bool,
    weights: Optional[str],
    input_shape: Optional[Tuple[int, int, int]],
    backbone_name: str,
) -> Tuple[int, int, int]:
    if include_top:
        raise ValueError("{} backbone hanya mendukung include_top=False.".format(backbone_name))
    if input_shape is None or len(input_shape) != 3:
        raise ValueError("input_shape untuk {} tidak valid: {}".format(backbone_name, input_shape))
    if input_shape[0] is None or input_shape[1] is None or input_shape[2] is None:
        raise ValueError("input_shape untuk {} harus statis (H, W, C).".format(backbone_name))
    if weights not in (None, "imagenet"):
        raise ValueError(
            "Argumen weights={} tidak didukung untuk {}.".format(weights, backbone_name)
        )
    return (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))


@tf.keras.utils.register_keras_serializable(package="parkinson")
class LearnablePositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.embed_dim = int(embed_dim)

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.sequence_length, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "embed_dim": self.embed_dim,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="parkinson")
class AddClassDistillationTokens(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name="class_token",
            shape=(1, 1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.distillation_token = self.add_weight(
            name="distillation_token",
            shape=(1, 1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls = tf.broadcast_to(self.class_token, [batch_size, 1, self.embed_dim])
        dist = tf.broadcast_to(self.distillation_token, [batch_size, 1, self.embed_dim])
        return tf.concat([cls, dist, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="parkinson")
class Roll2D(tf.keras.layers.Layer):
    def __init__(self, shift_height: int, shift_width: int, **kwargs):
        super().__init__(**kwargs)
        self.shift_height = int(shift_height)
        self.shift_width = int(shift_width)

    def call(self, inputs):
        return tf.roll(inputs, shift=[self.shift_height, self.shift_width], axis=[1, 2])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shift_height": self.shift_height,
                "shift_width": self.shift_width,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="parkinson")
class DropLeadingTokens(tf.keras.layers.Layer):
    def __init__(self, num_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = int(num_tokens)

    def call(self, inputs):
        return inputs[:, self.num_tokens :, :]

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens})
        return config


@tf.keras.utils.register_keras_serializable(package="parkinson")
class WindowSelfAttention2D(tf.keras.layers.Layer):
    def __init__(self, dim: int, num_heads: int, window_size: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.dropout_rate = float(dropout_rate)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1, self.dim // self.num_heads),
            dropout=self.dropout_rate,
            name="window_mha",
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        token_shape = tf.TensorShape([None, self.window_size * self.window_size, self.dim])
        self.attn.build(token_shape, token_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        window_size = tf.cast(self.window_size, tf.int32)

        pad_h = (window_size - tf.math.mod(height, window_size)) % window_size
        pad_w = (window_size - tf.math.mod(width, window_size)) % window_size
        x = tf.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        padded_shape = tf.shape(x)
        padded_h = padded_shape[1]
        padded_w = padded_shape[2]

        x = tf.reshape(
            x,
            [
                batch_size,
                padded_h // window_size,
                window_size,
                padded_w // window_size,
                window_size,
                channels,
            ],
        )
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        windows = tf.reshape(x, [-1, window_size * window_size, channels])

        attended = self.attn(windows, windows, training=training)
        attended = self.dropout(attended, training=training)

        attended = tf.reshape(
            attended,
            [
                batch_size,
                padded_h // window_size,
                padded_w // window_size,
                window_size,
                window_size,
                channels,
            ],
        )
        attended = tf.transpose(attended, [0, 1, 3, 2, 4, 5])
        attended = tf.reshape(attended, [batch_size, padded_h, padded_w, channels])
        return attended[:, :height, :width, :]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


def _mlp_block(inputs, hidden_dim: int, dropout_rate: float, prefix: str):
    output_dim = inputs.shape[-1]
    if output_dim is None:
        raise ValueError("Dimensi channel MLP tidak diketahui pada {}.".format(prefix))
    x = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu, name="{}_dense1".format(prefix))(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name="{}_drop1".format(prefix))(x)
    x = tf.keras.layers.Dense(int(output_dim), name="{}_dense2".format(prefix))(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="{}_drop2".format(prefix))(x)
    return x


def _transformer_encoder(inputs, num_heads: int, mlp_ratio: float, dropout_rate: float, prefix: str):
    embed_dim = inputs.shape[-1]
    if embed_dim is None:
        raise ValueError("Dimensi embedding tidak diketahui pada {}.".format(prefix))
    embed_dim = int(embed_dim)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="{}_ln1".format(prefix))(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=max(1, embed_dim // num_heads),
        dropout=dropout_rate,
        name="{}_mha".format(prefix),
    )(x, x)
    x = tf.keras.layers.Dropout(dropout_rate, name="{}_attn_drop".format(prefix))(x)
    x = tf.keras.layers.Add(name="{}_attn_residual".format(prefix))([inputs, x])

    y = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="{}_ln2".format(prefix))(x)
    y = _mlp_block(y, hidden_dim=int(embed_dim * mlp_ratio), dropout_rate=dropout_rate, prefix="{}_mlp".format(prefix))
    return tf.keras.layers.Add(name="{}_mlp_residual".format(prefix))([x, y])


def _swin_block(
    inputs,
    num_heads: int,
    window_size: int,
    shift_size: int,
    mlp_ratio: float,
    dropout_rate: float,
    prefix: str,
):
    channels = inputs.shape[-1]
    if channels is None:
        raise ValueError("Dimensi channel Swin block tidak diketahui pada {}.".format(prefix))
    channels = int(channels)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="{}_ln1".format(prefix))(inputs)

    if shift_size > 0:
        x = Roll2D(
            shift_height=-shift_size,
            shift_width=-shift_size,
            name="{}_shift".format(prefix),
        )(x)

    x = WindowSelfAttention2D(
        dim=channels,
        num_heads=num_heads,
        window_size=window_size,
        dropout_rate=dropout_rate,
        name="{}_window_attn".format(prefix),
    )(x)

    if shift_size > 0:
        x = Roll2D(
            shift_height=shift_size,
            shift_width=shift_size,
            name="{}_unshift".format(prefix),
        )(x)

    x = tf.keras.layers.Add(name="{}_attn_residual".format(prefix))([inputs, x])

    y = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="{}_ln2".format(prefix))(x)
    y = _mlp_block(
        y,
        hidden_dim=int(channels * mlp_ratio),
        dropout_rate=dropout_rate,
        prefix="{}_mlp".format(prefix),
    )
    return tf.keras.layers.Add(name="{}_mlp_residual".format(prefix))([x, y])


def build_vit_backbone(include_top=False, weights=None, input_shape=None):
    input_shape = _validate_builder_args(include_top, weights, input_shape, "ViT")
    patch_size = 16
    embed_dim = 192
    depth = 8
    num_heads = 6
    mlp_ratio = 4.0
    dropout_rate = 0.1

    patch_h = input_shape[0] // patch_size
    patch_w = input_shape[1] // patch_size
    if patch_h < 1 or patch_w < 1:
        raise ValueError("Input terlalu kecil untuk patch ViT {}x{}.".format(patch_size, patch_size))
    num_patches = patch_h * patch_w

    inputs = tf.keras.Input(shape=input_shape, name="vit_input")
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="vit_patch_embed",
    )(inputs)
    x = tf.keras.layers.Reshape((num_patches, embed_dim), name="vit_patch_flatten")(x)
    x = LearnablePositionEmbedding(num_patches, embed_dim, name="vit_position_embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="vit_pos_drop")(x)

    for block_index in range(depth):
        x = _transformer_encoder(
            x,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            prefix="vit_block_{}".format(block_index),
        )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="vit_output_ln")(x)
    outputs = tf.keras.layers.Reshape((patch_h, patch_w, embed_dim), name="vit_feature_map")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vit_backbone")


def build_deit_backbone(include_top=False, weights=None, input_shape=None):
    input_shape = _validate_builder_args(include_top, weights, input_shape, "DeiT")
    patch_size = 16
    embed_dim = 192
    depth = 8
    num_heads = 6
    mlp_ratio = 4.0
    dropout_rate = 0.1

    patch_h = input_shape[0] // patch_size
    patch_w = input_shape[1] // patch_size
    if patch_h < 1 or patch_w < 1:
        raise ValueError("Input terlalu kecil untuk patch DeiT {}x{}.".format(patch_size, patch_size))
    num_patches = patch_h * patch_w

    inputs = tf.keras.Input(shape=input_shape, name="deit_input")
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="deit_patch_embed",
    )(inputs)
    x = tf.keras.layers.Reshape((num_patches, embed_dim), name="deit_patch_flatten")(x)
    x = AddClassDistillationTokens(embed_dim=embed_dim, name="deit_special_tokens")(x)
    x = LearnablePositionEmbedding(num_patches + 2, embed_dim, name="deit_position_embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="deit_pos_drop")(x)

    for block_index in range(depth):
        x = _transformer_encoder(
            x,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            prefix="deit_block_{}".format(block_index),
        )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="deit_output_ln")(x)
    x = DropLeadingTokens(num_tokens=2, name="deit_drop_special_tokens")(x)
    outputs = tf.keras.layers.Reshape((patch_h, patch_w, embed_dim), name="deit_feature_map")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="deit_backbone")


def build_swin_transformer_backbone(include_top=False, weights=None, input_shape=None):
    input_shape = _validate_builder_args(include_top, weights, input_shape, "SwinTransformer")
    patch_size = 4
    embed_dim = 96
    window_size = 7
    stage_depths = [2, 2, 4]
    stage_heads = [3, 6, 12]
    mlp_ratio = 4.0
    dropout_rate = 0.1

    inputs = tf.keras.Input(shape=input_shape, name="swin_input")
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name="swin_patch_embed",
    )(inputs)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="swin_patch_ln")(x)

    current_dim = embed_dim
    for stage_index, (depth, heads) in enumerate(zip(stage_depths, stage_heads)):
        for block_index in range(depth):
            shift_size = 0 if block_index % 2 == 0 else window_size // 2
            x = _swin_block(
                x,
                num_heads=heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                prefix="swin_stage{}_block{}".format(stage_index, block_index),
            )

        if stage_index < len(stage_depths) - 1:
            current_dim *= 2
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="swin_stage{}_merge_ln".format(stage_index))(x)
            x = tf.keras.layers.Conv2D(
                filters=current_dim,
                kernel_size=2,
                strides=2,
                padding="same",
                name="swin_stage{}_patch_merge".format(stage_index),
            )(x)

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="swin_output_ln")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="swin_transformer_backbone")


TRANSFORMER_CUSTOM_OBJECTS: Dict[str, object] = {
    "LearnablePositionEmbedding": LearnablePositionEmbedding,
    "AddClassDistillationTokens": AddClassDistillationTokens,
    "Roll2D": Roll2D,
    "DropLeadingTokens": DropLeadingTokens,
    "WindowSelfAttention2D": WindowSelfAttention2D,
}
