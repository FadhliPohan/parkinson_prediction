import tensorflow as tf

from training_common import build_common_arg_parser, run_training_pipeline


def main() -> None:
    parser = build_common_arg_parser("Training klasifikasi Parkinson dengan MobileNetV2.")
    args = parser.parse_args()

    run_training_pipeline(
        model_name="mobilenetv2",
        backbone_builder=tf.keras.applications.MobileNetV2,
        preprocess_fn=tf.keras.applications.mobilenet_v2.preprocess_input,
        args=args,
    )


if __name__ == "__main__":
    main()
