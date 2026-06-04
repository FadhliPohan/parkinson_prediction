import os
import sys
from pathlib import Path


def configure_cuda_library_path() -> None:
    if os.environ.get("PARKINSON_CUDA_ENV_READY") == "1":
        return

    project_root = Path(__file__).resolve().parents[2]
    lib_dirs = []
    for nvidia_root in sorted((project_root / ".venv" / "lib").glob("python*/site-packages/nvidia")):
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))

    if not lib_dirs:
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(lib_dirs)
    merged_ld_path = "{}:{}".format(prefix, current_ld_path) if current_ld_path else prefix
    if merged_ld_path == current_ld_path:
        return

    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = merged_ld_path
    new_env["PARKINSON_CUDA_ENV_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)


configure_cuda_library_path()
import tensorflow as tf

from training_common import build_common_arg_parser, run_training_pipeline


def main() -> None:
    parser = build_common_arg_parser("Training klasifikasi Parkinson dengan VGG16.")
    args = parser.parse_args()

    run_training_pipeline(
        model_name="vgg16",
        backbone_builder=tf.keras.applications.VGG16,
        preprocess_fn=tf.keras.applications.vgg16.preprocess_input,
        args=args,
    )


if __name__ == "__main__":
    main()
