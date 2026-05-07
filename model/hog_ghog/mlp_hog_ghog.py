import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = CURRENT_DIR.parent
PROJECT_ROOT = MODEL_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import build_feature_arg_parser, run_feature_training_pipeline


def main() -> None:
    parser = build_feature_arg_parser("Training classifier MLP dengan fitur HOG/GHOG untuk klasifikasi Parkinson.")
    args = parser.parse_args()
    run_feature_training_pipeline(classifier_name="mlp", args=args)


if __name__ == "__main__":
    main()
