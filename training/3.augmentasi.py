from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.transforms import print_augmentation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cek kebijakan augmentasi train on-the-fly.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/split/parkinson_multiclass",
        help="Folder dataset split yang berisi train/testing/validation.",
    )
    args = parser.parse_args()

    split_dir = Path(args.dataset_dir).resolve()
    print_augmentation_summary(split_dir)


if __name__ == "__main__":
    main()
