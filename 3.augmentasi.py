from __future__ import annotations

import argparse
from pathlib import Path

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
