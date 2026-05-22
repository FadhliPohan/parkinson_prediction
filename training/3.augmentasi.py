from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.registry import DatasetRegistry
from src.datasets.transforms import print_augmentation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cek kebijakan augmentasi train on-the-fly.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="ID dataset dari registry (opsional).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override folder dataset split yang berisi train/testing/validation.",
    )
    args = parser.parse_args()

    registry = DatasetRegistry()
    dataset_id = args.dataset or registry.default_dataset
    dataset_cfg = registry.get(dataset_id)
    split_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else dataset_cfg.split_path
    print("Dataset ID :", dataset_id)
    print("Split dir  :", split_dir)
    print_augmentation_summary(split_dir)


if __name__ == "__main__":
    main()
