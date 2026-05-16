from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.utils.config import load_datasets_config
from src.utils.paths import resolve_project_path


@dataclass
class DatasetConfig:
    dataset_id: str
    description: str
    original_dir: str
    split_dir: str
    class_mode: str
    seed: int
    valid_extensions: List[str]
    split: Dict[str, float]
    resize: Dict[str, object]
    augmentation_options: List[str]

    @property
    def original_path(self):
        return resolve_project_path(self.original_dir)

    @property
    def split_path(self):
        return resolve_project_path(self.split_dir)


class DatasetRegistry:
    def __init__(self):
        cfg = load_datasets_config()
        self._default_dataset = cfg.get("defaults", {}).get("default_dataset")
        raw_datasets = cfg.get("datasets", {})
        if not isinstance(raw_datasets, dict) or not raw_datasets:
            raise ValueError("Konfigurasi dataset kosong")

        self._datasets: Dict[str, DatasetConfig] = {}
        for dataset_id, item in raw_datasets.items():
            self._datasets[dataset_id] = DatasetConfig(
                dataset_id=dataset_id,
                description=str(item.get("description", "")),
                original_dir=str(item.get("original_dir", "dataset/original")),
                split_dir=str(item.get("split_dir", f"dataset/split/{dataset_id}")),
                class_mode=str(item.get("class_mode", "recursive_leaf")),
                seed=int(item.get("seed", 42)),
                valid_extensions=list(
                    item.get(
                        "valid_extensions",
                        [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"],
                    )
                ),
                split=dict(item.get("split", {})),
                resize=dict(item.get("resize", {})),
                augmentation_options=list(item.get("augmentation_options", [])),
            )

        if self._default_dataset not in self._datasets:
            self._default_dataset = next(iter(self._datasets.keys()))

    @property
    def default_dataset(self) -> str:
        return str(self._default_dataset)

    def list_dataset_ids(self) -> List[str]:
        # Pertahankan urutan dari file config agar eksekusi "all" bisa deterministik.
        return list(self._datasets.keys())

    def get(self, dataset_id: str) -> DatasetConfig:
        if dataset_id not in self._datasets:
            available = ", ".join(self.list_dataset_ids())
            raise KeyError(f"Dataset '{dataset_id}' tidak ditemukan. Tersedia: {available}")
        return self._datasets[dataset_id]
