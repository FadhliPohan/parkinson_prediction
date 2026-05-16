from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.datasets.registry import DatasetConfig
from src.utils.config import load_augmentations_config


@dataclass
class TrainingAugmentation:
    augmentation_id: str
    label: str
    description: str
    disable_augmentation: bool


class TrainingAugmentationRegistry:
    def __init__(self):
        cfg = load_augmentations_config()
        self._default_augmentation = cfg.get("defaults", {}).get("default_augmentation")
        raw_augmentations = cfg.get("augmentations", {})
        if not isinstance(raw_augmentations, dict) or not raw_augmentations:
            raise ValueError("Konfigurasi augmentations kosong")

        self._augmentations: Dict[str, TrainingAugmentation] = {}
        for augmentation_id, item in raw_augmentations.items():
            self._augmentations[augmentation_id] = TrainingAugmentation(
                augmentation_id=augmentation_id,
                label=str(item.get("label", augmentation_id)),
                description=str(item.get("description", "")),
                disable_augmentation=bool(item.get("disable_augmentation", False)),
            )

        if self._default_augmentation not in self._augmentations:
            self._default_augmentation = next(iter(self._augmentations.keys()))

    @property
    def default_augmentation(self) -> str:
        return str(self._default_augmentation)

    def list_augmentation_ids(self) -> List[str]:
        return list(self._augmentations.keys())

    def get(self, augmentation_id: str) -> TrainingAugmentation:
        if augmentation_id not in self._augmentations:
            available = ", ".join(self.list_augmentation_ids())
            raise KeyError(f"Augmentasi '{augmentation_id}' tidak ditemukan. Tersedia: {available}")
        return self._augmentations[augmentation_id]

    def resolve_ids(self, augmentations_arg: Optional[str]) -> List[str]:
        if augmentations_arg is None or not str(augmentations_arg).strip():
            return [self.default_augmentation]

        normalized = str(augmentations_arg).strip().lower()
        if normalized in {"all", "*"}:
            return self.list_augmentation_ids()

        requested = [item.strip() for item in str(augmentations_arg).split(",") if item.strip()]
        if not requested:
            raise ValueError("Argumen --augmentations kosong")

        available = set(self.list_augmentation_ids())
        unknown = [augmentation_id for augmentation_id in requested if augmentation_id not in available]
        if unknown:
            raise ValueError("Augmentasi tidak ditemukan: {}".format(", ".join(unknown)))

        return requested

    def resolve_from_preprocessing_mode(self, preprocessing_mode: Optional[str]) -> List[str]:
        if preprocessing_mode is None:
            return [self.default_augmentation]

        normalized = str(preprocessing_mode).strip().lower()
        if normalized == "augment":
            return ["augment_on_the_fly"]
        if normalized == "no_augment":
            return ["no_augment"]
        if normalized == "both":
            return ["no_augment", "augment_on_the_fly"]
        raise ValueError("preprocessing_mode tidak valid: {}".format(preprocessing_mode))

    def resolve_for_dataset(
        self,
        dataset_cfg: DatasetConfig,
        selected_augmentation_ids: List[str],
    ) -> List[TrainingAugmentation]:
        dataset_allowed = list(dataset_cfg.augmentation_options or [])
        if not dataset_allowed:
            dataset_allowed = self.list_augmentation_ids()

        resolved: List[TrainingAugmentation] = []
        for augmentation_id in selected_augmentation_ids:
            if augmentation_id not in dataset_allowed:
                continue
            resolved.append(self.get(augmentation_id))

        return resolved
