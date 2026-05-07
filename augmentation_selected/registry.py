import importlib.util
from pathlib import Path
from typing import Dict, List


PROFILE_DIR = Path(__file__).resolve().parent
PROFILE_FILES = {
    "without_augment": PROFILE_DIR / "without_augment.py",
    "mycostum_augment": PROFILE_DIR / "mycostum_augment.py",
}
PROFILE_ALIASES = {
    "none": "without_augment",
    "noaug": "without_augment",
    "without": "without_augment",
    "without_augment": "without_augment",
    "custom": "mycostum_augment",
    "augment": "mycostum_augment",
    "aug": "mycostum_augment",
    "mycostum_augment": "mycostum_augment",
    "all": "all",
}


def canonicalize_profile_name(profile_name: str) -> str:
    normalized = str(profile_name).strip().lower()
    if normalized not in PROFILE_ALIASES:
        raise KeyError(
            "Profile augmentasi '{}' tidak dikenal. Pilihan: {}".format(
                profile_name,
                ", ".join(sorted(PROFILE_FILES.keys()) + ["all"]),
            )
        )
    return PROFILE_ALIASES[normalized]


def resolve_profile_names(profile_name: str) -> List[str]:
    canonical_name = canonicalize_profile_name(profile_name)
    if canonical_name == "all":
        return ["without_augment", "mycostum_augment"]
    return [canonical_name]


def _load_module(module_key: str):
    module_path = PROFILE_FILES.get(module_key)
    if module_path is None or not module_path.exists():
        raise FileNotFoundError("File profile augmentasi tidak ditemukan: {}".format(module_path))

    spec = importlib.util.spec_from_file_location(
        "parkinson_augmentation_{}".format(module_key),
        str(module_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Gagal memuat profile augmentasi: {}".format(module_path))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_profile(profile_name: str) -> Dict[str, object]:
    canonical_name = canonicalize_profile_name(profile_name)
    if canonical_name == "all":
        raise ValueError("Gunakan resolve_profile_names() untuk mode 'all'.")

    module = _load_module(canonical_name)
    build_fn = getattr(module, "build_training_augmentation", None)
    describe_fn = getattr(module, "describe_augmentation_policy", None)

    profile = {
        "name": str(getattr(module, "PROFILE_NAME", canonical_name)),
        "display_name": str(getattr(module, "DISPLAY_NAME", canonical_name)),
        "run_tag": str(getattr(module, "RUN_TAG", canonical_name)),
        "enabled": bool(getattr(module, "AUGMENTATION_ENABLED", False)),
        "build_fn": build_fn if callable(build_fn) else None,
        "policy_lines": [str(line) for line in describe_fn()] if callable(describe_fn) else [],
        "source_path": PROFILE_FILES[canonical_name],
    }
    return profile


def list_available_profiles() -> List[Dict[str, object]]:
    profiles = []
    for profile_name in PROFILE_FILES.keys():
        profiles.append(load_profile(profile_name))
    return profiles


def get_profile_choices() -> List[str]:
    return ["without_augment", "mycostum_augment", "all"]
