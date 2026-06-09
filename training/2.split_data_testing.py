from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.registry import DatasetRegistry
from src.datasets.splitter import (
    SPLIT_PRESETS,
    normalize_preset,
    resolve_preset_list,
    resolve_split_cfg,
    split_dataset,
    split_dir_for_preset,
    validate_balanced_split_manifest,
)
from src.datasets.validator import list_image_files


ON_EXISTING_SPLIT_CHOICES = ("ask", "resplit", "skip")


def _split_manifest_path(split_dir: Path) -> Path:
    return split_dir / "_metadata" / "split_manifest.json"


def _load_split_manifest(split_dir: Path):
    path = _split_manifest_path(split_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None


def _split_dir_has_required_structure(split_dir: Path, extensions):
    for split_name in ["train", "testing", "validation"]:
        split_path = split_dir / split_name
        if not split_path.exists() or not split_path.is_dir():
            return False
        class_dirs = [path for path in split_path.iterdir() if path.is_dir()]
        if not class_dirs:
            return False
        has_images = any(list_image_files(class_dir, extensions) for class_dir in class_dirs)
        if not has_images:
            return False
    return True


def _ask_resplit(dataset_id: str, split_dir: Path) -> bool:
    answer = input(
        "\nFolder split untuk dataset '{}' sudah ada di:\n{}\nPerlu split ulang? [y/N]: ".format(
            dataset_id, split_dir
        )
    ).strip().lower()
    return answer in {"y", "yes"}


def _assert_split_balance(manifest, source_label: str) -> None:
    validation = manifest.get("balance_validation") if isinstance(manifest, dict) else None
    if not isinstance(validation, dict):
        validation = validate_balanced_split_manifest(manifest)
    if not validation.get("is_balanced", False):
        raise RuntimeError(
            "Split {} tidak seimbang antar kelas. Jalankan split ulang dengan konfigurasi dataset yang benar.".format(
                source_label
            )
        )
    print(
        "[VALIDASI] Split {} valid: train seimbang, test==validation, "
        "tanpa gambar sintetis di test/validation (anti-leakage).".format(source_label)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split dataset dinamis berbasis registry/config.")
    parser.add_argument("--dataset", type=str, default=None, help="ID dataset dari registry (config + auto folder)")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Override folder dataset original")
    parser.add_argument("--split-dir", type=str, default=None, help="Override folder output split")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--class-mode",
        type=str,
        default=None,
        choices=["direct", "recursive_leaf"],
        help="Override mode deteksi kelas",
    )
    parser.add_argument(
        "--on-existing-split",
        type=str,
        default="ask",
        choices=list(ON_EXISTING_SPLIT_CHOICES),
        help="Perilaku jika folder split sudah ada: ask, resplit, atau skip.",
    )
    parser.add_argument(
        "--split-preset",
        type=str,
        default=None,
        choices=sorted(SPLIT_PRESETS.keys()),
        help="Preset rasio split tunggal (mis. 80-10-10 atau 70-15-15).",
    )
    parser.add_argument(
        "--split-presets",
        type=str,
        default=None,
        help=(
            "Preset split (boleh banyak): 'both' (80-10-10 & 70-15-15), 'config' (rasio default config), "
            "satu preset, atau daftar dipisah koma. Tiap preset disimpan di folder terpisah."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=None, help="Override rasio train (0-1).")
    parser.add_argument("--test-ratio", type=float, default=None, help="Override rasio testing (0-1).")
    parser.add_argument("--val-ratio", type=float, default=None, help="Override rasio validation (0-1).")
    return parser


def _resolve_presets(args: argparse.Namespace):
    """Tentukan daftar preset yang akan diproses.

    Prioritas: --split-presets (multi/both) > --split-preset (tunggal) >
    rasio manual / config (None = config-default).
    """
    if args.split_presets is not None and str(args.split_presets).strip():
        return resolve_preset_list(args.split_presets)
    if args.split_preset:
        return [normalize_preset(args.split_preset)]
    return [None]


def _process_one_preset(
    dataset_id: str,
    dataset_cfg,
    preset,
    original_dir: Path,
    base_split_dir: Path,
    seed: int,
    class_mode: str,
    args: argparse.Namespace,
    explicit_split_dir: bool,
) -> None:
    split_dir = base_split_dir if explicit_split_dir else split_dir_for_preset(base_split_dir, preset)

    # Rasio split untuk preset ini (preset eksplisit -> rasio preset;
    # None -> rasio manual/config).
    split_cfg = resolve_split_cfg(
        preset=preset,
        train=args.train_ratio if preset is None else None,
        testing=args.test_ratio if preset is None else None,
        validation=args.val_ratio if preset is None else None,
        fallback=dataset_cfg.split,
    )
    preset_label = normalize_preset(preset) or "config-default"

    print("\n" + "=" * 60)
    print("=== Split Dataset (preset: {}) ===".format(preset_label))
    print("Dataset ID   :", dataset_id)
    print("Original dir :", original_dir)
    print("Split dir    :", split_dir)
    print("Class mode   :", class_mode)
    print("Seed         :", seed)
    print(
        "Rasio split  : train={train:.2f} | testing={testing:.2f} | validation={validation:.2f}".format(
            **split_cfg
        )
    )

    if split_dir.exists():
        if args.on_existing_split == "resplit":
            should_resplit = True
        elif args.on_existing_split == "skip":
            should_resplit = False
        elif not sys.stdin.isatty():
            print(
                "[INFO] Split folder sudah ada, sesi non-interaktif tidak bisa bertanya. "
                "Gunakan split existing. Pakai `--on-existing-split resplit` untuk memaksa split ulang."
            )
            should_resplit = False
        else:
            should_resplit = _ask_resplit(dataset_id=dataset_id, split_dir=split_dir)
    else:
        should_resplit = True

    if should_resplit:
        manifest = split_dataset(
            original_dir=original_dir,
            split_dir=split_dir,
            class_mode=class_mode,
            extensions=dataset_cfg.valid_extensions,
            split_cfg=split_cfg,
            resize_cfg=dataset_cfg.resize,
            seed=seed,
            split_preset=preset,
        )
        source_label = "baru"
    else:
        if not _split_dir_has_required_structure(split_dir, dataset_cfg.valid_extensions):
            raise RuntimeError(
                "Split existing tidak valid/kurang lengkap. Jalankan ulang dengan `--on-existing-split resplit`."
            )
        manifest = _load_split_manifest(split_dir)
        if manifest is None:
            raise RuntimeError(
                "split_manifest.json tidak ditemukan pada split existing. "
                "Jalankan ulang dengan `--on-existing-split resplit`."
            )
        source_label = "existing"

    split_stats = manifest.get("split_stats", {})
    print("\n=== Ringkasan Split (preset: {}) ===".format(manifest.get("split_preset", preset_label)))
    for split_name in ["train", "testing", "validation"]:
        per_class = split_stats.get(split_name, {})
        total = sum(int(v) for v in per_class.values())
        print(f"- {split_name:10s}: {total} gambar | {len(per_class)} kelas")
    balancing_info = manifest.get("class_balancing", {})
    if balancing_info:
        print("\n=== Ringkasan Balancing Kelas ===")
        if balancing_info.get("applied"):
            print("Status      : Aktif (dataset awal tidak seimbang)")
            print("Strategi    :", balancing_info.get("strategy", "-"))
            print("Target/kelas:", balancing_info.get("target_per_class", "-"))
            print("Total tambah:", balancing_info.get("total_generated", 0))
            generated_per_class = balancing_info.get("generated_per_class", {})
            for class_name in sorted(generated_per_class.keys()):
                generated = int(generated_per_class[class_name])
                if generated > 0:
                    print(f"- {class_name}: +{generated} gambar rotasi")
        else:
            print("Status      : Tidak perlu (dataset sudah seimbang)")
    _assert_split_balance(manifest, source_label=source_label)
    print("Split preset '{}' selesai.".format(preset_label))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = DatasetRegistry()
    dataset_id = args.dataset or registry.default_dataset
    dataset_cfg = registry.get(dataset_id)

    original_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else dataset_cfg.original_path
    explicit_split_dir = bool(args.split_dir)
    base_split_dir = Path(args.split_dir).resolve() if explicit_split_dir else dataset_cfg.split_path
    seed = int(args.seed) if args.seed is not None else int(dataset_cfg.seed)
    class_mode = args.class_mode or dataset_cfg.class_mode

    presets = _resolve_presets(args)
    if explicit_split_dir and len(presets) > 1:
        raise SystemExit(
            "--split-dir tidak bisa dipakai bersamaan dengan beberapa preset (mis. 'both'). "
            "Pilih satu preset atau lepas --split-dir."
        )

    preset_labels = [normalize_preset(p) or "config-default" for p in presets]
    print("\n=== Rencana Split ===")
    print("Dataset ID :", dataset_id)
    print("Preset     :", ", ".join(preset_labels))

    for preset in presets:
        _process_one_preset(
            dataset_id=dataset_id,
            dataset_cfg=dataset_cfg,
            preset=preset,
            original_dir=original_dir,
            base_split_dir=base_split_dir,
            seed=seed,
            class_mode=class_mode,
            args=args,
            explicit_split_dir=explicit_split_dir,
        )

    print("\nSemua proses split selesai untuk preset:", ", ".join(preset_labels))


if __name__ == "__main__":
    main()
