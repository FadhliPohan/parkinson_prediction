from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from .paths import PROJECT_ROOT

VENV_DIR = PROJECT_ROOT / ".venv"


def build_runtime_env() -> dict:
    env = os.environ.copy()
    lib_dirs: List[str] = []

    nvidia_root_candidates = sorted((VENV_DIR / "lib").glob("python*/site-packages/nvidia"))
    for nvidia_root in nvidia_root_candidates:
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))

    if lib_dirs:
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        prefix = ":".join(lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{prefix}:{current_ld_path}" if current_ld_path else prefix

    # Pastikan subprocess dari subfolder (mis. training/, model/legacy_or_wrappers/)
    # tetap bisa mengimpor paket internal `src`.
    project_root = str(PROJECT_ROOT)
    current_pythonpath = env.get("PYTHONPATH", "")
    path_parts = [part for part in current_pythonpath.split(":") if part]
    if project_root not in path_parts:
        path_parts.insert(0, project_root)
    env["PYTHONPATH"] = ":".join(path_parts)

    return env


def get_venv_python_path() -> Path:
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_runtime_python() -> Path:
    venv_python = get_venv_python_path()
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)
