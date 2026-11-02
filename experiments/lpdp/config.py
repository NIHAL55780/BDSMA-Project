"""YAML config loading with `inherit:` support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "inherit":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(name: str | Path) -> dict[str, Any]:
    """Load `experiments/configs/{name}.yaml` (or an absolute path)."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{path.name}.yaml"
    elif not path.is_absolute():
        candidate = CONFIG_DIR / path.name
        path = candidate if candidate.exists() else (REPO_ROOT / path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inherit = data.get("inherit")
    if inherit:
        parent = load_config(inherit)
        data = _deep_merge(parent, data)
    data["_config_path"] = str(path)
    data["_repo_root"] = str(REPO_ROOT)
    return data


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    rel = Path(cfg["paths"][key])
    root = Path(cfg["_repo_root"])
    return rel if rel.is_absolute() else root / rel


def rng_seed(cfg: dict[str, Any], extra: int = 0) -> int:
    return int(cfg.get("seed", 0)) + extra
