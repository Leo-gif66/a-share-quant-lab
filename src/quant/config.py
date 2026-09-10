from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml


def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(base: str = "configs/base.yaml", override: str | None = None) -> dict:
    with Path(base).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if override:
        with Path(override).open("r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    return cfg
