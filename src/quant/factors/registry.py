from __future__ import annotations
import importlib
import pkgutil
from .base import Factor

_FACTORS: dict[str, Factor] = {}
_DISCOVERED = False


def register(factor: Factor) -> Factor:
    if factor.name in _FACTORS:
        raise ValueError(f"duplicate factor: {factor.name}")
    _FACTORS[factor.name] = factor
    return factor


def autodiscover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module("quant.factors")
    for info in pkgutil.iter_modules(package.__path__):
        if info.name not in {"base", "registry"} and not info.name.startswith("_"):
            importlib.import_module(f"quant.factors.{info.name}")
    _DISCOVERED = True


def get(name: str) -> Factor:
    autodiscover()
    if name not in _FACTORS:
        raise KeyError(f"unknown factor {name}; available={sorted(_FACTORS)}")
    return _FACTORS[name]


def names() -> list[str]:
    autodiscover()
    return sorted(_FACTORS)
