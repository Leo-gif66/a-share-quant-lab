from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from .base import BaseAlphaModel

_MODELS: dict[str, Callable[..., BaseAlphaModel]] = {}
_DISCOVERED = False


def register(name: str):
    def deco(factory):
        _MODELS[name] = factory
        return factory
    return deco


def autodiscover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module("quant.models")
    for info in pkgutil.iter_modules(package.__path__):
        if info.name not in {"base", "registry", "factory"} and not info.name.startswith("_"):
            importlib.import_module(f"quant.models.{info.name}")
    _DISCOVERED = True


def create(name: str, params: dict | None = None) -> BaseAlphaModel:
    autodiscover()
    if name not in _MODELS:
        raise KeyError(f"unknown model {name}; available={sorted(_MODELS)}")
    return _MODELS[name](params or {})


def names():
    autodiscover()
    return sorted(_MODELS)
