"""Interfaces and test-friendly implementations for index constituents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..universe_builder import IndexImportSpec


class ComponentLoader(Protocol):
    """Load the current constituent records for a requested index."""

    def load(self, spec: IndexImportSpec) -> Iterable[Mapping[str, Any]]:
        """Return records containing at least ``code`` and ``name``."""


class InMemoryComponentLoader:
    """Injectable mapping-backed loader suitable for tests and offline imports."""

    def __init__(self, components: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
        self._components = {key: list(rows) for key, rows in components.items()}

    def load(self, spec: IndexImportSpec) -> Iterable[Mapping[str, Any]]:
        return self._components.get(spec.key, ())
