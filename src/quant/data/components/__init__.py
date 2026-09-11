"""Injectable index-component and sector-metadata adapters."""

from .base import ComponentLoader, InMemoryComponentLoader
from .akshare_provider import AKShareComponentProvider, ComponentUpdateResult
from .provider import ComponentProvider, LocalFirstComponentProvider
from .sector_mapper import SectorMapper

__all__ = [
    "ComponentLoader",
    "ComponentProvider",
    "ComponentUpdateResult",
    "AKShareComponentProvider",
    "InMemoryComponentLoader",
    "LocalFirstComponentProvider",
    "SectorMapper",
]
