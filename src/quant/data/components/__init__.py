"""Injectable index-component and sector-metadata adapters."""

from .akshare_provider import AKShareComponentProvider, ComponentUpdateResult
from .base import ComponentLoader, InMemoryComponentLoader
from .provider import ComponentProvider, LocalFirstComponentProvider
from .sector_mapper import SectorMapper

__all__ = [
    "AKShareComponentProvider",
    "ComponentLoader",
    "ComponentProvider",
    "ComponentUpdateResult",
    "InMemoryComponentLoader",
    "LocalFirstComponentProvider",
    "SectorMapper",
]
