"""Industry classification adapters used by the data and universe layers."""

from .provider import (
    AKShareIndustryProvider,
    IndustryProvider,
    IndustryUpdateResult,
    load_industry_metadata,
)

__all__ = [
    "AKShareIndustryProvider",
    "IndustryProvider",
    "IndustryUpdateResult",
    "load_industry_metadata",
]
