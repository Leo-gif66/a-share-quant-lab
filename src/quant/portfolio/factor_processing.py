"""Cross-sectional preparation of factors before portfolio scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..research.preprocessing import ResearchPreprocessor


@dataclass(frozen=True)
class FactorSpec:
    """One factor's weight and expected return direction."""

    weight: float
    direction: int

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("factor weight must not be negative")
        if self.direction not in (-1, 1):
            raise ValueError("factor direction must be either -1 or 1")


FACTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "momentum_5": ("momentum_5", "mom_5"),
    "momentum_20": ("momentum_20", "mom_20"),
    "momentum_60": ("momentum_60", "mom_60"),
    "trend_20": ("trend_20",),
    "trend_60": ("trend_60",),
    "volatility": ("volatility", "volatility_20", "vol_20"),
    "liquidity": ("liquidity", "liquidity_20"),
    "turnover": ("turnover", "turnover_20"),
    "drawdown": ("drawdown", "drawdown_60"),
}


def load_factor_specs(path: str | Path = "configs/factor_weights.yaml") -> dict[str, FactorSpec]:
    """Load factor weights and directions from the v1.0 YAML configuration."""
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping) or not isinstance(config.get("factors"), Mapping):
        raise TypeError("factor configuration requires a factors mapping")

    specs: dict[str, FactorSpec] = {}
    for name, values in config["factors"].items():
        if not isinstance(name, str) or not isinstance(values, Mapping):
            raise TypeError("each configured factor requires a name and mapping")
        try:
            specs[name] = FactorSpec(weight=float(values["weight"]), direction=int(values["direction"]))
        except KeyError as exc:
            raise ValueError(f"factor {name!r} requires weight and direction") from exc
    if not specs:
        raise ValueError("factor configuration must contain at least one factor")
    return specs


def winsorize(
    panel: pd.DataFrame, factor_columns: tuple[str, ...] | list[str], sigma: float = 3.0
) -> pd.DataFrame:
    """Clip each factor to its per-date mean plus or minus ``sigma`` standard deviations."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if "date" not in panel:
        raise ValueError("factor panel requires a date column")
    missing = [column for column in factor_columns if column not in panel]
    if missing:
        raise ValueError(f"factor panel missing columns: {', '.join(missing)}")

    result = panel.copy()
    for column in factor_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result.groupby("date", sort=False)[column].transform(
            lambda values: _clip_sigma(values, sigma)
        )
    return result


def standardize(
    panel: pd.DataFrame, factor_columns: tuple[str, ...] | list[str]
) -> pd.DataFrame:
    """Append a per-date zero-mean, unit-variance ``<factor>_z`` for every factor."""
    if "date" not in panel:
        raise ValueError("factor panel requires a date column")
    missing = [column for column in factor_columns if column not in panel]
    if missing:
        raise ValueError(f"factor panel missing columns: {', '.join(missing)}")

    result = panel.copy()
    for column in factor_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_z"] = values.groupby(result["date"], sort=False).transform(_zscore)
    return result


class FactorProcessor:
    """Resolve factor aliases, winsorize, and standardize a factor panel."""

    def __init__(self, factor_specs: Mapping[str, FactorSpec], sigma: float = 3.0) -> None:
        if not factor_specs:
            raise ValueError("at least one factor specification is required")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.factor_specs = dict(factor_specs)
        self.sigma = sigma
        self.preprocessor = ResearchPreprocessor(sigma=sigma)
        self.last_coverage = pd.DataFrame(columns=ResearchPreprocessor.COVERAGE_COLUMNS)

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/factor_weights.yaml", sigma: float = 3.0):
        return cls(load_factor_specs(path), sigma=sigma)

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(self.factor_specs)

    def align_factor_columns(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Copy configured aliases into stable logical factor-column names."""
        result = panel.copy()
        missing: list[str] = []
        for factor in self.factor_names:
            source = self._source_column(panel, factor)
            if source is None:
                missing.append(factor)
            else:
                result[factor] = pd.to_numeric(panel[source], errors="coerce")
        if missing:
            raise ValueError(f"factor panel missing columns: {', '.join(missing)}")
        return result

    def process(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Validate, clean, winsorize, and z-score factors by rebalance date."""
        if not {"date", "code"}.issubset(panel.columns):
            raise ValueError("factor panel requires date and code columns")
        aligned = self.align_factor_columns(panel)
        cleaned = self.preprocessor.process(aligned, self.factor_names)
        self.last_coverage = cleaned.coverage
        return standardize(cleaned.data, self.factor_names)

    @staticmethod
    def _source_column(panel: pd.DataFrame, factor: str) -> str | None:
        aliases = FACTOR_ALIASES.get(factor, (factor,))
        return next((column for column in aliases if column in panel), None)


def _clip_sigma(values: pd.Series, sigma: float) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return values
    return values.clip(lower=mean - sigma * std, upper=mean + sigma * std)


def _zscore(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return values.where(values.isna(), 0.0)
    return (values - mean) / std
