"""Immutable factor-weight version snapshots for strategy governance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class StrategyVersion:
    version: str
    path: Path
    timestamp: str
    changed: bool


class StrategyVersionStore:
    """Append immutable strategy snapshots only when factor values change."""

    def __init__(self, directory: str | Path = "configs/strategy_versions") -> None:
        self.directory = Path(directory)

    def record_if_changed(
        self,
        weights: pd.DataFrame | Mapping[str, Any],
        reason: str,
        performance_before: Mapping[str, Any] | None = None,
        performance_after: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> StrategyVersion:
        factors = self._factors(weights)
        latest = self._latest_payload()
        if latest is not None and latest.get("factors") == factors:
            return StrategyVersion(str(latest["version"]), self.directory / f"{latest['version']}.yaml", str(latest["timestamp"]), False)
        self.directory.mkdir(parents=True, exist_ok=True)
        version = f"v{len(self._paths()) + 1:03d}"
        recorded_at = (timestamp or datetime.now(UTC)).isoformat()
        payload = {
            "version": version,
            "timestamp": recorded_at,
            "reason": str(reason),
            "performance_before": dict(performance_before or {}),
            "performance_after": dict(performance_after or {}),
            "factors": factors,
        }
        path = self.directory / f"{version}.yaml"
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        temporary.replace(path)
        return StrategyVersion(version, path, recorded_at, True)

    def history(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for path in self._paths():
            payload = self._read(path)
            factors = payload.get("factors", {})
            rows.append(
                {
                    "version": payload.get("version", path.stem),
                    "timestamp": payload.get("timestamp"),
                    "reason": payload.get("reason", ""),
                    "factor_count": len(factors) if isinstance(factors, Mapping) else 0,
                    "performance_before": payload.get("performance_before", {}),
                    "performance_after": payload.get("performance_after", {}),
                    "path": str(path),
                }
            )
        return pd.DataFrame(rows, columns=("version", "timestamp", "reason", "factor_count", "performance_before", "performance_after", "path"))

    def _factors(self, weights: pd.DataFrame | Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
        if isinstance(weights, pd.DataFrame):
            required = {"factor", "weight", "direction"}
            if not required.issubset(weights.columns):
                raise ValueError("weight frame requires factor, weight, and direction")
            source = {
                str(row.factor): {"weight": float(row.weight), "direction": int(row.direction)}
                for row in weights.itertuples(index=False)
            }
        else:
            container = weights.get("factors", weights)
            if not isinstance(container, Mapping):
                raise TypeError("weights must be a factor mapping or configuration mapping")
            source = {
                str(name): {
                    "weight": float(values["weight"]),
                    "direction": int(values["direction"]),
                }
                for name, values in container.items()
                if isinstance(values, Mapping)
            }
        if not source:
            raise ValueError("at least one factor is required for a strategy version")
        values = np.asarray([item["weight"] for item in source.values()], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1.0, atol=1e-8):
            raise ValueError("factor weights must be finite, non-negative, and sum to one")
        if any(item["direction"] not in (-1, 1) for item in source.values()):
            raise ValueError("factor direction must be -1 or 1")
        return {name: source[name] for name in sorted(source)}

    def _paths(self) -> list[Path]:
        return sorted(self.directory.glob("v[0-9][0-9][0-9].yaml")) if self.directory.exists() else []

    def _latest_payload(self) -> dict[str, Any] | None:
        paths = self._paths()
        return self._read(paths[-1]) if paths else None

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"strategy version {path} must contain a mapping")
        return payload
