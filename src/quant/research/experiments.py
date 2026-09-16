"""Immutable metadata records for reproducible V5 research experiments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentMetadata:
    """The minimum provenance record required for every V5 experiment."""

    experiment_id: str
    timestamp: str
    git_commit: str
    data_coverage: dict[str, Any]
    universe: dict[str, Any]
    factor_set: list[str]
    model: str
    parameters: dict[str, Any]
    result_metrics: dict[str, float | None]
    notes: str


class ExperimentRegistry:
    """Store experiment metadata as one JSON document per immutable run."""

    def __init__(self, directory: str | Path = "research/experiments") -> None:
        self.directory = Path(directory)

    def record(
        self,
        experiment_id: str,
        *,
        data_coverage: dict[str, Any],
        universe: dict[str, Any],
        factor_set: list[str],
        model: str,
        parameters: dict[str, Any],
        result_metrics: dict[str, float | None],
        notes: str,
        git_commit: str | None = None,
    ) -> Path:
        """Write a new run record without overwriting a prior experiment."""
        normalized_id = _experiment_id(experiment_id)
        target = self.directory / f"{normalized_id}.json"
        if target.exists():
            raise FileExistsError(f"experiment metadata already exists: {target}")
        metadata = ExperimentMetadata(
            experiment_id=normalized_id,
            timestamp=datetime.now(UTC).isoformat(),
            git_commit=git_commit or current_git_commit(),
            data_coverage=dict(data_coverage),
            universe=dict(universe),
            factor_set=list(factor_set),
            model=str(model),
            parameters=dict(parameters),
            result_metrics=dict(result_metrics),
            notes=str(notes),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def current_git_commit() -> str:
    """Return the checked-out commit, or an explicit marker outside a Git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _experiment_id(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    normalized = normalized.strip("-")
    if not normalized:
        raise ValueError("experiment_id must contain letters, digits, hyphens, or underscores")
    return normalized
