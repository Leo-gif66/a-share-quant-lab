"""Explain point-in-time coverage bottlenecks before changing validation windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

import pandas as pd

from .walk_forward import WalkForwardSettings


@dataclass(frozen=True)
class CoverageAuditResult:
    raw_date_range: dict[str, str | None]
    feature_date_range: dict[str, str | None]
    labelled_date_range: dict[str, str | None]
    benchmark_range: dict[str, str | None]
    eligible_signal_dates: int
    training_periods: int
    test_periods: int
    rebalances: int
    rows_removed: pd.DataFrame
    bottleneck: str

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)]).drop(columns="rows_removed")


class ValidationCoverageAuditor:
    """Audit the inputs used by the V4 walk-forward simulator without replaying it."""

    def __init__(
        self,
        *,
        raw_dir: str | Path = "data/raw",
        features_dir: str | Path = "data/features",
        scores_path: str | Path = "data/features/composite_score.parquet",
        benchmark_path: str | Path = "data/raw/000300.parquet",
        settings: WalkForwardSettings | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.features_dir = Path(features_dir)
        self.scores_path = Path(scores_path)
        self.benchmark_path = Path(benchmark_path)
        self.settings = settings or WalkForwardSettings()

    def audit(self) -> CoverageAuditResult:
        """Return date coverage and every material filtering stage for V4 inputs."""
        if not self.scores_path.exists():
            raise FileNotFoundError(f"score history not found: {self.scores_path}")
        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"benchmark history not found: {self.benchmark_path}")
        scores = self._scores(pd.read_parquet(self.scores_path))
        benchmark = self._dates(pd.read_parquet(self.benchmark_path), "benchmark")
        raw = self._directory_dates(self.raw_dir)
        features = self._directory_dates(self.features_dir)
        raw_dates = pd.DatetimeIndex(raw["date"].drop_duplicates().sort_values())
        feature_dates = pd.DatetimeIndex(features["date"].drop_duplicates().sort_values())
        score_dates = pd.DatetimeIndex(scores["date"].drop_duplicates().sort_values())
        benchmark_dates = pd.DatetimeIndex(benchmark["date"].drop_duplicates().sort_values())
        price_counts = raw.groupby("date")["code"].nunique() if not raw.empty else pd.Series(dtype="int64")
        score_counts = scores.groupby("date")["code"].nunique()
        common = score_dates.intersection(raw_dates).intersection(benchmark_dates)
        eligible = pd.DatetimeIndex(
            date
            for date in common
            if score_counts.loc[date] >= self.settings.top_n and price_counts.loc[date] >= self.settings.top_n
        )
        events = list(
            range(self.settings.train_window, max(self.settings.train_window, len(eligible) - 1), self.settings.rebalance_frequency)
        )
        rebalances = len(events)
        stages = pd.DataFrame(
            [
                {"stage": "raw_price_rows", "input_rows": len(raw), "output_rows": len(raw), "removed_rows": 0, "note": "local stock price records"},
                {"stage": "feature_rows", "input_rows": len(features), "output_rows": len(features), "removed_rows": 0, "note": "per-stock feature records"},
                {"stage": "score_rows", "input_rows": len(scores), "output_rows": len(scores), "removed_rows": 0, "note": "legacy composite score records"},
                {"stage": "common_dates", "input_rows": len(score_dates), "output_rows": len(common), "removed_rows": len(score_dates) - len(common), "note": "score, price, and CSI300 intersection"},
                {"stage": "eligible_signal_dates", "input_rows": len(common), "output_rows": len(eligible), "removed_rows": len(common) - len(eligible), "note": f"at least {self.settings.top_n} scores and prices"},
                {"stage": "walk_forward_events", "input_rows": len(eligible), "output_rows": rebalances, "removed_rows": len(eligible) - rebalances, "note": f"{self.settings.train_window}-session train window and {self.settings.rebalance_frequency}-session rebalance"},
            ]
        )
        bottleneck = self._bottleneck(raw_dates, feature_dates, score_dates, eligible)
        return CoverageAuditResult(
            raw_date_range=_date_range(raw_dates),
            feature_date_range=_date_range(feature_dates),
            labelled_date_range=_date_range(score_dates),
            benchmark_range=_date_range(benchmark_dates),
            eligible_signal_dates=len(eligible),
            training_periods=rebalances,
            test_periods=rebalances,
            rebalances=rebalances,
            rows_removed=stages,
            bottleneck=bottleneck,
        )

    def write_report(
        self, result: CoverageAuditResult, output_path: str | Path = "reports/validation_coverage.html"
    ) -> Path:
        """Write a self-contained, reviewable coverage report."""
        summary = result.summary().to_html(index=False, escape=True, border=0)
        removed = result.rows_removed.to_html(index=False, escape=True, border=0)
        document = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Validation Coverage Audit</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
</head><body><h1>Validation Coverage Audit</h1><p>{escape(result.bottleneck)}</p><h2>Summary</h2>{summary}<h2>Filtering stages</h2>{removed}</body></html>"""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8")
        return target

    @staticmethod
    def _scores(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "code", "composite_score"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"score history missing: {', '.join(sorted(missing))}")
        result = frame.loc[:, ["date", "code", "composite_score"]].copy()
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
        result["code"] = result["code"].astype(str).str.zfill(6)
        return result.dropna(subset=["composite_score"]).drop_duplicates(["date", "code"], keep="last")

    @staticmethod
    def _dates(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        if "date" not in frame:
            raise ValueError(f"{name} history requires date")
        result = frame.loc[:, ["date"]].copy()
        result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
        return result.dropna().drop_duplicates()

    def _directory_dates(self, directory: Path) -> pd.DataFrame:
        records: list[pd.DataFrame] = []
        for path in sorted(directory.glob("*.parquet")):
            if not path.stem.isdigit() or len(path.stem) != 6:
                continue
            frame = pd.read_parquet(path, columns=["date"])
            dates = self._dates(frame, path.name)
            dates["code"] = path.stem
            records.append(dates)
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["date", "code"])

    @staticmethod
    def _bottleneck(
        raw_dates: pd.DatetimeIndex,
        feature_dates: pd.DatetimeIndex,
        score_dates: pd.DatetimeIndex,
        eligible: pd.DatetimeIndex,
    ) -> str:
        if len(score_dates) < len(feature_dates):
            return (
                "The legacy composite-score history is shorter than the available feature history; "
                "this, rather than raw market-data coverage, limits V4 walk-forward observations."
            )
        if len(eligible) < len(score_dates):
            return "Cross-source date intersection or per-date universe coverage removes score dates."
        if len(raw_dates) < len(feature_dates):
            return "Raw price history is shorter than feature history and limits eligible signals."
        return "No single coverage bottleneck is detected; inspect the filtering-stage table."


def _date_range(dates: pd.DatetimeIndex) -> dict[str, str | None]:
    if dates.empty:
        return {"start": None, "end": None, "count": 0}
    return {"start": dates.min().date().isoformat(), "end": dates.max().date().isoformat(), "count": len(dates)}
