# Command Reference

All commands run through the project environment:

```bash
uv run quant <command>
```

Commands that read prices, features, models, or paper-trading state require their corresponding local artifacts. Those artifacts are intentionally ignored by Git.

## Data

| Command | Purpose |
| --- | --- |
| `data-update` | Download or refresh configured A-share market data. |
| `component-update` | Refresh index-component CSV snapshots through the component provider. |
| `industry-update` | Refresh industry information used by the universe and portfolio layers. |
| `universe-build` | Build the configured research universe. |
| `universe-check` | Validate universe membership and summary counts. |
| `benchmark-update` | Refresh the configured benchmark series. |
| `data-check` | Run local data-readiness checks. |
| `research-check` | Check research inputs, including universe and data requirements. |

## Research

| Command | Purpose |
| --- | --- |
| `factor-build` | Build factor values from local market data. |
| `factor-rank` | Rank or inspect configured factor signals. |
| `factor-analysis` | Produce factor diagnostics. |
| `alpha-research` | Run the factor-research workflow and produce research summaries. |
| `alpha-backtest` | Compare handcrafted and ML-ranking portfolio research results. |
| `trade-review` | Review realized paper-trading decisions and outcomes. |
| `strategy-diagnosis` | Diagnose recorded signal and strategy failures. |
| `strategy-history` | Inspect stored strategy-version history. |
| `live-simulation` | Run a local live-like research simulation; it is not broker execution. |

## Models

| Command | Purpose |
| --- | --- |
| `ml-train --model-name lightgbm` | Train a supported ranking model with the ML workflow. |
| `train` | Run the legacy configured training entry point. |
| `features-build` | Build feature and labelled-dataset artifacts used by training. |
| `signal` | Print the latest configured scored candidates. |

## Portfolio

| Command | Purpose |
| --- | --- |
| `portfolio-backtest` | Run the multi-factor portfolio backtest. |
| `industry-portfolio-backtest` | Run the industry-neutral portfolio workflow. |
| `institutional-backtest` | Run the institutional portfolio backtest and diagnostics. |
| `intelligent-backtest` | Run portfolio construction with risk-aware controls. |
| `regime-check` | Inspect the current market-regime classification. |
| `backtest` | Run the legacy configured core backtest. |

## Trading

| Command | Purpose |
| --- | --- |
| `daily-run` | Run the local daily research pipeline and produce candidates and review material. |
| `paper-start` | Initialize a paper-trading account state. |
| `paper-run` | Process a paper-trading cycle from local scores and prices. |
| `demo` | Run an offline synthetic-data pipeline check. |

## Validation

| Command | Purpose |
| --- | --- |
| `walk-forward` | Run chronological walk-forward validation. |
| `robustness` | Run perturbations for costs, weights, universe coverage, and rebalance settings. |

## Reporting

| Command | Purpose |
| --- | --- |
| `report-build` | Render an institutional backtest report. |
| `alpha-research` | Emit factor-research and quality reports. |
| `alpha-backtest` | Emit the portfolio-comparison report. |
| `daily-run` | Emit the daily research and strategy-evolution reports. |
| `walk-forward` | Emit a walk-forward validation report. |
| `robustness` | Emit a robustness report. |

For command options and defaults, run `uv run quant <command> --help`. Begin with `uv run quant doctor` when setting up a new local environment.
