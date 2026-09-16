# v4.0 Results Summary

## Status of this record

This document preserves the latest local v4.0 research artifacts in a compact, reviewable form. Values are historical outputs from this repository's research snapshot, not live performance, forecasts, or investment advice. Raw data, fitted models, and generated reports are deliberately excluded from Git; rerun the documented workflows to reproduce them locally.

## Data scale

| Item | Recorded value |
| --- | ---: |
| Configured A-share universe | 809 stocks |
| Local price histories | 811 files |
| Mean history per price series | 2,353 trading days |
| LightGBM training rows | 851,206 |
| LightGBM validation rows | 330,554 |
| LightGBM test rows | 305,272 |

## Software quality

The release check runs the full `pytest` suite and `ruff check .`. The v4.0 release check collects 119 tests spanning the core pipeline, data validation, factors, models, portfolio construction, risk controls, paper trading, memory, attribution, walk-forward validation, and robustness.

## Held-out ML metric

| Model | Prediction IC | Rank IC | Reported observations |
| --- | ---: | ---: | ---: |
| LightGBM ranking | 0.0234 | 0.0332 | 379 |

The positive held-out association is modest. It must be interpreted jointly with portfolio results below: it did not produce positive net performance in the current implementation.

## Portfolio comparison

| Historical portfolio result | Annual return | Sharpe | Max drawdown | Alpha | Beta | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Handcrafted | -2.15% | -0.115 | -21.09% | -1.22% | 0.531 | 3.948 |
| ML ranking | -2.75% | -0.169 | -21.09% | -1.86% | 0.522 | 3.948 |

Neither recorded candidate meets a reasonable profitability threshold. The ML model's positive IC therefore remains a research signal, not a trading result.

## Walk-forward validation

The recorded annual walk-forward artifact contains the following strategy returns:

| Period | Strategy return | Sharpe | Drawdown |
| --- | ---: | ---: | ---: |
| 2019 | -62.42% | -1.219 | -61.40% |
| 2020 | 28.95% | 1.342 | -9.01% |
| 2021 | 101.14% | 1.778 | -16.26% |
| 2022 | 26.25% | 1.518 | -7.10% |
| 2023 | 17.96% | 1.708 | -2.34% |
| 2024 | 20.27% | 1.184 | -6.04% |
| 2025 | 51.83% | 3.078 | -4.26% |
| 2026 | 8.00% | 0.732 | -7.41% |

The benchmark field in this particular historical artifact is 0.0 for every period. Accordingly, its alpha column is not a valid benchmark-relative claim. The severe 2019 loss and the missing meaningful benchmark series prevent this table from establishing robust out-of-sample alpha.

## Robustness framework and all recorded scenarios

The robustness suite varies rebalancing, transaction costs, factor weights, and universe coverage. All 10 recorded scenarios are reported here to avoid cherry-picking.

| Category | Scenario | Annual return | Sharpe | Max drawdown |
| --- | --- | ---: | ---: | ---: |
| Rebalance frequency | 5 days | -33.99% | -2.128 | -8.18% |
| Rebalance frequency | 10 days (base) | -47.45% | -3.385 | -10.89% |
| Rebalance frequency | 20 days | -30.27% | -1.888 | -8.31% |
| Transaction cost | 0.5× | -46.08% | -3.238 | -10.60% |
| Transaction cost | 1.0× (base) | -47.45% | -3.385 | -10.89% |
| Transaction cost | 2.0× | -50.09% | -3.672 | -11.58% |
| Factor-weight perturbation | -5% | -80.96% | -3.231 | -29.50% |
| Factor-weight perturbation | +5% | -85.75% | -3.336 | -32.17% |
| Universe change | 80% of universe | -38.52% | -3.273 | -8.01% |
| Universe change | 100% of universe (base) | -47.45% | -3.385 | -10.89% |

These results reject the current factor-weighted strategy as a robust alpha candidate. In particular, modest factor-weight perturbations materially worsen outcomes, indicating sensitivity rather than stability.

## Risk engineering across versions

The system's risk engineering has expanded through the project's development: industry-neutral construction, explicit turnover and transaction costs, market-regime detection, volatility scaling, drawdown scaling, execution accounting, paper trading, and scenario-based validation are now first-class components. This is a meaningful reduction in unmodelled implementation risk compared with a simple score-only backtest.

No controlled, like-for-like historical series of pre-v4 versus v4 risk metrics is preserved in the repository. It would be misleading to claim a numeric cross-version drawdown reduction without that evidence. The v4.0 record instead demonstrates that the stronger controls expose a negative alpha baseline under adverse assumptions.

## Conclusion

Risk and validation infrastructure improved substantially; alpha generation did not. The next research problem is Alpha Reconstruction: economically grounded, point-in-time-aware fundamental and market hypotheses that are evaluated under the same strict validation and execution standards.
