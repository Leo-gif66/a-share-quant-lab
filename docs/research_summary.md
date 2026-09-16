# Research Summary for Supervisor

## Problem statement

The project asks whether an extensible, reproducible research system can turn publicly available A-share market data into robust cross-sectional equity signals after realistic portfolio constraints and costs. The research question is intentionally stronger than whether a single backtest can look attractive: a candidate must survive chronological validation, perturbations, and risk-aware implementation.

## Motivation and system evolution

The lab was built to replace an ad hoc script workflow with a traceable research stack. It evolved from data ingestion, factor calculation, and simple backtests into a v4.0 baseline with a large configured universe, provider abstraction, factor diagnostics, ranking models, industry-neutral construction, risk overlays, paper trading, trade memory, attribution, and institutional-style validation. The current release is an engineering and validation milestone; it does not claim a solved alpha problem.

## Dataset scale

The configured universe contains 809 A-share securities. The local snapshot used for the recorded research contains 811 price histories (the additional files are non-universe market series) with a mean of 2,353 trading days per price series. This supports multi-year factor analysis and chronological splits, while remaining small enough for a locally reproducible workflow.

## Methodology

- Market data are obtained through replaceable Tencent/AKShare-oriented provider interfaces and checked before research use.
- Factor panels and future-return labels are aligned by date and security. Chronological train/validation/test partitions, with label-horizon purging where configured, address obvious temporal leakage.
- Factor candidates are evaluated with IC, Rank IC, coverage, stability, and long–short diagnostics rather than only in-sample fit.
- Portfolio tests impose concentration, industry, regime, volatility, drawdown, turnover, and transaction-cost considerations.
- Walk-forward and robustness runs perturb rebalance frequency, cost assumptions, factor weights, and universe coverage. Attribution and trade-memory outputs support post-hoc diagnosis.

## ML approach

The model layer exposes ranking-model interfaces and includes LightGBM alongside linear and random-forest baselines. The recorded LightGBM artifact used 851,206 training rows, 330,554 validation rows, and 305,272 test rows. Its held-out Prediction IC was 0.0234 and Rank IC was 0.0332 across 379 reported observations. These are model-level association measures, not evidence of deployable portfolio alpha.

## Risk-management approach

Risk engineering is applied after score generation: industry-neutral target construction reduces unintended sector concentration; market-regime, realized-volatility, and drawdown controls scale exposure; execution accounting applies costs and turnover; and paper trading records operational state. The validation layer reports drawdowns and scenario sensitivity rather than relying on a headline return alone.

## Current findings

The most important result is negative. In the recorded portfolio comparison, the handcrafted portfolio returned -2.15% annually (Sharpe -0.115) and the ML-ranking portfolio returned -2.75% annually (Sharpe -0.169). In the recorded robustness base scenario, annual return was -47.45% with a -3.39 Sharpe ratio. All recorded robustness perturbations remained negative.

The walk-forward research artifact contains a strongly negative 2019 period (-62.42%) and positive later periods. Its benchmark column is recorded as zero throughout, so it cannot support an excess-return conclusion. The result should therefore be treated as a diagnostic output, not as evidence of performance persistence.

## Why the negative findings matter

The evidence distinguishes two claims that are often conflated: the system can produce weakly positive model-level ranking association, and the current strategy can generate reliable net alpha. The first has limited support in the LightGBM metric; the second is contradicted by the portfolio and robustness evidence. Retaining this distinction makes the repository a credible platform for hypothesis testing and prevents result selection from obscuring failure modes.

## Limitations

- Public-provider data can change, be revised, rate-limited, or be unavailable.
- The present snapshot is not a full point-in-time fundamental database; survivorship and revision bias remain material concerns.
- Daily-bar execution approximates liquidity, suspension, price-limit, and market-impact effects.
- Current validation artifacts are local historical outputs, not a preregistered study or live-trading evidence.
- The strategy baseline is negative after the available validation procedures.

## Dissertation and research directions

1. Build a point-in-time fundamental and corporate-action data layer, including survivorship-aware historical membership.
2. Develop economically motivated alpha hypotheses and evaluate them with purged, embargoed, and regime-conditional validation.
3. Compare cross-sectional ranking, calibrated return forecasting, and risk-aware objective functions under fixed execution assumptions.
4. Quantify how industry neutrality, volatility targeting, and drawdown controls change tail risk independently of signal quality.
5. Model A-share-specific microstructure—price limits, suspensions, liquidity, and market impact—and test whether apparent signals survive implementation.
