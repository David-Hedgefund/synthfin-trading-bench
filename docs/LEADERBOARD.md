# SynthFin Trading Bench — Leaderboard

- **Run:** `v1_1000x10` (baselines + models)  
- **Corpus:** `research1000x10` (1000 tickers × 10 scenarios)  
- **Corpus hash:** `fa7e8c9e4a0173b8…`  
- **Rebalance:** every 21 trading days  |  **Cost:** 5 bps

Ranked by mean Sharpe across scenarios. **Appraisal** = annualized alpha / idiosyncratic vol from a CAPM regression of the book's realized returns on the market proxy — the risk-adjusted stock-selection skill net of market beta. **Alpha** is annualized Jensen's alpha. Both use only realized, tradeable prices.

| # | Agent | Mean Ret | Sharpe | Sortino | MaxDD | Alpha (ann) | Appraisal | Beta | Excess vs mkt | Win rate | Turnover | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **baseline_ew_rebal** | +4.84% | +0.28 | +0.38 | -23.06% | +0.12% | +0.14 | +0.99 | +0.25% | +60.00% | +10.53% | 0 |
| 2 | **baseline_buyhold** | +0.25% | +0.10 | +0.17 | -23.54% | -2.41% | -4.51 | +0.99 | -4.33% | +0.00% | +7.27% | 0 |
| 3 | **claude-fable-5** | -8.90% | -0.36 | -0.44 | -26.18% | -8.07% | -1.08 | +0.97 | -13.49% | +40.00% | +37.81% | 0 |
| 4 | **grok-4.5** | -15.66% | -0.59 | -0.70 | -30.02% | -12.96% | -1.64 | +0.96 | -20.25% | +20.00% | +55.85% | 0 |
| 5 | **gpt-5.5** | -16.40% | -0.64 | -0.77 | -29.21% | -13.58% | -1.81 | +0.94 | -20.99% | +20.00% | +49.37% | 0 |
| 6 | **gemini-3.5-flash** | -24.31% | -0.97 | -1.31 | -33.02% | -18.83% | -2.62 | +0.98 | -28.89% | +0.00% | +60.19% | 0 |
| 7 | **baseline_momentum** | -26.53% | -1.12 | -1.36 | -33.46% | -20.79% | -4.83 | +1.02 | -31.12% | +0.00% | +120.07% | 0 |

> Mean Ret, MaxDD, Alpha and Excess are per-scenario averages. Sharpe/Sortino/Appraisal and Alpha are annualized. Win rate is the fraction of scenarios beating the market proxy (SPY/QQQ). A separate *generator attribution* (approximate) is available per scenario in the score files.
