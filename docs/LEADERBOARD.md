# SynthFin Trading Bench — Leaderboard

- **Run:** `baselines_1000x10`  
- **Corpus:** `research1000x10` (1000 tickers × 10 scenarios)  
- **Corpus hash:** `fa7e8c9e4a0173b8…`  
- **Rebalance:** every 21 trading days  |  **Cost:** 5 bps

Ranked by mean Sharpe across scenarios. **Appraisal** = annualized alpha / idiosyncratic vol from a CAPM regression of the book's realized returns on the market proxy — the risk-adjusted stock-selection skill net of market beta. **Alpha** is annualized Jensen's alpha. Both use only realized, tradeable prices.

| # | Agent | Mean Ret | Sharpe | Sortino | MaxDD | Alpha (ann) | Appraisal | Beta | Excess vs mkt | Win rate | Turnover | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **baseline_ew_rebal** | +4.84% | +0.28 | +0.38 | -23.06% | +0.12% | +0.14 | +0.99 | +0.25% | +60.00% | +10.53% | 0 |
| 2 | **baseline_buyhold** | +0.25% | +0.10 | +0.17 | -23.54% | -2.41% | -4.51 | +0.99 | -4.33% | +0.00% | +7.27% | 0 |
| 3 | **baseline_random** | -2.01% | -0.00 | +0.06 | -25.12% | -3.68% | -0.65 | +1.00 | -6.60% | +30.00% | +190.91% | 0 |
| 4 | **grok-4.5** | -15.66% | -0.59 | -0.70 | -30.02% | -12.96% | -1.64 | +0.96 | -20.25% | +20.00% | +55.85% | 0 |
| 5 | **gpt-5.4-mini** | -17.73% | -0.74 | -0.95 | -28.06% | -14.66% | -2.08 | +0.91 | -22.32% | +0.00% | +44.92% | 0 |
| 6 | **gemini-3.5-flash** | -24.31% | -0.97 | -1.31 | -33.02% | -18.83% | -2.62 | +0.98 | -28.89% | +0.00% | +60.19% | 0 |
| 7 | **baseline_momentum** | -26.53% | -1.12 | -1.36 | -33.46% | -20.79% | -4.83 | +1.02 | -31.12% | +0.00% | +120.07% | 0 |

> Mean Ret, MaxDD, Alpha and Excess are per-scenario averages. Sharpe/Sortino/Appraisal and Alpha are annualized. Win rate is the fraction of scenarios beating the market proxy (SPY/QQQ). A separate *generator attribution* (approximate) is available per scenario in the score files.
