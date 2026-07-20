# Methodology

This document specifies the task, the simulation protocol, and the scoring so that results are
reproducible and comparable across models. It is the reference for the leaderboard.

## 1. Task

An agent acts as a portfolio manager over one synthetic **scenario** — a multi-year market with a
fixed universe of tickers. Time advances in trading days. On a fixed cadence
(`rebalance_every_days`, default 21 ≈ monthly) the agent is given a point-in-time observation and
must return a **target portfolio** expressed as weights per ticker. Everything not held is cash.

The agent is *single-shot per decision*: it sees the observation and returns a target. It does not
place intraday orders or run a tool loop. This keeps cost bounded and the comparison across
providers clean (identical prompt, identical information).

## 2. Point-in-time discipline (no lookahead)

On decision day `t`, the observation may contain **only** information knowable at the close of `t`.
Every SynthFin data source is exposed to the agent, each gated by its own availability date:

| Source | Field | Point-in-time gate |
|---|---|---|
| Daily prices (OHLCV, adj close) | `universe[].last_price`, trailing 1m/3m/12m returns | bar date ≤ `t` |
| Earnings (8-K) | `holdings_detail[].latest_earnings` (EPS/rev surprise) | `filing_date` ≤ `t` |
| Quarterly financials (10-Q) | `holdings_detail[].statements` (income stmt + balance sheet) | `filing_date` ≤ `t` |
| TTM fundamentals | `holdings_detail[].ttm`, `universe[].pe_ttm` | **derived** from the last 4 filed 10-Qs |
| Analyst estimates | `holdings_detail[].analyst` (consensus target, rating mix, revisions) | `estimate_date` ≤ `t` |
| Guidance | `holdings_detail[].guidance` | issue date ≤ `t` |
| Institutional 13F holdings | `holdings_detail[].institutional` | report date ≤ `t` (when corpus carries it) |
| Insider (Form 4) trades | `holdings_detail[].insider` (trailing net buy/sell) | transaction date ≤ `t` (when present) |
| News, 10-K/10-Q MD&A, risk factors, 8-Ks, transcripts, market news | `recent_docs` | doc `date` ≤ `t`, within `news_window_days` |
| Macro series | `macro` | series date ≤ `t` |

**Deliberate lookahead fix:** the corpus's `metrics_ttm.json` is a single *end-of-scenario* snapshot
with no availability date, so it is **not** used; TTM revenue/EPS/margins and P/E are recomputed from
the four most recent filing-date-gated 10-Qs (`corpus._ttm_from_statements`).

**Coverage vs. cost:** the compact universe table (price/returns/sector, and P/E for in-focus names)
spans the whole universe so the agent can screen breadth; **deep** data (financials, estimates,
ownership, news/filing text) is attached for current holdings plus the top/bottom `detail_candidates`
by trailing momentum — feeding every filing for 1000 names each period is infeasible. This focus set
is chosen from trailing data only, so it introduces no lookahead; widen it via `detail_candidates`,
`max_universe_rows`, `max_docs`. Gating is enforced in `corpus.py` and covered by
`tests/test_bench.py` (`test_no_lookahead_docs_and_fundamentals`, `test_ttm_is_point_in_time_not_terminal_snapshot`).

## 3. Execution & accounting

- A target chosen at `t` executes at the **next** session's adjusted open (`t+1`). This one-bar
  delay removes same-bar lookahead.
- **Transaction cost:** `cost_bps` basis points charged on turnover (dollar value traded) each
  rebalance. Default 5 bps.
- **Split-adjusted space:** positions are held as adjusted-space units; NAV is marked daily at
  adjusted close. Corporate actions therefore never distort share counts or P&L.
- **Constraints** (validated and clipped every period, so malformed model output can never produce
  an invalid book): `long_only`, `max_position_weight`, `gross_leverage`. Weights exceeding gross
  leverage are renormalized; negative weights under long-only are dropped.

## 4. Metrics

### 4.1 Net performance (from the realized NAV series)
Total return, CAGR, annualized volatility, **Sharpe**, **Sortino**, **max drawdown**, Calmar,
average turnover, and cost drag. These are what the book actually earns after costs.

### 4.2 Skill vs the market — the headline
We regress the book's realized daily returns `r_p` on the market proxy's realized returns `r_m` — a
CAPM/single-index regression. The proxy is an index ETF when the corpus has one (SPY → QQQ → DIA);
for an operating-only universe with no ETF (e.g. the 1000-name corpus) it falls back to the
**equal-weight universe index**, a self-contained broad-market factor, so the metric is always
well-defined and comparable across corpora (`market_proxy` records which was used):

```
r_p(d) = alpha + beta · r_m(d) + ε(d)
```

- **beta_market** — market exposure.
- **alpha_ann** — annualized Jensen's alpha, `alpha · 252`.
- **appraisal_ratio** — `alpha / std(ε) · √252`, the annualized information ratio of the
  market-neutral residual. This is the standard risk-adjusted measure of stock-selection skill and
  is the primary ranking-adjacent skill number. It uses only realized, tradeable prices, so it is
  fully defensible independent of any generator internals.

We also report `benchmark_return`, `excess_return` (book − proxy), and `benchmark_win_rate` (share
of scenarios beating the proxy).

### 4.3 Generator attribution — diagnostic, approximate
The synthetic generator labels every bar's return with latent drivers (market, sector, stress,
idiosyncratic, jumps, mean-reversion, …). Weighting by the book's holdings gives an *intended-driver*
breakdown: `gen_market_contribution`, `gen_sector_contribution`, `gen_selection_contribution`
(name-specific), and `gen_residual_contribution`. The labeled drivers **do not** exactly reconstruct
the realized price path — the residual absorbs the unexplained part — so this attribution is a
**diagnostic that supplements, never replaces, §4.2**. It is unique to synthetic data and useful for
sanity-checking *why* a book made money, but the leaderboard ranks on realized, regression-based
metrics.

## 5. Aggregation & ranking

Per-scenario scores are averaged per agent. The leaderboard is **ranked by mean Sharpe** across
scenarios, with alpha, appraisal ratio, beta, drawdown, excess return and win rate shown alongside.
Cross-scenario standard deviations are reported for Sharpe, total return, alpha and appraisal so
readers can see stability, not just central tendency. Always include the baselines — a model's
numbers are only meaningful relative to buy-and-hold, momentum, and random.

## 6. Reproducibility contract

A result is fully described by:

1. **Corpus content hash** (`scripts/hash_corpus.py`) — the frozen dataset version.
2. **RunConfig** — every knob (cadence, costs, constraints, observation shaping), saved in
   `run_meta.json`.
3. **Model id + decoding params** — pinned model snapshots, `temperature=0` by default, recorded
   verbatim in each trajectory.
4. **Seed** — baselines and any stochastic component are seeded.

Given the same four, a run reproduces (modulo provider-side nondeterminism, which we bound with
`temperature=0` and report as run-to-run variance where relevant).

## 7. Known limitations

- **Provider nondeterminism.** Even at `temperature=0`, hosted models are not bit-reproducible.
  Report multi-seed / multi-run variance for headline claims.
- **Single-shot decisions.** No intraperiod trading or tool use; this is a deliberate scope choice.
- **Synthetic realism.** Scenarios are calibrated to be realistic but are not real markets; the
  benchmark measures decision quality under a known DGP, not live-market P&L.
- **Prompt sensitivity.** All models share one prompt (`bench/agents/prompt.py`); it is held fixed
  across the leaderboard and versioned with the repo.
