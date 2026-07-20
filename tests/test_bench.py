"""Invariant tests for the benchmark harness.

The corpus-dependent tests use the local 50x10 export and skip automatically if it isn't
present, so the pure-logic tests still run anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.config import RunConfig
from bench.corpus import Corpus
from bench.scoring import score_trajectory
from bench.simulator import Simulator, validate_target
from bench.agents.baselines import Momentum

CORPUS = os.environ.get(
    "SFTB_TEST_CORPUS",
    str(Path(__file__).resolve().parents[1] / "data" / "synthfin_50x10"),
)
have_corpus = Path(CORPUS).exists()
corpus_required = pytest.mark.skipif(not have_corpus, reason=f"no corpus at {CORPUS}")


# ------------------------------------------------------------------ pure logic ---------


def test_validate_target_long_only_clips_negatives():
    cfg = RunConfig(long_only=True, max_position_weight=0.1, gross_leverage=1.0)
    out = validate_target({"AAPL": -0.5, "MSFT": 0.05}, {"AAPL", "MSFT"}, cfg)
    assert "AAPL" not in out  # negative clipped away under long-only
    assert out["MSFT"] == pytest.approx(0.05)


def test_validate_target_caps_position_and_gross():
    cfg = RunConfig(long_only=True, max_position_weight=0.1, gross_leverage=1.0)
    out = validate_target({t: 0.5 for t in ["A", "B", "C", "D", "E"]}, set("ABCDE"), cfg)
    assert all(w <= 0.1 + 1e-9 for w in out.values())  # per-name cap
    assert sum(out.values()) <= 1.0 + 1e-9  # gross cap


def test_validate_target_drops_unknown_and_nan():
    cfg = RunConfig()
    out = validate_target({"AAPL": 0.05, "ZZZZ": 0.1, "MSFT": float("nan")}, {"AAPL"}, cfg)
    assert out == {"AAPL": pytest.approx(0.05)}


# ------------------------------------------------------------- point-in-time -----------


@corpus_required
def test_no_lookahead_docs_and_fundamentals():
    corpus = Corpus(CORPUS)
    scn = corpus.load(corpus.scenarios()[0])
    cutoff = scn.trading_days[len(scn.trading_days) // 2]
    for doc in scn.docs_between("0000-00-00", cutoff):
        assert doc.date <= cutoff, f"doc dated {doc.date} leaked past {cutoff}"
    tk = scn.priced_tickers[0]
    fnd = scn.fundamentals_as_of(tk, cutoff)
    fd = fnd.latest_earnings.get("filing_date")
    if fd:
        assert fd <= cutoff, "earnings filing leaked from the future"
    # analyst estimates gated by estimate_date
    if fnd.analyst.get("latest_estimate_date"):
        assert fnd.analyst["latest_estimate_date"] <= cutoff, "analyst estimate leaked"
    # 10-Q statement gated by report/filing period
    if fnd.statements.get("report_period"):
        assert fnd.statements["report_period"] <= cutoff, "10-Q statement leaked from the future"


@corpus_required
def test_ttm_is_point_in_time_not_terminal_snapshot():
    """Regression guard: TTM must be derived from filed quarters as of the date, so it changes
    across the timeline — not the single end-of-scenario metrics_ttm.json snapshot (lookahead)."""
    corpus = Corpus(CORPUS)
    scn = corpus.load(corpus.scenarios()[0])
    tk = scn.priced_tickers[0]
    early = scn.fundamentals_as_of(tk, scn.trading_days[80]).ttm
    late = scn.fundamentals_as_of(tk, scn.trading_days[-1]).ttm
    # both point-in-time; the later one should reflect more/newer filings (revenue differs)
    if early.get("ttm_revenue") and late.get("ttm_revenue"):
        assert early["ttm_revenue"] != late["ttm_revenue"], "TTM looks static → possible lookahead"


# ---------------------------------------------------------------- simulator ------------


@corpus_required
def test_accounting_identity_and_no_future_exec():
    corpus = Corpus(CORPUS)
    scn = corpus.load(corpus.scenarios()[0])
    cfg = RunConfig(rebalance_every_days=42, warmup_days=63, max_decisions=6)
    traj = Simulator(cfg).run(Momentum(top_frac=0.2), scn, agent_meta={"agent_name": "m"})

    # dates strictly increasing
    ds = [r["date"] for r in traj["nav_series"]]
    assert ds == sorted(ds) and len(set(ds)) == len(ds)

    # every execution happens strictly after its decision
    for rb in traj["rebalances"]:
        assert rb["exec_date"] > rb["decision_date"]

    # NAV stays finite and positive throughout
    navs = [r["nav"] for r in traj["nav_series"]]
    assert all(n > 0 for n in navs)
    assert traj["nav_series"][0]["nav"] == pytest.approx(cfg.initial_nav, rel=1e-6)


# ------------------------------------------------------------------ scoring ------------


@corpus_required
def test_generator_attribution_identity_and_capm_finite():
    """The generator attribution buckets + residual are an exact additive split of the book's
    gross return (residual absorbs whatever the labeled drivers miss). Separately, the CAPM
    skill metrics must be finite and beta must be in a sane range."""
    corpus = Corpus(CORPUS)
    scn = corpus.load(corpus.scenarios()[0])
    cfg = RunConfig(rebalance_every_days=42, warmup_days=63, max_decisions=6)
    traj = Simulator(cfg).run(Momentum(top_frac=0.2), scn, agent_meta={"agent_name": "m"})
    s = score_trajectory(traj, scn)

    # additive identity (exact by construction, small tolerance for compounding vs additive gross)
    bucket_sum = (
        s["gen_market_contribution"]
        + s["gen_sector_contribution"]
        + s["gen_selection_contribution"]
        + s["gen_residual_contribution"]
    )
    assert abs(bucket_sum - s["gross_book_return"]) < 0.02

    # headline skill metrics are finite and sane
    assert s["appraisal_ratio"] == s["appraisal_ratio"]  # not NaN
    assert s["alpha_ann"] == s["alpha_ann"]
    assert -3.0 < s["beta_market"] < 3.0
    assert s["market_proxy"] in ("SPY", "QQQ", "DIA", "EW_UNIVERSE")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
