"""Core data types shared across the benchmark.

These are deliberately plain dataclasses (no numpy/pandas) so they serialize cleanly to
JSON for trajectory logs and are easy to reason about. Heavy numerics live in scoring.py.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------------------
# Point-in-time market data
# --------------------------------------------------------------------------------------


@dataclass
class PriceBar:
    """One daily OHLCV bar with the generator's return attribution.

    We trade in split-adjusted space: ``adj_close`` (and ``adj_open`` derived from it) are
    the prices used for accounting, so corporate actions never distort share counts. The
    ``*_return`` fields are the DGP's exact linear decomposition of that bar's return and are
    what make skill-vs-luck attribution exact rather than regressed.
    """

    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    market_return: float = 0.0
    sector_return: float = 0.0
    idiosyncratic_return: float = 0.0
    event_return: float = 0.0
    # Full 13-way generator decomposition from bar_metadata.return_components, when present.
    # This reconciles to the realized bar return up to price rounding; the four fields above
    # are a coarser summary that does not. Scoring prefers this dict and falls back to the four.
    components: dict[str, float] = field(default_factory=dict)

    @property
    def adj_factor(self) -> float:
        """adj_close / close, used to put open/high/low into adjusted space."""
        return self.adj_close / self.close if self.close else 1.0

    @property
    def adj_open(self) -> float:
        return self.open * self.adj_factor


@dataclass
class Doc:
    """A dated unstructured document (news, filing, transcript, 8-K, ...).

    ``date`` is the availability date used for point-in-time gating: a document is only
    visible to the agent on decision days on or after this date.
    """

    date: str
    ticker: str
    doc_type: str
    title: str
    period: str
    path: str
    words: int = 0
    _text: Optional[str] = field(default=None, repr=False)


@dataclass
class Fundamentals:
    """Latest-known structured fundamentals for a ticker as of a decision date.

    Every field is point-in-time: earnings/statements are gated by filing date, estimates by
    estimate date, holdings/insider records by their report/transaction date. Fields stay empty
    when the corpus does not carry that source (e.g. institutional/insider data is only present in
    corpora bundled with ownership panels)."""

    ticker: str
    as_of: str
    company: dict[str, Any] = field(default_factory=dict)
    latest_earnings: dict[str, Any] = field(default_factory=dict)
    ttm: dict[str, Any] = field(default_factory=dict)  # trailing-12m from filing-date-gated 10-Qs
    guidance: dict[str, Any] = field(default_factory=dict)
    statements: dict[str, Any] = field(default_factory=dict)  # latest 10-Q income/balance lines
    analyst: dict[str, Any] = field(default_factory=dict)  # consensus rating/target/revisions
    institutional: dict[str, Any] = field(default_factory=dict)  # 13F-style ownership summary
    insider: dict[str, Any] = field(default_factory=dict)  # Form-4-style recent insider activity


# --------------------------------------------------------------------------------------
# Agent I/O
# --------------------------------------------------------------------------------------


@dataclass
class TickerSnapshot:
    """Compact per-ticker row shown to the agent for the whole universe."""

    ticker: str
    sector: str
    last_price: float
    ret_1m: Optional[float]
    ret_3m: Optional[float]
    ret_12m: Optional[float]
    pe_ttm: Optional[float]
    held_weight: float = 0.0


@dataclass
class Observation:
    """Everything the agent sees on a single decision day (strictly point-in-time)."""

    scenario_id: str
    scenario_name: str
    decision_date: str
    step: int
    total_steps: int
    horizon_days: int  # trading days until the next decision
    cash_weight: float
    positions: dict[str, float]  # ticker -> current weight
    nav: float
    universe: list[TickerSnapshot]
    holdings_detail: dict[str, Fundamentals]  # deep data for current holdings + candidates
    recent_docs: list[dict[str, Any]]  # {date,ticker,doc_type,title,text} within the window
    macro: dict[str, Any]
    constraints: dict[str, Any]  # long_only, max_position_weight, etc.


@dataclass
class Action:
    """The agent's decision: a full target portfolio expressed as weights.

    ``target_weights`` is the *complete* desired book — any ticker not listed is treated as a
    0% target. The remainder (1 - sum) is held in cash. The simulator validates, clips and
    renormalizes these before executing, so an arithmetic slip by the model can never produce
    an invalid book.
    """

    target_weights: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    raw: Optional[str] = field(default=None, repr=False)  # raw model output, for the log


def asdict(obj: Any) -> Any:
    """dataclasses.asdict that drops private (leading-underscore) fields."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            if f.name.startswith("_"):
                continue
            out[f.name] = asdict(getattr(obj, f.name))
        return out
    if isinstance(obj, list):
        return [asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    return obj
