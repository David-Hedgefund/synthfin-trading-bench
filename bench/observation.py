"""Builds the strictly point-in-time :class:`Observation` handed to an agent each period.

Nothing here may read a bar, doc, or filing dated after ``decision_date``. Candidate deep-data
selection (which tickers get full fundamentals + recent news) is momentum-based and uses only
trailing data, so it introduces no lookahead.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from .corpus import ScenarioData
from .schema import Observation, TickerSnapshot


def _trailing_return(scn: ScenarioData, ticker: str, day_index: int, lookback: int) -> Optional[float]:
    days = scn.trading_days
    if day_index - lookback < 0:
        return None
    d_now = days[day_index]
    d_then = days[day_index - lookback]
    b_now = scn.last_bar_on_or_before(ticker, d_now)
    b_then = scn.last_bar_on_or_before(ticker, d_then)
    if not b_now or not b_then or not b_then.adj_close:
        return None
    return b_now.adj_close / b_then.adj_close - 1.0


def build_observation(
    scn: ScenarioData,
    *,
    decision_date: str,
    day_index: int,
    step: int,
    total_steps: int,
    horizon_days: int,
    nav: float,
    cash_weight: float,
    positions: dict[str, float],
    config,
) -> Observation:
    tickers = scn.priced_tickers

    # -- compact universe snapshot (one row per ticker) -------------------------------
    # Price/return/sector only here — deep fundamentals are computed just for the in-focus names
    # below, so this stays cheap even for a 1000-ticker universe.
    snaps: list[TickerSnapshot] = []
    for tk in tickers:
        bar = scn.last_bar_on_or_before(tk, decision_date)
        if bar is None:
            continue
        snaps.append(
            TickerSnapshot(
                ticker=tk,
                sector=scn.sector_of(tk),
                last_price=round(bar.adj_close, 4),
                ret_1m=_round(_trailing_return(scn, tk, day_index, config.lb_1m)),
                ret_3m=_round(_trailing_return(scn, tk, day_index, config.lb_3m)),
                ret_12m=_round(_trailing_return(scn, tk, day_index, config.lb_12m)),
                pe_ttm=None,  # filled below for in-focus names from point-in-time TTM
                held_weight=round(positions.get(tk, 0.0), 4),
            )
        )

    if config.max_universe_rows and len(snaps) > config.max_universe_rows:
        # keep everything currently held, then fill by |3m momentum| so both legs are visible
        held = [s for s in snaps if s.held_weight != 0.0]
        rest = [s for s in snaps if s.held_weight == 0.0]
        rest.sort(key=lambda s: abs(s.ret_3m or 0.0), reverse=True)
        snaps = held + rest[: max(0, config.max_universe_rows - len(held))]

    # -- deep detail + news for holdings and top/bottom momentum candidates ------------
    # Cap deep detail to the largest holdings — computing fundamentals for every name in a
    # 1000-stock book is both slow and useless (no prompt can carry it). Bounded set = big
    # holdings + top/bottom momentum candidates.
    held_sorted = sorted(
        ((t, w) for t, w in positions.items() if abs(w) > 1e-9), key=lambda x: -abs(x[1])
    )
    held_tickers = [t for t, _ in held_sorted[: max(2 * config.detail_candidates, 20)]]
    ranked = sorted(snaps, key=lambda s: (s.ret_3m or 0.0), reverse=True)
    momentum_candidates = [s.ticker for s in ranked[: config.detail_candidates]]
    contrarian_candidates = [s.ticker for s in ranked[-config.detail_candidates :]]
    detail_tickers = list(dict.fromkeys(held_tickers + momentum_candidates + contrarian_candidates))

    holdings_detail = {tk: scn.fundamentals_as_of(tk, decision_date) for tk in detail_tickers}

    # fill point-in-time P/E for in-focus names: current close / trailing-12m EPS (no lookahead)
    snap_by_ticker = {s.ticker: s for s in snaps}
    for tk in detail_tickers:
        s = snap_by_ticker.get(tk)
        bar = scn.last_bar_on_or_before(tk, decision_date)
        ttm_eps = holdings_detail[tk].ttm.get("ttm_eps")
        if s and bar and isinstance(ttm_eps, (int, float)) and ttm_eps > 0:
            s.pe_ttm = round(bar.close / ttm_eps, 2)

    # -- recent documents within the news window --------------------------------------
    window_start = (
        datetime.strptime(decision_date, "%Y-%m-%d") - timedelta(days=config.news_window_days)
    ).strftime("%Y-%m-%d")
    docs = scn.docs_between(window_start, decision_date, tickers=detail_tickers)
    docs = sorted(docs, key=lambda d: d.date, reverse=True)[: config.max_docs]
    recent_docs = [
        {
            "date": d.date,
            "ticker": d.ticker,
            "doc_type": d.doc_type,
            "title": d.title,
            "text": scn.doc_text(d, config.doc_max_chars),
        }
        for d in docs
    ]

    return Observation(
        scenario_id=scn.meta.id,
        scenario_name=scn.meta.name,
        decision_date=decision_date,
        step=step,
        total_steps=total_steps,
        horizon_days=horizon_days,
        cash_weight=round(cash_weight, 4),
        positions={t: round(w, 4) for t, w in positions.items() if abs(w) > 1e-9},
        nav=round(nav, 2),
        universe=snaps,
        holdings_detail=holdings_detail,
        recent_docs=recent_docs,
        macro=scn.macro_as_of(decision_date),
        constraints={
            "long_only": config.long_only,
            "max_position_weight": config.max_position_weight,
            "gross_leverage": config.gross_leverage,
        },
    )


def _round(x: Optional[float], n: int = 4) -> Optional[float]:
    return round(x, n) if isinstance(x, (int, float)) else None
