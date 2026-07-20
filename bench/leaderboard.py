"""Render an aggregated leaderboard as Markdown, and a CLI to (re)build it from a results dir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def _pct(x: Optional[float]) -> str:
    return f"{x:+.2%}" if isinstance(x, (int, float)) else "—"


def _num(x: Optional[float], n: int = 2) -> str:
    return f"{x:+.{n}f}" if isinstance(x, (int, float)) else "—"


def render_markdown(leaderboard: list[dict[str, Any]], run_meta: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# SynthFin Trading Bench — Leaderboard\n")
    lines.append(
        f"- **Run:** `{run_meta.get('run_id')}`  \n"
        f"- **Corpus:** `{run_meta.get('corpus_profile')}` "
        f"({run_meta.get('n_tickers')} tickers × {len(run_meta.get('scenarios', []))} scenarios)  \n"
        f"- **Corpus hash:** `{run_meta.get('corpus_hash', '')[:16]}…`  \n"
        f"- **Rebalance:** every {run_meta.get('run_config', {}).get('rebalance_every_days')} "
        f"trading days  |  **Cost:** {run_meta.get('run_config', {}).get('cost_bps')} bps\n"
    )
    lines.append(
        "Ranked by mean Sharpe across scenarios. **Appraisal** = annualized alpha / idiosyncratic "
        "vol from a CAPM regression of the book's realized returns on the market proxy — the "
        "risk-adjusted stock-selection skill net of market beta. **Alpha** is annualized Jensen's "
        "alpha. Both use only realized, tradeable prices.\n"
    )
    header = (
        "| # | Agent | Mean Ret | Sharpe | Sortino | MaxDD | Alpha (ann) | Appraisal | Beta | "
        "Excess vs mkt | Win rate | Turnover | Errors |"
    )
    sep = "|" + "|".join(["---"] * 13) + "|"
    lines.append(header)
    lines.append(sep)
    for i, row in enumerate(leaderboard, 1):
        lines.append(
            f"| {i} | **{row.get('agent_name')}** | {_pct(row.get('mean_total_return'))} | "
            f"{_num(row.get('mean_sharpe'))} | {_num(row.get('mean_sortino'))} | "
            f"{_pct(row.get('mean_max_drawdown'))} | {_pct(row.get('mean_alpha_ann'))} | "
            f"{_num(row.get('mean_appraisal_ratio'))} | {_num(row.get('mean_beta_market'))} | "
            f"{_pct(row.get('mean_excess_return'))} | "
            f"{_pct(row.get('benchmark_win_rate'))} | {_pct(row.get('mean_avg_turnover'))} | "
            f"{row.get('n_errors', 0)} |"
        )
    lines.append(
        "\n> Mean Ret, MaxDD, Alpha and Excess are per-scenario averages. Sharpe/Sortino/Appraisal "
        "and Alpha are annualized. Win rate is the fraction of scenarios beating the market proxy "
        "(SPY/QQQ). A separate *generator attribution* (approximate) is available per scenario in "
        "the score files.\n"
    )
    return "\n".join(lines)


def build_from_results(results_dir: str | Path) -> str:
    from .scoring import aggregate

    results_dir = Path(results_dir)
    run_meta = json.loads((results_dir / "run_meta.json").read_text())
    scores: dict[str, list[dict[str, Any]]] = {}
    for sf in (results_dir / "scores").glob("*.json"):
        s = json.loads(sf.read_text())
        if "agent_name" in s:
            scores.setdefault(s["agent_name"], []).append(s)
    leaderboard = [aggregate(v) for v in scores.values()]
    leaderboard.sort(key=lambda r: (r.get("mean_sharpe") or -99), reverse=True)
    (results_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    md = render_markdown(leaderboard, run_meta)
    (results_dir / "leaderboard.md").write_text(md)
    return md


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Rebuild the leaderboard from a results directory.")
    ap.add_argument("results_dir", help="results/<run_id> directory")
    args = ap.parse_args(argv)
    print(build_from_results(args.results_dir))


if __name__ == "__main__":
    main()
