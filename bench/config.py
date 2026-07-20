"""Run configuration. Every knob that can change a benchmark number lives here so a run is
fully described by (corpus content-hash, RunConfig, model id + decoding params, seed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    # --- market / calendar ---
    rebalance_every_days: int = 21  # trading days between decisions (21 ~ monthly, 5 ~ weekly)
    warmup_days: int = 63  # skip this many leading days so momentum signals have history
    max_decisions: int = 0  # 0 = run to end of scenario; else cap (cost control)

    # --- portfolio rules ---
    initial_nav: float = 1_000_000.0
    long_only: bool = True
    max_position_weight: float = 0.10
    gross_leverage: float = 1.0
    cost_bps: float = 5.0  # transaction cost per unit turnover, in basis points

    # --- observation shaping ---
    lb_1m: int = 21
    lb_3m: int = 63
    lb_12m: int = 252
    max_universe_rows: int = 0  # 0 = show all tickers; else cap the compact snapshot table
    detail_candidates: int = 15  # extra tickers (each side) that get deep data + news
    news_window_days: int = 30
    max_docs: int = 40
    doc_max_chars: int = 1500

    # --- reproducibility ---
    seed: int = 12345

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class BenchConfig:
    """Top-level config: which corpus, which agents, and the shared RunConfig."""

    corpus_path: str = ""
    scenarios: list[str] = field(default_factory=list)  # slugs/ids; empty = all
    agents: list[dict[str, Any]] = field(default_factory=list)  # [{name, provider, model, ...}]
    run: RunConfig = field(default_factory=RunConfig)
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        run = RunConfig(**(raw.pop("run", {}) or {}))
        return cls(
            corpus_path=raw.get("corpus_path", ""),
            scenarios=raw.get("scenarios", []) or [],
            agents=raw.get("agents", []) or [],
            run=run,
            output_dir=raw.get("output_dir", "results"),
        )
