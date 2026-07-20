"""Orchestrates a benchmark run: (agents x scenarios) -> trajectories -> scores -> leaderboard.

Everything a run produces is written under ``results/<run_id>/`` so it is fully reproducible and
inspectable: the exact config, the corpus content-hash, every decision trajectory, every score,
and the aggregated leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .agents import build_agent
from .config import BenchConfig
from .corpus import Corpus, ScenarioMeta
from .leaderboard import render_markdown
from .schema import asdict
from .scoring import aggregate, score_trajectory
from .simulator import Simulator


def _select(metas: list[ScenarioMeta], wanted: list[str]) -> list[ScenarioMeta]:
    if not wanted:
        return metas
    keep = []
    for m in metas:
        if m.slug in wanted or m.id in wanted or m.slug.split("_")[0] in wanted:
            keep.append(m)
    return keep


def run_benchmark(cfg: BenchConfig, *, run_id: str = "", verbose: bool = True) -> dict[str, Any]:
    corpus = Corpus(cfg.corpus_path)
    metas = _select(corpus.scenarios(), cfg.scenarios)
    if not metas:
        raise SystemExit(f"no scenarios matched {cfg.scenarios!r} in {cfg.corpus_path}")

    run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    out = Path(cfg.output_dir) / run_id
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)

    corpus_hash = corpus.content_hash()
    run_meta = {
        "run_id": run_id,
        "corpus_path": str(corpus.root),
        "corpus_profile": corpus.profile,
        "corpus_hash": corpus_hash,
        "n_tickers": len(corpus.tickers),
        "scenarios": [m.slug for m in metas],
        "agents": cfg.agents,
        "run_config": cfg.run.to_dict(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    sim = Simulator(cfg.run)
    all_scores: dict[str, list[dict[str, Any]]] = {}

    # Load each scenario once, run every agent against it (amortizes corpus I/O).
    for meta in metas:
        if verbose:
            print(f"[scenario] {meta.slug}", file=sys.stderr)
        scn = corpus.load(meta)
        for spec in cfg.agents:
            agent = build_agent(spec)
            tag = f"{agent.name}__{meta.slug}"
            try:
                agent_meta = {**agent.meta(), "corpus_hash": corpus_hash}
                traj = sim.run(agent, scn, agent_meta=agent_meta)
                traj_ser = _serialize(traj)
                (out / "trajectories" / f"{tag}.json").write_text(json.dumps(traj_ser))
                score = score_trajectory(traj, scn)
                (out / "scores" / f"{tag}.json").write_text(json.dumps(score, indent=2))
                all_scores.setdefault(agent.name, []).append(score)
                if verbose:
                    print(
                        f"    {agent.name:24s} ret={score.get('total_return', 0):+.2%} "
                        f"sharpe={score.get('sharpe', 0):+.2f} "
                        f"alpha={score.get('alpha_ann', 0):+.2%} "
                        f"appraisal={score.get('appraisal_ratio', 0):+.2f} "
                        f"errs={score.get('n_errors', 0)}",
                        file=sys.stderr,
                    )
            except Exception as exc:  # noqa: BLE001 — one agent's failure must not abort the run
                print(f"    !! {agent.name} failed on {meta.slug}: {exc}", file=sys.stderr)
                traceback.print_exc()

    leaderboard = [aggregate(v) for v in all_scores.values()]
    leaderboard.sort(key=lambda r: (r.get("mean_sharpe") or -99), reverse=True)
    (out / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    md = render_markdown(leaderboard, run_meta)
    (out / "leaderboard.md").write_text(md)
    if verbose:
        print("\n" + md, file=sys.stderr)
        print(f"\nResults written to {out}", file=sys.stderr)
    return {"run_id": run_id, "out": str(out), "leaderboard": leaderboard}


def _serialize(obj: Any) -> Any:
    """Trajectories are already plain dicts, but rationale/usage may carry dataclasses."""
    return asdict(obj)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the SynthFin trading benchmark.")
    ap.add_argument("config", help="path to a benchmark YAML config")
    ap.add_argument("--corpus", help="override corpus_path from the config")
    ap.add_argument("--run-id", default="", help="name this run's output folder")
    ap.add_argument("--scenarios", nargs="*", help="override scenario selection (slugs/ids)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = BenchConfig.from_yaml(args.config)
    if args.corpus:
        cfg.corpus_path = args.corpus
    if args.scenarios is not None:
        cfg.scenarios = args.scenarios
    run_benchmark(cfg, run_id=args.run_id, verbose=not args.quiet)


if __name__ == "__main__":
    main()
