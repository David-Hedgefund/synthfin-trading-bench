#!/usr/bin/env python3
"""Carve a smaller benchmark corpus out of a larger SynthFin export.

Given a source corpus (e.g. the 1000-ticker × 25-scenario run) this selects a subset of scenarios
and/or tickers and writes a new, self-contained corpus in the identical standard layout — a valid
`corpus_path` for the benchmark. Used to produce the published 1000×10 corpus from the 1000×25 run:

    python make_subset.py /data/synthfin_1000x25 ./data/synthfin_1000x10 \
        --seeds 70000-70009            # keep the 10 curated scenarios

To also cut the universe (only needed if the source has >1000 tickers):

    python make_subset.py SRC DST --seeds 70000-70009 --tickers-file top1000.txt
    python make_subset.py SRC DST --n-scenarios 10 --n-tickers 1000   # first-N fallback

The script copies only the selected scenarios' structured + unstructured trees, prunes per-ticker
folders when a ticker subset is given, and rewrites index.json. It never mutates the source.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_seeds(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="source corpus directory (must contain index.json)")
    ap.add_argument("dst", help="destination corpus directory (created)")
    ap.add_argument("--seeds", help="scenario seeds to keep, e.g. '70000-70009' or '70000,70003'")
    ap.add_argument("--slugs", nargs="*", help="scenario slugs/ids to keep (alternative to --seeds)")
    ap.add_argument("--n-scenarios", type=int, default=0, help="keep the first N scenarios")
    ap.add_argument("--tickers-file", help="newline-delimited list of tickers to keep")
    ap.add_argument("--n-tickers", type=int, default=0, help="keep the first N tickers")
    ap.add_argument("--dry-run", action="store_true", help="report the selection without copying")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    index = json.loads((src / "index.json").read_text())
    scenarios = index.get("scenarios", [])
    tickers = list(index.get("tickers", []))

    # --- select scenarios ---
    if args.seeds:
        seeds = parse_seeds(args.seeds)
        keep = [s for s in scenarios if s.get("seed") in seeds]
    elif args.slugs:
        want = set(args.slugs)
        keep = [s for s in scenarios if s.get("slug") in want or s.get("id") in want]
    elif args.n_scenarios:
        keep = scenarios[: args.n_scenarios]
    else:
        keep = scenarios
    if not keep:
        raise SystemExit("no scenarios selected — check --seeds/--slugs/--n-scenarios")

    # --- select tickers ---
    if args.tickers_file:
        wanted = [t.strip() for t in Path(args.tickers_file).read_text().split() if t.strip()]
        keep_tickers = [t for t in tickers if t in set(wanted)]
    elif args.n_tickers:
        keep_tickers = tickers[: args.n_tickers]
    else:
        keep_tickers = tickers
    ticker_set = set(keep_tickers)

    print(f"source:      {src}")
    print(f"scenarios:   {len(keep)}/{len(scenarios)}  seeds={sorted(s.get('seed') for s in keep)}")
    print(f"tickers:     {len(keep_tickers)}/{len(tickers)}")
    if args.dry_run:
        for s in keep:
            print("  keep", s.get("slug"))
        return

    dst.mkdir(parents=True, exist_ok=True)
    prune = len(keep_tickers) != len(tickers)

    for s in keep:
        for kind in ("structured_path", "unstructured_path"):
            rel = s.get(kind) or ""
            src_dir = src / rel
            dst_dir = dst / rel
            if not src_dir.exists():
                print(f"  !! missing {src_dir}")
                continue
            if prune and (src_dir / "tickers").exists():
                # structured: copy scenario-level files, then only selected ticker folders
                dst_dir.mkdir(parents=True, exist_ok=True)
                for f in src_dir.glob("*.json"):
                    shutil.copy2(f, dst_dir / f.name)
                (dst_dir / "tickers").mkdir(exist_ok=True)
                for tdir in (src_dir / "tickers").iterdir():
                    if tdir.is_dir() and tdir.name in ticker_set:
                        shutil.copytree(tdir, dst_dir / "tickers" / tdir.name, dirs_exist_ok=True)
            elif prune:
                # unstructured: copy only selected ticker subdirs (+ MARKET)
                dst_dir.mkdir(parents=True, exist_ok=True)
                for sub in src_dir.iterdir():
                    if sub.is_dir() and (sub.name in ticker_set or sub.name == "MARKET"):
                        shutil.copytree(sub, dst_dir / sub.name, dirs_exist_ok=True)
                    elif sub.is_file():
                        shutil.copy2(sub, dst_dir / sub.name)
            else:
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print(f"  copied {s.get('slug')}")

    # copy optional top-level config dir if present
    if (src / "config").exists():
        shutil.copytree(src / "config", dst / "config", dirs_exist_ok=True)

    new_index = dict(index)
    new_index["scenarios"] = keep
    new_index["tickers"] = keep_tickers
    new_index["scenario_count"] = len(keep)
    new_index["profile"] = f"{len(keep_tickers)}x{len(keep)}"
    new_index["derived_from"] = index.get("profile", "unknown")
    (dst / "index.json").write_text(json.dumps(new_index, indent=2))

    # a minimal README so the artifact is self-describing
    (dst / "README.txt").write_text(
        f"SynthFin benchmark corpus subset\n"
        f"profile: {new_index['profile']}  (derived from {new_index['derived_from']})\n"
        f"scenarios: {[s.get('slug') for s in keep]}\n"
    )
    print(f"\nwrote {dst}  (profile {new_index['profile']})")
    print("verify with:  python scripts/hash_corpus.py", dst)


if __name__ == "__main__":
    main()
