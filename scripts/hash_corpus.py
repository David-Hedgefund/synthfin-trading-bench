#!/usr/bin/env python3
"""Print the content hash of a corpus — the value you cite as the frozen dataset version."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.corpus import Corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_path")
    args = ap.parse_args()
    c = Corpus(args.corpus_path)
    print(f"profile: {c.profile}")
    print(f"tickers: {len(c.tickers)}")
    print(f"scenarios: {len(c.scenarios())}")
    print(f"content_hash: {c.content_hash()}")


if __name__ == "__main__":
    main()
