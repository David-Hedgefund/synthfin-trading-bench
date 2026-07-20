#!/usr/bin/env python3
"""Upload a corpus to a (private) Hugging Face Dataset repo.

Auth uses the standard HF token resolution: the ``HF_TOKEN`` env var or a prior
``huggingface-cli login``. The script never takes a token on the command line.

    export HF_TOKEN=hf_...            # a token with write scope
    python scripts/upload_to_hf.py ./data/synthfin_1000x10 \
        --repo davidhf/synthfin-trading-bench-1000x10 --private

Start private; flip to public later with:  huggingface-cli repo settings <repo> --private false
(or the web UI). The dataset's content hash (scripts/hash_corpus.py) should be recorded as the
version in the repo card and cited by every leaderboard run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.corpus import Corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", help="local corpus directory (standard layout)")
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. org/name")
    ap.add_argument("--private", action="store_true", help="create/keep the repo private")
    ap.add_argument("--message", default="Add SynthFin Trading Bench corpus")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub first")

    corpus = Corpus(args.corpus_dir)
    chash = corpus.content_hash()
    print(f"corpus profile={corpus.profile} tickers={len(corpus.tickers)} "
          f"scenarios={len(corpus.scenarios())}")
    print(f"content_hash={chash}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    # write the content hash as the version tag inside the card
    print(f"uploading {args.corpus_dir} -> {args.repo} (private={args.private}) ...")
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.corpus_dir,
        # exclude local point-in-time index caches; they are rebuilt on load
        ignore_patterns=["**/.ptindex.json", "**/.DS_Store"],
    )
    print(f"done. dataset: https://huggingface.co/datasets/{args.repo}")
    print(f"record this version:  sha256:{chash}")


if __name__ == "__main__":
    main()
