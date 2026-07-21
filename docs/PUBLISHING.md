# Publishing the benchmark

How to release SynthFin Trading Bench so others can trust, reproduce, and submit to it. This mirrors
what strong 2025-era benchmarks do (contamination controls, frozen versioned data, public scorer,
baselines, a leaderboard, and a technical report).

## The three artifacts

A benchmark is a **task + dataset + leaderboard**, published as three linked artifacts:

1. **Code repository (this repo)** — the harness, scorer, baselines, configs, docs, and tests.
   Host on GitHub under Apache-2.0. This is the source of truth for *how* models are evaluated.
2. **Dataset** — the frozen corpus, hosted where large data belongs and versioned by content hash.
   **Recommended: a Hugging Face Dataset** (`daviddata1/synthfin-trading-bench-1000x10`) — it gives you
   versioned revisions, a rendered data card, a viewer, and `datasets.load_dataset(...)` access.
   Alternatives: a release asset / object storage with a published SHA-256. Never commit the corpus
   into the code repo.
3. **Leaderboard + report** — a human-readable results page and a short technical report (arXiv or a
   PDF in `docs/`) describing the task, contamination argument, baselines, and headline numbers.
   The leaderboard can be a static page generated from `results/<run>/leaderboard.md`, or a Hugging
   Face Space that renders submissions.

## Release checklist

- [ ] **Freeze the corpus.** Generate the final 1000×10, run `scripts/hash_corpus.py`, and record
      the hash in `DATA_CARD.md` and the report. This is `v1`.
- [ ] **Pin model snapshots.** Put exact model ids + decoding params in `configs/full_1000x10.yaml`.
- [ ] **Run baselines + reference models.** Produce `results/v1/` and commit `leaderboard.md` (not
      the raw trajectories — those can be a release asset).
- [ ] **Write the report.** Task, protocol, metrics (link METHODOLOGY), contamination statement
      (link CONTAMINATION), baseline vs model numbers with cross-scenario variance.
- [ ] **License clearly.** Apache-2.0 for code; choose and state the data license (e.g. CC BY-NC 4.0)
      in `DATA_CARD.md`.
- [ ] **Publish the dataset** to Hugging Face with the data card; tag revision `v1`.
- [ ] **Stand up the leaderboard** (static page or HF Space) and link all three artifacts to each
      other.
- [ ] **Document submission** (below) so third parties can be added.

## Versioning

- **Dataset version = content hash.** Every leaderboard row cites the corpus hash it was produced on.
- **Semantic-ish tags:** `v1`, `v1.1` (same task, refreshed corpus), `v2` (task/metric change).
  Never silently mutate a released corpus — cut a new version so old numbers stay meaningful.
- **Anti-staleness:** if contamination is suspected, generate a fresh corpus (new seeds), publish it
  as `v1.1`, and re-run. Keep the old version for continuity.

## Accepting submissions

Two models, pick per how much you want to run yourself:

- **Self-run (recommended to start):** submitters open a PR adding an agent spec to a config; you run
  it on the frozen corpus and publish the row. Guarantees identical conditions; you control cost.
- **Bring-your-own-results:** submitters run the public harness on the published corpus and PR their
  `results/<run>/` (trajectories + scores). You verify by re-scoring their trajectories with the
  public scorer and spot-checking a scenario. The content hash in `run_meta.json` must match the
  released corpus, or the submission is rejected.

Require in every submission: model id + decoding params, the corpus hash, the harness commit, and the
full `results/` directory. Reject anything whose hash or config doesn't match the frozen benchmark.

## Reporting results (template)

> Model *X* (snapshot `x-2026-…`, temp 0) scored a mean Sharpe of **S** and mean appraisal ratio of
> **A** across the 10 scenarios of SynthFin Trading Bench v1 (corpus `sha256:…`, harness `abc1234`),
> vs equal-weight buy-and-hold (Sharpe **S₀**) and momentum (Sharpe **S₁**). Per-scenario variance
> and the full leaderboard are in `results/v1/`.

Always publish alongside the baselines and the cross-scenario standard deviation — a single averaged
number without a baseline or dispersion is not interpretable.
