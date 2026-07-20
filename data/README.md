# data/

Corpora are **not** committed to this repo (they are large and versioned separately — see
[`../docs/DATA_CARD.md`](../docs/DATA_CARD.md) and [`../docs/PUBLISHING.md`](../docs/PUBLISHING.md)).

Download a corpus here and point a config's `corpus_path` at it:

```bash
# the published benchmark corpus (see docs/DATA_CARD.md for the hosted dataset)
#   ./synthfin_1000x10/       # unzip the dataset here
# the small validation corpus
#   ./synthfin_50x10/
```

A valid corpus is any SynthFin export directory containing `index.json`, `structured/`, and
`unstructured/`. Verify one with:

```bash
python ../scripts/hash_corpus.py ./synthfin_1000x10
```

Everything under `data/synthfin_*/` is gitignored.
