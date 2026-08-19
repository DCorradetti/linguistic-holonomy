# Reproducibility scripts

The scripts support *Linguistic Holonomy and the Erosion of Statistical Watermarks*. Each experiment creates a fresh timestamped directory under `results/runs/` containing an output JSON file, `log.txt`, and `manifest.json`. Recorded runs are never edited after completion.

## Modules

| File | Role |
|---|---|
| `loop_geometry.py` | Minimal rotations, loop rotation, holonomy, angle spectra, and retention rates. |
| `wm_schemes.py` | Green-list, Unigram, and context-seeded exponential schemes with exact detectors. |
| `runio.py` | Immutable run directories, hashes, environment capture, and manifests. |

## Experiments

Run from the repository root:

```bash
python scripts/exp_geometry_v1.py
python scripts/exp_decay_v1.py
python scripts/exp_detector_validation_v1.py
python scripts/exp_corpus_v1.py --n-prompts 30 --max-new-tokens 180
python scripts/exp_indicators_v1.py
python scripts/exp_analysis_v1.py
```

The first three use only the core dependencies. The final three require the full environment and download model weights on first use. `exp_indicators_v1.py` and `exp_analysis_v1.py` automatically select the newest completed upstream run, or accept explicit paths through `--corpus` and `--indicators`.

## Recorded environment

- Python 3.11
- NumPy 2.4.6
- SciPy 1.16.1
- PyTorch 2.8.0+cpu
- Transformers 4.57.6
- SentencePiece and Sacremoses for Marian tokenizers

See each recorded manifest for the exact command, hashes, conclusions, and known limitations.
