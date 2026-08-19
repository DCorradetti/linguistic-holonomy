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

For a long CPU run, the six transformation chains can be split and merged without regenerating the watermarked passages:

```bash
python scripts/exp_corpus_v1.py --n-prompts 30 --max-new-tokens 180 \
  --chains rt_de,rt_fr,rt_es
python scripts/exp_corpus_v1.py --n-prompts 30 --max-new-tokens 180 \
  --chains rt_de_fr,rt_de_de \
  --resume-stage1 "<previous run>/corpus_stage1.json" \
  --merge "<previous run>/corpus.json"
```

`--merge` accepts only completed runs carrying a manifest. `--resume-stage1` can be audited with `--verify-resume N`; `--audit-only` performs that check without translating. Targeted repeated audits are available through `--audit-uids` and `--audit-repeats`. The audit reports differences rather than overwriting or discarding the stored corpus.

## Expanded recorded result

The current release records 90 watermarked passages, six translation chains per passage, and 540 attacked texts. The preregistered ordinary-least-squares criterion is supported in KGW and Unigram but not EXP. Because the six chains reuse each base passage, `exp_analysis_v1.py` also reports cluster-robust and passage-fixed inference; only Unigram clears the corrected threshold under those stricter checks. The script additionally reports within-passage intact-window correlations, matched-semantic-deficit strata, and the collinearity between semantic deficit and holonomy energy.

The generated passages are intentionally omitted from the public repository. Their hashes and commands are in the corpus manifest; the 540 derived measurements and full analysis output are published under `results/recorded/`.

## Recorded environment

- Python 3.11
- NumPy 2.4.6
- SciPy 1.16.1
- PyTorch 2.8.0+cpu
- Transformers 4.57.6
- SentencePiece and Sacremoses for Marian tokenizers

See each recorded manifest for the exact command, hashes, conclusions, and known limitations.
