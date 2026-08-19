# Recorded runs

These directories are immutable copies of completed runs from the research workspace.

| Run | Supports | Public contents |
|---|---|---|
| `run_exp_geometry_v1_20260819T135822` | Propositions A and B; Gauss-Bonnet check | output, log, manifest |
| `run_exp_decay_v1_20260819T135047` | intact-window and context-width decay laws | output, log, manifest |
| `run_exp_detector_validation_v1_20260819T141737` | detector calibration certificate | output, log, manifest |
| `run_exp_corpus_v1_20260819T122428` | reduced real-text pilot | log and manifest; generated text omitted |
| `run_exp_indicators_v1_20260819T143755` | geometric indicators for the pilot | derived output, log, manifest |
| `run_exp_analysis_v1_20260819T145755` | verified rerun of the pilot analysis | output, log, manifest; verdict `UNDETERMINED` |

The three certificate-supporting runs match the published experiment-script hashes in their manifests. The corpus and indicator stages are preserved for transparency; later edits to those two scripts mean their archived manifests must not be represented as exact regeneration records for the current source. The analysis was rerun with the published current source and correctly returns `UNDETERMINED`. A new adequately powered pilot must create new run directories rather than overwrite these.
