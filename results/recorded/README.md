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
| `run_exp_corpus_v1_20260819T154350` | final merged 90-passage, six-chain corpus | log and manifest; generated text omitted |
| `run_exp_indicators_v1_20260819T161524` | 540 geometric and detector measurements | derived output, log, manifest |
| `run_exp_analysis_v1_20260819T163238` | preregistered Result D test and robustness analyses | output, log, manifest; verdict `SUPPORTED` in 2/3 classical tests |
| `run_exp_corpus_v1_20260819T174004` | targeted regeneration audit | audit output, log, manifest; generated completions omitted |

The three certificate-supporting runs match the published experiment-script hashes in their manifests. The latest indicator and analysis scripts also match their expanded-run manifests byte for byte; the current corpus script matches the later audit manifest and extends the full-corpus source only with audit controls. Earlier pilot runs remain historical records and are not rewritten.

The `SUPPORTED` verdict refers strictly to the preregistered ordinary-least-squares criterion. The analysis manifest also records that only Unigram remains below the corrected threshold with cluster-robust or passage-fixed inference. Read [`STATUS.md`](../../STATUS.md) before citing the result.
