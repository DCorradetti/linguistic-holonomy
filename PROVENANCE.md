# Provenance

This public repository was assembled from the `Article_LLW` research workspace on 19 August 2026 and updated later that day with the completed 540-chain experiment.

## Copied without substantive changes

- `paper/Article_LLW_v1.tex` from `1. Drafts/Article_LLW/Article_LLW_v1.tex`;
- all four files in `certificates/` from `3. Certificates/Article_LLW/`;
- the experiment and computational modules in `scripts/` from `2. Scripts/Article_LLW/`;
- the successful certificate-supporting outputs in `results/recorded/` from `7. Results/Article_LLW/`.

The article and certificates remain versioned research artifacts. Corrections belong in a new manuscript or certificate version, not as silent edits to these copies.

## Public-layout adaptation

The only executable layout change is in `scripts/runio.py`: fresh runs are written to `results/runs/` instead of the workspace-specific `7. Results/Article_LLW/`. The mathematical and statistical logic is unchanged. The updated `exp_corpus_v1.py`, `exp_indicators_v1.py`, and `exp_analysis_v1.py` are byte-for-byte copies of the sources recorded by their latest manifests.

`paper/Article_LLW_v2.tex` is the results-complete public snapshot of the current workspace manuscript. The original public `Article_LLW_v1.tex` and its PDF remain untouched. The v2 header was relabelled to identify this public results update, and one reproducibility sentence was clarified to state that generated passages are represented publicly by hashes and manifests rather than redistributed.

The top-level README, status page, test suite, dependency lists, GitHub workflow, citation metadata, and repository artwork were created for this public release.

## Recorded-run rule

Every recorded run contains a `manifest.json` with the script hash, command, environment, output hashes, conclusions, and limitations. A directory without a manifest supports no claim. The public release includes only completed, claim-relevant runs; aborted workspace runs are intentionally omitted.

The generated text corpus and stage-one completions are not redistributed. Their manifests and logs document the exact hashes, commands, merge chain, and regeneration audit. The derived indicator file and complete analysis output are included because they are the smallest public artifacts from which the statistical claims can be inspected without releasing generated passages.

The expanded public evidence chain is:

1. `run_exp_corpus_v1_20260819T154350` — final merged corpus manifest and log; generated texts omitted;
2. `run_exp_indicators_v1_20260819T161524` — 540 derived chain measurements;
3. `run_exp_analysis_v1_20260819T163238` — preregistered verdict and robustness checks;
4. `run_exp_corpus_v1_20260819T174004` — targeted cross-process regeneration audit.

## Artwork

`assets/linguistic-holonomy-hero.png` was generated for this repository with OpenAI's image-generation system from an original prompt describing a geodesic linguistic loop, a holonomy marker, and a dissolving statistical watermark lattice. It contains no embedded text, logo, or third-party trademark.
