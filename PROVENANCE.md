# Provenance

This public repository was assembled from the `Article_LLW` research workspace on 19 August 2026.

## Copied without substantive changes

- `paper/Article_LLW_v1.tex` from `1. Drafts/Article_LLW/Article_LLW_v1.tex`;
- all four files in `certificates/` from `3. Certificates/Article_LLW/`;
- the experiment and computational modules in `scripts/` from `2. Scripts/Article_LLW/`;
- the successful certificate-supporting outputs in `results/recorded/` from `7. Results/Article_LLW/`.

The article and certificates remain versioned research artifacts. Corrections belong in a new manuscript or certificate version, not as silent edits to these copies.

## Public-layout adaptation

The only executable layout change is in `scripts/runio.py`: fresh runs are written to `results/runs/` instead of the workspace-specific `7. Results/Article_LLW/`. The mathematical and statistical logic is unchanged.

The top-level README, status page, test suite, dependency lists, GitHub workflow, citation metadata, and repository artwork were created for this public release.

## Recorded-run rule

Every recorded run contains a `manifest.json` with the script hash, command, environment, output hashes, conclusions, and limitations. A directory without a manifest supports no claim. The public release includes only completed, claim-relevant runs; aborted workspace runs are intentionally omitted.

The generated text corpus is not redistributed. Its recorded manifest and log are sufficient to document its hashes and regeneration command, in accordance with the project's release plan. The current analysis code was rerun during packaging and produced a new immutable `UNDETERMINED` result because no scheme met the minimum sample size.

## Artwork

`assets/linguistic-holonomy-hero.png` was generated for this repository with OpenAI's image-generation system from an original prompt describing a geodesic linguistic loop, a holonomy marker, and a dissolving statistical watermark lattice. It contains no embedded text, logo, or third-party trademark.
