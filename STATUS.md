# Research status

This repository records the state of the project on **19 August 2026**. It is a research release, not a declaration of submission readiness.

## Evidence map

| Component | Current state | What the evidence licenses |
|---|---|---|
| Proposition A: signature degeneracy | proved and numerically checked | The negative index is identically zero and the signature retains only the number of non-trivial rotation planes. |
| Proposition B: endpoint-path factorisation | proved and numerically checked | The path residue is a holonomy fixing the initial state; endpoint and path data are mathematically independent. |
| Theorem C: intact-window law | proved under explicit random-oracle and homogeneous-marking hypotheses | Edit arrangement, not retention rate alone, controls the expected detector residual. |
| Detector implementations | self-calibrated on synthetic nulls | The synthetic experiments can be interpreted at face value within the certificate's limits. |
| Real rewriting chains | expanded CPU study: 90 passages, 6 chains, 540 attacked texts | The preregistered classical test is supported in 2/3 schemes, with important clustering, collinearity, scale, and attack-family limitations. |

## The expanded real-text result

The expanded run contains 30 base passages for each of three watermark schemes. Every passage follows six translation chains, for **540 attacked texts**; 538 have finite residual statistics and enter the regressions. Of the 540 attacked texts, 464 remain detected above `z = 4`.

The criterion fixed in `Plan_v1.md` requires the partial coefficient of holonomy energy to be negative and significant at `0.01/3` in at least two of the three schemes. With classical ordinary-least-squares errors, KGW and Unigram pass and EXP does not, so the immutable analysis returns `SUPPORTED`:

| Scheme | Holonomy coefficient | Classical p-value | Clustered p-value | Passage-fixed p-value |
|---|---:|---:|---:|---:|
| KGW | -0.070 | 1.7e-3 | 3.5e-2 | 2.5e-2 |
| Unigram | -0.145 | 4.0e-9 | 8.2e-4 | 2.1e-4 |
| EXP | -0.014 | 7.4e-1 | 7.7e-1 | 1.3e-1 |

The six chains of a scheme share the same 30 base passages, so classical errors are optimistic. Under the stricter clustered and passage-fixed checks, only Unigram clears the Bonferroni threshold. Moreover, semantic deficit and holonomy energy are strongly collinear in these translation chains (`r = 0.85` to `0.91`). The appropriate reading is therefore: the preregistered criterion is formally met, the sign is stable, and the evidence is promising, but the result is not yet a robust causal or cross-attack claim. See [`run_exp_analysis_v1_20260819T163238`](results/recorded/run_exp_analysis_v1_20260819T163238/).

An earlier 18-chain pilot remains in the repository as historical evidence and correctly returns `UNDETERMINED`; it has not been deleted or rewritten.

## Why this is not yet submission-ready

The workspace publication gate has not been met. In particular:

1. no internal referee report or external-readiness audit exists yet;
2. the real-text study still uses only 30 base passages per scheme, a 0.5B generator, and round-trip translation;
3. embedding dependence must be tested with at least three multilingual encoders;
4. deliberately constructed matched-endpoint detours are needed to break the high collinearity between semantic deficit and holonomy energy;
5. the detector implementations still require cross-validation against reference repositories or MarkLLM;
6. the aligned detector of the published Kuditipudi scheme is not implemented;
7. the priority literature queue must be read in full before final novelty claims are made;
8. ROC curves and true-positive rates at fixed false-positive rates are still missing;
9. the generation audit found that 87 of 90 stored completions regenerate exactly on the same machine, while two repeatedly diverge at fixed token boundaries. The stored corpus is hash-stable, but generation is not bit-exact across processes.

## Next empirical gate

The next run should test whether the signal survives a larger, less collinear, multi-attack design. The target is:

- at least 500 completions per scheme from a 3-8B instruction model;
- chain lengths from one to five;
- round-trip translation, learned paraphrase, controlled substitution, and deliberately constructed detours;
- three multilingual embedding models;
- cluster-robust or prompt-level resampling to account for repeated chains from the same source;
- the preregistered regression of residual detector statistic on retention, semantic deficit, and holonomy energy;
- complete manifests and immutable outputs for every run.

Until that gate is passed, the strongest supported contribution is the combination of Propositions A and B with the intact-window law and its arrangement corollary.

## Automation status

The repository includes a GitHub Actions matrix for Python 3.11 and 3.12. On the initial push, GitHub refused to allocate a runner and annotated the job with `The job was not started because your account is locked due to a billing issue.` No remote test step ran. The identical core suite passes locally: **5 tests passed** on Python 3.11.9.

The workflow remains available through pull requests and manual dispatch. Once the GitHub account-level billing lock is resolved, it should be dispatched again to obtain independent remote confirmation.
