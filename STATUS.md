# Research status

This repository records the state of the project on **19 August 2026**. It is a research release, not a declaration of submission readiness.

## Evidence map

| Component | Current state | What the evidence licenses |
|---|---|---|
| Proposition A: signature degeneracy | proved and numerically checked | The negative index is identically zero and the signature retains only the number of non-trivial rotation planes. |
| Proposition B: endpoint-path factorisation | proved and numerically checked | The path residue is a holonomy fixing the initial state; endpoint and path data are mathematically independent. |
| Theorem C: intact-window law | proved under explicit random-oracle and homogeneous-marking hypotheses | Edit arrangement, not retention rate alone, controls the expected detector residual. |
| Detector implementations | self-calibrated on synthetic nulls | The synthetic experiments can be interpreted at face value within the certificate's limits. |
| Real rewriting chains | reduced CPU pilot | The pilot demonstrates the pipeline, but it does not establish the preregistered predictive claim. |

## The pilot verdict

An early analysis labelled Result D `REFUTED`. That label was not scientifically operative: after the detection filter, only 12 of 18 chains remained, and **zero schemes had enough observations to run the registered regression**. The published current analysis was rerun on the recorded indicators and correctly returns `UNDETERMINED`; its immutable output is [`run_exp_analysis_v1_20260819T145755`](results/recorded/run_exp_analysis_v1_20260819T145755/).

Accordingly, this repository makes neither a positive nor a negative empirical claim about holonomy energy as an independent predictor on real text. It publishes the pilot because a transparent negative or underpowered run is more useful than an invisible one.

## Why this is not yet submission-ready

The workspace publication gate has not been met. In particular:

1. the manuscript v1 still contains a placeholder for the real-transformation analysis;
2. no internal referee report or external-readiness audit exists yet;
3. the real-text study needs substantially more observations, learned paraphrasers, and a 3-8B generator;
4. embedding dependence must be tested with at least three multilingual encoders;
5. the detector implementations still require cross-validation against reference repositories or MarkLLM;
6. the aligned detector of the published Kuditipudi scheme is not implemented;
7. the priority literature queue must be read in full before final novelty claims are made;
8. ROC curves and true-positive rates at fixed false-positive rates are still missing.

## Next empirical gate

The next run should contain enough independent texts per scheme to fit the registered model after filtering. The target design is:

- at least 500 completions per scheme from a 3-8B instruction model;
- chain lengths from one to five;
- round-trip translation, learned paraphrase, controlled substitution, and deliberately constructed detours;
- three multilingual embedding models;
- cluster-robust or prompt-level resampling to account for repeated chains from the same source;
- the preregistered regression of residual detector statistic on retention, semantic deficit, and holonomy energy;
- complete manifests and immutable outputs for every run.

Until that gate is passed, the strongest supported contribution is the combination of Propositions A and B with the intact-window law and its arrangement corollary.
