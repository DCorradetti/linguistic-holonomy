# Certificate: Null calibration of the implemented detectors, v1

## Statement

The three detectors implemented in `2. Scripts/Article_LLW/wm_schemes.py`
are correctly calibrated, in the following four senses.

**(N1)** On text not produced with the key, the green fraction equals
`gamma` and, for context width `h >= 1`, the `z`-statistic is standard
normal.

**(N1b)** For `h = 0` the null is *key-dependent by construction*: the
green list is fixed for the whole text, so the statistic is centred not
on `gamma` but on the fraction of the vocabulary that the particular key
colours green. The resulting displacement of the `z`-score has standard
deviation

```
    sqrt( T / |V| )
```

across keys. This is a property of the scheme, not of the
implementation, and it vanishes for `h >= 1` because a fresh list is
drawn at every position.

**(N2)** Watermarked text scored with a key other than the one that
produced it is statistically indistinguishable from unwatermarked text.

**(N3)** The same text scored with the correct key is detected.

**(N4)** For the exponential family, the Gamma p-values are uniform on
`(0,1)` under the null.

## Notation

`gamma` is the green fraction, `T` the token count, `|V|` the vocabulary
size, `h` the context width. `z` is the statistic of Eq. (5.1) of the
draft.

## Dependencies

- `wm_schemes.py` v1: the splitmix64 hash, the context seeding, the
  green-list and Gamma detectors.
- No external reference implementation. The schemes are re-implemented
  rather than imported, so there are no published numbers to reproduce;
  self-calibration is what is available and what a referee needs.

## Proof

The claims are empirical and are established by the supporting run.
Only (N1b) has an argument, which we give because it is the one result
here that is not a check but a small theorem.

Under a random-oracle hash, the set of green tokens for a fixed seed is
obtained by including each of the `|V|` vocabulary items independently
with probability `gamma`. The realised green fraction is therefore

```
    gamma_hat = gamma + O_P( sqrt( gamma (1 - gamma) / |V| ) ) .
```

For `h = 0` the same list serves every position, so on unwatermarked
text whose tokens are spread over the vocabulary the expected green
count is `T gamma_hat` rather than `T gamma`. Substituting into

```
    z = ( G - gamma T ) / sqrt( T gamma (1 - gamma) )
```

gives a displacement

```
    E[z] = T ( gamma_hat - gamma ) / sqrt( T gamma (1 - gamma) )
         = O_P( sqrt( T / |V| ) ) ,
```

the `gamma(1-gamma)` factors cancelling. For `h >= 1` the seed changes
at every position, so the realised fractions are independent across
positions and average out at rate `1/sqrt(T)`, leaving no displacement.
[]

## Verification of hypotheses

Nothing is assumed beyond the definitions of the schemes. (N1) is
itself the empirical test of hypothesis (H1) of
`Certificate_context_decay_v1`: were the hash not behaving as a random
oracle, the green fraction of unwatermarked text would drift away from
`gamma`, and it does not.

## Computational support

Run `run_exp_detector_validation_v1_20260819T141737` under
`7. Results/Article_LLW/`. Vocabulary 4000, `T = 400`, 400 null streams,
120 watermarked texts per scheme.

**(N1) Unwatermarked streams.**

| `h` | green fraction | mean `z` | sd `z` | KS vs `N(0,1)` (p) | FPR at `z > 4` |
|---|---|---|---|---|---|
| 0 | 0.2539 | +0.181 | 1.064 | 0.000 | 0.0000 |
| 1 | 0.2475 | -0.115 | 0.989 | 0.039 | 0.0000 |
| 2 | 0.2508 | +0.037 | 0.971 | 0.074 | 0.0000 |
| 3 | 0.2498 | -0.011 | 1.033 | 0.152 | 0.0000 |

The green fraction is within 0.004 of `gamma` in every case. For
`h >= 1` the statistic is standard normal. The row `h = 0` fails a
Kolmogorov-Smirnov test against `N(0,1)`, and correctly so: it is being
tested against the wrong hypothesis, as (N1b) explains.

**(N1b) The key-dependent offset.** Over 24 independent keys, the mean
offset is `+0.073` and its standard deviation is `0.3279`, against the
predicted `sqrt(T/|V|) = 0.3162`. The ratio of observed to predicted is
`1.04`. For the corpus experiment, where `T = 180` and
`|V| = 151665`, the same quantity is `0.0345` and is negligible.

**(N2) and (N3).**

| scheme | `h` | mean `z`, right key | detection at `z > 4` | mean `z`, wrong key | false alarms |
|---|---|---|---|---|---|
| KGW | 1 | 20.77 | 100.0% | -0.030 (sd 0.975) | 0.0% |
| UNIGRAM | 0 | 20.93 | 100.0% | +0.027 (sd 1.006) | 0.0% |
| EXP | 1 | 37.05 | 100.0% | -0.293 (sd 1.099) | 0.0% |

**(N4)** Kolmogorov-Smirnov statistic of the exponential-family
p-values against `Uniform(0,1)`: `0.0443`, `p = 0.401`, mean p-value
`0.4990`.

## Failure modes / limitations

- **Self-calibration only.** A cross-check against `MarkLLM` or against
  the reference repositories of [KGW23] and [Kud24] has not been
  performed, and remains an open item in `journal_targets.md`.
- **The easiest null.** Null streams are uniform over the vocabulary.
  Natural text is bursty and repetitive, and a deployed detector must
  be calibrated against that, not against this.
- **The EXP detector is not the detector of [Kud24].** Ours is
  alignment-free and context-seeded; theirs uses an edit-distance
  alignment. Every claim about the exponential family in this article
  is a claim about the context-seeded variant, and the draft says so.
- The `h = 0` offset was discovered by this run rather than anticipated.
  It has no bearing on the corpus experiment, where it is a fortieth of
  the effect sizes involved, but it does mean that a Unigram-style
  scheme deployed with a small vocabulary would need its null
  calibrated per key.

## Public-manuscript status

Summarised in the reproducibility section of the main draft, with the
(N2)/(N3) table included. The derivation in (N1b) belongs in a numbered
remark: it is short, it is not in the sources we consulted, and it is
the kind of detail a referee of a forensics journal will want to see
that the authors noticed.

## Changelog

- **v1** (2026-08-19). First issue. The `h = 0` null offset was found
  when the run failed its own normality check; the check was wrong, not
  the code, and the offset is now derived and measured.
