# Certificate: The intact-window law and the context-width decay, v1

## Statement

Consider a green-list watermark with green fraction `gamma` in `(0,1)`,
context width `h >= 0` and logit bias `delta_b > 0`, detected on a token
sequence `y_0, ..., y_{T-1}` by

```
    z = ( G - gamma T' ) / sqrt( T' gamma (1 - gamma) ) ,     T' := T - h ,
```

where `G` counts the scored positions `t = h, ..., T-1` at which `y_t`
belongs to the green list seeded by `(y_{t-h}, ..., y_{t-1})`.

Let an attack replace the tokens at a set `E` of positions, leaving the
length unchanged, and define the **intact-window set**

```
    I := { t : h <= t < T ,  [t - h, t] intersect E = empty } .
```

**(C0) (Intact-window law.)** Under hypotheses (H1)-(H2) below,

```
    E[ z_att ] = ( |I| / T' ) . E[ z_0 ] .
```

Residual detectability is exactly proportional to the number of scored
positions whose entire seeding window survived. It is a functional of
the *pattern* of the edits, not of their number.

**(C1) (Context-width decay law.)** If in addition each position is
retained independently with probability `rho` (hypothesis (H3)), then
`E[|I|] = rho^(h+1) T'` and

```
    E[ z_att ] = rho^(h+1) . E[ z_0 ] .
```

The signal decays exponentially in the context width and only linearly
in the retention rate.

**(C2) (Detectable length.)** To reach a fixed threshold `z*` the
attacked text must satisfy

```
    T' >= ( z* )^2 gamma (1 - gamma) / ( rho^(h+1) (gamma_w - gamma) )^2 ,
```

so the length needed for reliable detection grows like `rho^(-2(h+1))`.

**(C3) (Arrangement dominates the rate.)** Fix the retention rate
`rho = 1 - |E|/T`. Then

- if `E` is a single contiguous run,
  `|I| >= T' - |E| - h`, so the ratio is at least `rho - h/T'`,
  *independently of* `h`;
- if `E` consists of `b` maximal contiguous runs,
  `|I| >= T' - |E| - h b`;
- if `E` contains an arithmetic progression of step `k <= h + 1`
  covering `[0, T)`, then `I` is empty and the expected residual signal
  is exactly zero.

Consequently, whenever `1 - rho >= 1/(h+1)`, the residual ratio ranges
over essentially the whole interval `[0, rho]` as the arrangement of `E`
varies at fixed `rho`. **The retention rate does not determine the
residual signal.**

**(C4) (Exponential family.)** For the distortion-free scheme that
seeds a uniform vector `xi` from the same context and scores
`S = sum_t -log(1 - xi_{y_t})`, whose null mean is `T'`, the same
argument gives

```
    E[ S_att ] - T' = ( |I| / T' ) ( E[ S_0 ] - T' ) ,
```

and hence the analogues of (C1)-(C3). The law is a property of context
seeding, not of the biasing mechanism.

## Notation

`gamma_w` is the probability that a scored position of unattacked
watermarked text is green; `gamma_w > gamma` is what watermarking
achieves and depends on `delta_b` and on the entropy of the generator.
`z_0` is the detector statistic before the attack.

## Dependencies

- Kirchenbauer et al., *A Watermark for Large Language Models*,
  ICML 2023, for the scheme and the `z` statistic.
- Zhao et al., ICLR 2024, for the case `h = 0`.
- No result from this project.

## Proof

*Hypotheses.*

**(H1) Random-oracle hashing.** The green indicator
`g(s, w) = 1{ hash(s, w) < gamma }` behaves, across distinct pairs
`(s, w)` of seed and token, as an i.i.d. Bernoulli(`gamma`) family.

**(H2) Homogeneous marking.** At each scored position of the
unattacked text the green indicator is Bernoulli(`gamma_w`),
independently across positions.

**(H3) Independent retention.** Positions are retained independently
with probability `rho`, and replacement tokens are chosen independently
of the key.

*(C0).* Fix a scored position `t`. Two cases.

If `t` lies in `I`, then `y_t` and its whole seeding window
`(y_{t-h}, ..., y_{t-1})` are the ones the generator produced. The pair
`(s_t, y_t)` evaluated by the detector is therefore the very pair the
generator evaluated, and by (H2) it is green with probability
`gamma_w`.

If `t` does not lie in `I`, then either the seeding window or the
scored token differs from what the generator produced, so the pair
`(s_t, y_t)` is one the generator never biased towards green. By (H1)
its green indicator is a fresh Bernoulli(`gamma`).

Summing,

```
    E[G_att] = |I| gamma_w + (T' - |I|) gamma
             = gamma T' + |I| ( gamma_w - gamma ) ,
```

hence

```
    E[z_att] = |I| ( gamma_w - gamma ) / sqrt( T' gamma (1 - gamma) ) .
```

The same computation with `|I| = T'` gives
`E[z_0] = T'(gamma_w - gamma)/sqrt(T' gamma(1-gamma))`, and dividing
yields (C0).

*(C1).* Under (H3) the event `t in I` is the intersection of the `h+1`
independent retention events for the positions `t-h, ..., t`, so
`P(t in I) = rho^(h+1)` and `E|I| = rho^(h+1) T'`. Substituting into
(C0), and using that (C0) is linear in `|I|`, gives (C1).

*(C2).* Solve `E[z_att] >= z*` for `T'` using (C1) and
`E[z_0] = sqrt(T')(gamma_w - gamma)/sqrt(gamma(1-gamma))`.

*(C3).* A scored position leaves `I` only if its window
`[t-h, t]`, of `h+1` consecutive positions, meets `E`. If `E` is a
single run of length `|E|`, the windows meeting it are those with
`t` in `[min E, max E + h]`, at most `|E| + h` of them; the first
bullet follows, and with `b` runs the bound is additive. If `E`
contains every `k`-th position with `k <= h+1`, then every window of
`h+1` consecutive positions contains a member of `E`, so `I` is empty
and `E[z_att] = 0` by (C0). The final claim follows by comparing the
extremes, using that `|E| = (1-rho)T`. []

## Verification of hypotheses

- **(H1)** is a modelling idealisation of the concrete hash. The
  implementation uses splitmix64 on the pair `(seed, token)`; the
  detector recomputes membership from the context alone, so there is no
  leakage between the two sides. Deviations from ideality would show up
  as a mismatch between the measured green fraction of *unwatermarked*
  text and `gamma`; the run logs this.
- **(H2)** replaces a position-dependent green probability by its
  average. It is a mean-field step and is the weakest of the three: in
  real text the entropy, hence `gamma_w`, varies strongly with content.
  This is why the synthetic generator of the supporting run fixes the
  entropy by construction, and why the corpus experiment is reported
  separately rather than being used to test this theorem.
- **(H3)** is used only for (C1). (C0) and (C3) are free of it, which
  is the point of stating (C0) first: the general law is deterministic
  in `|I|` and the i.i.d. assumption enters only to compute `E|I|`.
- **Length preservation.** Substitution keeps `T` fixed. Insertions and
  deletions shift the window and are outside the model; see
  limitations.

## Computational support

Run `run_exp_decay_v1_20260819T135047` under `7. Results/Article_LLW/`,
produced by `2. Scripts/Article_LLW/exp_decay_v1.py`. Synthetic
generator, vocabulary 4000, `T = 400`, 120 sequences per condition,
exact detectors from `wm_schemes.py`.

**(C1) under i.i.d. edits.** Mean absolute error between the observed
ratio and `rho^(h+1)`, over `rho` in `{0.95, ..., 0.5}`:

| scheme | `h` | mean abs. error |
|---|---|---|
| KGW | 0 | 0.0023 |
| KGW | 1 | 0.0023 |
| KGW | 2 | 0.0022 |
| KGW | 3 | 0.0021 |
| EXP | 1 | 0.0011 |

The exponent is confirmed across two decades of the ratio: at
`rho = 0.5` the observed ratios are `0.499, 0.247, 0.125, 0.065` for
`h = 0, 1, 2, 3`, against predictions `0.500, 0.250, 0.125, 0.063`.

**(C3), the headline check.** KGW with `h = 1`, at *identical*
retention rate, the observed residual ratio depends only on the
arrangement of the edits:

| `rho` | i.i.d. | contiguous block | periodic |
|---|---|---|---|
| 0.95 | 0.904 | 0.947 | 0.903 |
| 0.90 | 0.806 | 0.897 | 0.802 |
| 0.80 | 0.637 | 0.795 | 0.595 |
| 0.70 | 0.492 | 0.697 | 0.399 |
| 0.60 | 0.362 | 0.594 | 0.196 |
| 0.50 | 0.247 | 0.488 | **-0.005** |

At `rho = 0.5` the same fraction of tokens survives in all three
columns, and the residual signal is half the original, a quarter of it,
or exactly nothing. The periodic column at `rho = 0.5` is the case
`k = 2 <= h + 1` of (C3), where `I` is provably empty; the measured
value is zero to within the noise of 120 sequences.

The intact-window law (C0) predicts these numbers quantitatively and
not merely qualitatively. Counting `|I|` directly from the edit pattern,
for `T = 400` and `h = 1`, gives

| `rho` | block: predicted / observed | periodic: predicted / observed |
|---|---|---|
| 0.95 | 0.9474 / 0.9470 | 0.9023 / 0.9028 |
| 0.90 | 0.8972 / 0.8967 | 0.8020 / 0.8016 |
| 0.80 | 0.7970 / 0.7947 | 0.6015 / 0.5951 |
| 0.70 | 0.6967 / 0.6973 | 0.4010 / 0.3988 |
| 0.60 | 0.5965 / 0.5943 | 0.2005 / 0.1962 |
| 0.50 | 0.4962 / 0.4880 | 0.0000 / -0.0048 |

The largest discrepancy is 0.008 and the typical one 0.003, over
120 sequences per cell. This is the agreement one expects if (C0) is an
identity rather than an approximation; note in particular that the
periodic pattern at `rho = 0.5` is predicted to give *exactly* zero, and
does.

## Failure modes / limitations

- **Insertions and deletions.** These shift positions and break the
  alignment between the generated and the observed index sets. The
  correct generalisation replaces `I` by the set of positions whose
  window survives *as a contiguous block in the attacked text*, which
  is what the longest-common-subsequence estimator in `loop_geometry.py`
  approximates. The identity (C0) then holds only approximately, and
  the article must not claim otherwise.
- **(H2) in real text.** Low-entropy passages carry almost no mark, so
  a real attack that happens to preserve low-entropy spans and rewrite
  high-entropy ones destroys more signal than `|I|` alone suggests.
  This makes (C0) an *upper* bound on residual detectability in the
  wild, and the corpus experiment is expected to fall below it.
- **Adaptive attackers.** (C3) shows an attacker who knows `h` can
  achieve zero residual signal at retention rate `1 - 1/(h+1)`, that is
  by editing only one token in `h+1`. For the common choice `h = 1`
  this is one token in two. The article should state this plainly: it
  is a design consequence, not a novel attack, but it has not to our
  knowledge been written down in this exact form.
- The result says nothing about *undisclosed* production schemes.

## Public-manuscript status

Included in the main draft, Section 5. (C0) is stated as the theorem
and (C1) as its corollary, reversing the order of discovery; the
arrangement statement (C3) is given its own subsection because it is
the surface-channel counterpart of the geometric path-dependence claim,
and it is the one a security referee will care about most.

## Changelog

- **v1** (2026-08-19). First issue. The planned statement was the
  `rho^(h+1)` law alone; the supporting run showed that the correct and
  stronger statement is the intact-window identity (C0), of which the
  decay law is the i.i.d. corollary. (C3) was added after the
  arrangement probe returned a residual ratio of zero at a retention
  rate of one half.
