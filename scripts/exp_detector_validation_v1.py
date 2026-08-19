"""
exp_detector_validation_v1.py -- null calibration of the detectors.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, sub-goal S6; hypothesis (H1) of Theorem C
Version  : v1
Purpose  : Establish that the detectors implemented in `wm_schemes.py`
           are correctly calibrated under the null, so that every
           statistic reported elsewhere in the article can be read at
           face value.
Input    : none (pseudo-random token streams, fixed seeds)
Output   : run_exp_detector_validation_v1_<stamp>/validation.json

Four checks, each of which would fail loudly if the hash, the seeding or
the statistic were wrong.

  N1  Unwatermarked text.  The green fraction must equal gamma, the
      z-score must be standard normal, and the empirical false-positive
      rate at a nominal threshold must match the nominal rate.  This is
      a direct test of hypothesis (H1) of Theorem C: if the hash were
      not behaving as a random oracle, the green fraction would drift
      away from gamma.
  N2  Wrong key.  Watermarked text scored with a key other than the one
      that produced it must look exactly like unwatermarked text.  A
      failure here would mean the statistic responds to something other
      than the secret.
  N3  Right key.  The same text scored with the right key must be
      detected, which fixes the sign and scale of the effect.
  N4  Exponential family.  Under the null the Gamma p-values must be
      uniform on (0,1); we report a Kolmogorov-Smirnov statistic.

There are no published reference numbers to reproduce here, because the
schemes are re-implemented rather than imported.  Self-calibration is
what an external referee needs and is what this run provides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runio import Run                                            # noqa: E402
from wm_schemes import (WMConfig, apply_greenlist_bias, detect,   # noqa: E402
                        exp_sample, is_green)

VOCAB = 4000
T = 400
N = 400
KEY = 20260819
OTHER_KEY = 987654321
GAMMA = 0.25
BIAS = 2.0


def zipf_probs(rng, v, s=1.0):
    w = 1.0 / np.power(np.arange(1, v + 1), s)
    w = w / w.sum()
    return w[rng.permutation(v)]


def generate(cfg, rng, n_tokens):
    toks = []
    for _ in range(n_tokens):
        p = zipf_probs(rng, cfg.vocab_size)
        if cfg.family == "greenlist":
            logits = np.log(np.clip(p, 1e-300, None))
            b = apply_greenlist_bias(cfg, logits, toks)
            b -= b.max()
            q = np.exp(b)
            q /= q.sum()
            toks.append(int(rng.choice(cfg.vocab_size, p=q)))
        else:
            toks.append(exp_sample(cfg, p, toks))
    return toks


def main() -> None:
    run = Run("exp_detector_validation_v1")
    rng = np.random.default_rng(KEY)
    out = {}

    # ---------------- N1: unwatermarked text -------------------------
    run.log("N1  unwatermarked token streams")
    plain = [[int(x) for x in rng.integers(0, VOCAB, size=T)] for _ in range(N)]
    for h in (0, 1, 2, 3):
        cfg = WMConfig("KGW", KEY, VOCAB, h=h, gamma=GAMMA, delta_bias=BIAS)
        dets = [detect(cfg, t) for t in plain]
        z = np.array([d["z"] for d in dets])
        gf = np.array([d["green_frac"] for d in dets])
        fpr4 = float(np.mean(z > 4.0))
        ks = stats.kstest(z, "norm")
        out[f"N1_h{h}"] = {
            "mean_green_fraction": float(gf.mean()),
            "gamma": GAMMA,
            "mean_z": float(z.mean()), "sd_z": float(z.std(ddof=1)),
            "empirical_fpr_at_z4": fpr4,
            "nominal_fpr_at_z4": float(stats.norm.sf(4.0)),
            "ks_stat_vs_standard_normal": float(ks.statistic),
            "ks_p": float(ks.pvalue),
        }
        run.log(f"  h={h}  green fraction {gf.mean():.4f} (gamma={GAMMA})  "
                f"z ~ N({z.mean():+.3f}, {z.std(ddof=1):.3f})  "
                f"KS={ks.statistic:.4f} p={ks.pvalue:.3f}  "
                f"FPR@z>4 = {fpr4:.4f}")

    # ---------------- N1b: the key-dependent null of h = 0 ------------
    # For h = 0 the green list is fixed for the whole text, so the null is
    # centred not on gamma but on the fraction of the vocabulary that this
    # particular key happens to colour green.  That fraction deviates from
    # gamma by O(sqrt(gamma(1-gamma)/|V|)), which displaces the z-score by
    # O(sqrt(T/|V|)).  The effect is real, it is a property of the scheme
    # and not of our implementation, and it disappears for h >= 1 because a
    # fresh list is drawn at every position.
    run.log("")
    run.log("N1b  key-dependent null offset of the h = 0 scheme")
    offsets = []
    for k in range(24):
        cfg0 = WMConfig("UNIGRAM", KEY + 1000 * (k + 1), VOCAB, h=0,
                        gamma=GAMMA, delta_bias=BIAS)
        zs = np.array([detect(cfg0, t)["z"] for t in plain[:80]])
        offsets.append(float(zs.mean()))
    offsets = np.array(offsets)
    predicted = float(np.sqrt(T / VOCAB))
    out["N1b"] = {
        "n_keys": len(offsets),
        "mean_offset_over_keys": float(offsets.mean()),
        "sd_offset_over_keys": float(offsets.std(ddof=1)),
        "predicted_sd_sqrt_T_over_V": predicted,
        "ratio_observed_to_predicted": float(offsets.std(ddof=1) / predicted),
    }
    run.log(f"  over {len(offsets)} keys: offset mean {offsets.mean():+.4f}, "
            f"sd {offsets.std(ddof=1):.4f}; predicted sd sqrt(T/|V|) = "
            f"{predicted:.4f}")
    run.log(f"  for the corpus experiment (T = 180, |V| = 151665) the same "
            f"quantity is {float(np.sqrt(180/151665)):.4f}, i.e. negligible")

    # ---------------- N2 and N3: wrong key versus right key ----------
    run.log("")
    run.log("N2/N3  watermarked text, wrong key versus right key")
    for name, h in (("KGW", 1), ("UNIGRAM", 0), ("EXP", 1)):
        cfg = WMConfig(name, KEY, VOCAB, h=h, gamma=GAMMA, delta_bias=BIAS)
        wrong = WMConfig(name, OTHER_KEY, VOCAB, h=h, gamma=GAMMA, delta_bias=BIAS)
        g = np.random.default_rng(KEY + 7)
        texts = [generate(cfg, g, T) for _ in range(120)]
        z_right = np.array([detect(cfg, t)["z"] for t in texts])
        z_wrong = np.array([detect(wrong, t)["z"] for t in texts])
        out[f"N23_{name}"] = {
            "h": h,
            "mean_z_right_key": float(z_right.mean()),
            "mean_z_wrong_key": float(z_wrong.mean()),
            "sd_z_wrong_key": float(z_wrong.std(ddof=1)),
            "detection_rate_right_key_at_z4": float(np.mean(z_right > 4.0)),
            "false_detection_rate_wrong_key_at_z4": float(np.mean(z_wrong > 4.0)),
        }
        run.log(f"  {name:8s} h={h}  right key: mean z = {z_right.mean():6.2f}, "
                f"detected {np.mean(z_right > 4.0)*100:5.1f}%   |   "
                f"wrong key: mean z = {z_wrong.mean():+.3f} "
                f"(sd {z_wrong.std(ddof=1):.3f}), "
                f"false alarms {np.mean(z_wrong > 4.0)*100:.1f}%")

    # ---------------- N4: uniformity of the EXP p-values -------------
    run.log("")
    run.log("N4  exponential family, uniformity of the p-values under the null")
    cfg = WMConfig("EXP", KEY, VOCAB, h=1)
    p = np.array([detect(cfg, t)["p"] for t in plain])
    ks = stats.kstest(p, "uniform")
    out["N4"] = {"ks_stat_vs_uniform": float(ks.statistic),
                 "ks_p": float(ks.pvalue), "mean_p": float(p.mean())}
    run.log(f"  KS against Uniform(0,1) = {ks.statistic:.4f}  p = {ks.pvalue:.3f}  "
            f"mean p-value = {p.mean():.4f}")

    green_ok = all(abs(out[f"N1_h{h}"]["mean_green_fraction"] - GAMMA) < 0.01
                   for h in (0, 1, 2, 3))
    # h >= 1 must be standard normal outright.  h = 0 is tested against the
    # scheme's own key-dependent null, i.e. the offset must be consistent
    # with sqrt(T/|V|) rather than with zero; demanding N(0,1) there would
    # be testing the wrong hypothesis.
    normal_ok = all(out[f"N1_h{h}"]["ks_p"] > 0.01 for h in (1, 2, 3))
    offset_ok = (abs(out["N1b"]["mean_offset_over_keys"]) < 0.15
                 and 0.4 < out["N1b"]["ratio_observed_to_predicted"] < 2.5)
    keyed_ok = all(abs(out[f"N23_{n}"]["mean_z_wrong_key"]) < 0.5
                   for n in ("KGW", "UNIGRAM", "EXP"))
    detect_ok = all(out[f"N23_{n}"]["detection_rate_right_key_at_z4"] > 0.95
                    for n in ("KGW", "UNIGRAM", "EXP"))
    unif_ok = out["N4"]["ks_p"] > 0.01
    passed = (green_ok and normal_ok and offset_ok and keyed_ok
              and detect_ok and unif_ok)

    run.log("")
    run.log(f"calibration checks passed: {passed}  "
            f"(green={green_ok}, normal[h>=1]={normal_ok}, "
            f"h0_offset={offset_ok}, keyed={keyed_ok}, "
            f"detect={detect_ok}, uniform={unif_ok})")

    run.write_json("validation.json", out)
    run.finish(
        conclusions={"all_checks_passed": bool(passed),
                     "green_fraction_ok": bool(green_ok),
                     "null_normal_ok_h_ge_1": bool(normal_ok),
                     "h0_key_offset_as_predicted": bool(offset_ok),
                     "h0_offset_sd": out["N1b"]["sd_offset_over_keys"],
                     "key_specific_ok": bool(keyed_ok),
                     "detects_with_right_key": bool(detect_ok),
                     "exp_pvalues_uniform": bool(unif_ok)},
        limitations=[
            "Self-calibration only: the schemes are re-implemented here "
            "rather than imported, so there are no published reference "
            "numbers to reproduce. Cross-checking against MarkLLM or the "
            "reference repositories remains open.",
            "The null streams are uniform over the vocabulary, which is "
            "the easiest null; natural text has burstiness that a real "
            "deployment must calibrate against.",
            "For h = 0 the null is key-dependent by construction, so the "
            "z-score is not exactly standard normal; the offset is of "
            "order sqrt(T/|V|) and is negligible at realistic vocabulary "
            "sizes but not at the synthetic |V| = 4000 used here.",
        ],
        inputs={"vocab": VOCAB, "T": T, "n_null": N, "key": KEY,
                "other_key": OTHER_KEY},
        command="python exp_detector_validation_v1.py",
    )


if __name__ == "__main__":
    main()
