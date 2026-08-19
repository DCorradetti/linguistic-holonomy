"""
exp_analysis_v1.py -- the preregistered test of Result D.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, sub-goals S9 and S10; Result D; falsification clause
Version  : v1
Purpose  : Decide, against the clause fixed in Plan_v1.md before any
           measurement was taken, whether the holonomy energy explains
           residual watermark detectability beyond the retention rate
           and the semantic deficit; and check Theorem C on real text by
           comparing the two context widths.
Input    : the newest run_exp_indicators_v1_* directory
Output   : run_exp_analysis_v1_<stamp>/analysis.json

The preregistered clause, quoted from Plan_v1.md
------------------------------------------------
  "If, in the regression of residual z-score on (rho, delta, eta) over
   matched-delta strata, the partial coefficient on eta is not
   significantly different from zero (alpha = 0.01, corrected for
   multiple comparisons) in at least two of the three schemes, then
   Result D is recorded as refuted, reported as a negative result in
   the manuscript, and the article is restructured around Props. A, B
   and Thm. C."

Ordinary least squares is implemented here in fifteen lines of numpy
rather than imported.  This is not asceticism: the regression is the
single number on which a preregistered claim turns, and it should be
auditable by a referee without tracing library defaults for centring,
degrees of freedom or standard-error conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runio import RESULTS, Run                                # noqa: E402

ALPHA = 0.01
MIN_Z0 = 4.0          # only texts the detector actually flags to begin with
N_STRATA = 4


def newest_indicators() -> Path:
    runs = sorted(RESULTS.glob("run_exp_indicators_v1_*"),
                  key=lambda p: p.name, reverse=True)
    for r in runs:
        if (r / "indicators.json").exists():
            return r / "indicators.json"
    raise FileNotFoundError("no completed indicators run found")


def ols(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """
    Least squares with an explicit intercept, classical standard errors.

    X is (n, k) without the intercept column; it is standardised so the
    coefficients are on a common scale and directly comparable.
    """
    n = X.shape[0]
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = np.column_stack([np.ones(n), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    dof = n - Z.shape[1]
    if dof <= 0:
        raise ValueError("not enough observations")
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Z.T @ Z)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t = beta / np.where(se < 1e-300, np.nan, se)
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(n), "dof": int(dof), "r2": float(r2),
        "terms": ["intercept"] + names,
        "beta": [float(b) for b in beta],
        "se": [float(s) for s in se],
        "t": [float(v) for v in t],
        "p": [float(v) for v in p],
    }


def report(run: Run, label: str, fit: dict) -> None:
    run.log(f"  {label}: n={fit['n']}  R2={fit['r2']:.3f}")
    for name, b, s, t, p in zip(fit["terms"], fit["beta"], fit["se"],
                                fit["t"], fit["p"]):
        run.log(f"     {name:22s} beta={b:+8.4f}  se={s:.4f}  "
                f"t={t:+7.2f}  p={p:.3e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicators", type=str, default=None)
    args = ap.parse_args()

    path = Path(args.indicators) if args.indicators else newest_indicators()
    run = Run("exp_analysis_v1")
    run.log(f"indicators = {path}")
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]

    kept = [r for r in rows if r["z0"] >= MIN_Z0 and np.isfinite(r["ratio"])]
    run.log(f"{len(kept)} of {len(rows)} chains retained "
            f"(z0 >= {MIN_Z0}); the rest were never detected to begin with")

    schemes = sorted({r["scheme"] for r in kept})
    out: dict = {"settings": {"alpha": ALPHA, "min_z0": MIN_Z0,
                              "n_strata": N_STRATA,
                              "n_total": len(rows), "n_kept": len(kept)}}

    # ---------------------------------------------------------------- #
    #  1. Theorem C on real text: does the measured intact-window        #
    #     fraction predict the residual, and do the two context widths   #
    #     separate as rho^(h+1) says they must?                          #
    # ---------------------------------------------------------------- #
    run.log("")
    run.log("Theorem C on real transformation chains")
    thmC = {}
    for s in schemes:
        sub = [r for r in kept if r["scheme"] == s]
        if len(sub) < 10:
            continue
        rho = np.array([r["rho"] for r in sub])
        rho_ctx = np.array([r["rho_ctx"] for r in sub])
        ratio = np.array([r["ratio"] for r in sub])
        h = sub[0]["h"]
        thmC[s] = {
            "h": h, "n": len(sub),
            "median_rho": float(np.median(rho)),
            "median_rho_ctx": float(np.median(rho_ctx)),
            "median_rho_pow": float(np.median(rho ** (h + 1))),
            "median_ratio": float(np.median(ratio)),
            "corr_ratio_rho_ctx": float(np.corrcoef(rho_ctx, ratio)[0, 1]),
            "corr_ratio_rho": float(np.corrcoef(rho, ratio)[0, 1]),
            "mean_abs_err_vs_rho_ctx": float(np.mean(np.abs(ratio - rho_ctx))),
            "mean_abs_err_vs_rho_pow": float(np.mean(np.abs(ratio - rho ** (h + 1)))),
        }
        t = thmC[s]
        run.log(f"  {s:8s} h={h}  n={t['n']:4d}  rho={t['median_rho']:.3f}  "
                f"rho^(h+1)={t['median_rho_pow']:.3f}  "
                f"measured intact-window={t['median_rho_ctx']:.3f}  "
                f"observed z-ratio={t['median_ratio']:.3f}")
        run.log(f"           corr(ratio, intact-window)={t['corr_ratio_rho_ctx']:+.3f}  "
                f"corr(ratio, rho)={t['corr_ratio_rho']:+.3f}")
    out["theorem_C"] = thmC

    # ---------------------------------------------------------------- #
    #  2. Result D, the preregistered test                               #
    # ---------------------------------------------------------------- #
    run.log("")
    run.log("Result D: does the holonomy energy add anything to (rho, delta)?")
    n_tests = max(1, len(schemes))
    alpha_corr = ALPHA / n_tests
    run.log(f"  Bonferroni-corrected threshold: alpha = {ALPHA} / {n_tests} "
            f"= {alpha_corr:.2e}")

    resultD, significant = {}, 0
    for s in schemes:
        sub = [r for r in kept if r["scheme"] == s]
        if len(sub) < 20:
            run.log(f"  {s}: only {len(sub)} observations, skipped")
            continue
        X = np.array([[r["rho"], r["semantic_deficit"], r["holonomy_energy"]]
                      for r in sub])
        y = np.array([r["ratio"] for r in sub])
        fit = ols(X, y, ["rho", "semantic_deficit", "holonomy_energy"])
        report(run, s, fit)
        i = fit["terms"].index("holonomy_energy")
        passes = bool(fit["p"][i] < alpha_corr and fit["beta"][i] < 0)
        significant += int(passes)
        run.log(f"     -> holonomy term significant and negative: {passes}")
        resultD[s] = {"fit": fit, "eta_beta": fit["beta"][i],
                      "eta_p": fit["p"][i], "passes": passes}
    out["result_D_full"] = resultD

    # ---------------------------------------------------------------- #
    #  3. Matched-delta strata: the design the plan actually specified   #
    # ---------------------------------------------------------------- #
    run.log("")
    run.log("Matched-delta strata (semantic endpoint held approximately fixed)")
    strata_out = {}
    for s in schemes:
        sub = [r for r in kept if r["scheme"] == s]
        if len(sub) < 40:
            continue
        d = np.array([r["semantic_deficit"] for r in sub])
        edges = np.quantile(d, np.linspace(0, 1, N_STRATA + 1))
        bands = []
        for k in range(N_STRATA):
            lo, hi = edges[k], edges[k + 1]
            idx = [i for i, v in enumerate(d)
                   if (v >= lo and (v < hi or k == N_STRATA - 1))]
            if len(idx) < 12:
                continue
            eta = np.array([sub[i]["holonomy_energy"] for i in idx])
            rat = np.array([sub[i]["ratio"] for i in idx])
            rho = np.array([sub[i]["rho"] for i in idx])
            rr, pp = stats.spearmanr(eta, rat)
            bands.append({
                "band": k, "delta_lo": float(lo), "delta_hi": float(hi),
                "n": len(idx),
                "delta_spread": float(hi - lo),
                "eta_min": float(eta.min()), "eta_max": float(eta.max()),
                "ratio_min": float(rat.min()), "ratio_max": float(rat.max()),
                "rho_spread": float(rho.max() - rho.min()),
                "spearman_eta_ratio": float(rr), "spearman_p": float(pp),
            })
            run.log(f"  {s:8s} band {k}  delta in [{lo:.4f}, {hi:.4f}]  "
                    f"n={len(idx):3d}  eta in [{eta.min():.3f}, {eta.max():.3f}]  "
                    f"spearman(eta, z-ratio)={rr:+.3f} (p={pp:.2e})")
        strata_out[s] = bands
    out["matched_delta_strata"] = strata_out

    # ---------------------------------------------------------------- #
    #  3b. EXPLORATORY, and labelled as such.                            #
    #      The preregistered clause names the holonomy energy of the      #
    #      full chain. The same regression run on the English-waypoint    #
    #      channel is reported because a multilingual encoder is nearly   #
    #      invariant under translation and therefore flattens the very    #
    #      excursion the path datum is meant to capture. It is NOT        #
    #      allowed to change the verdict, and does not.                   #
    # ---------------------------------------------------------------- #
    run.log("")
    run.log("Exploratory: the same test on the English-waypoint channel")
    run.log("  (not preregistered; cannot change the verdict)")
    explor = {}
    for s in schemes:
        sub = [r for r in kept if r["scheme"] == s
               and r.get("holonomy_energy_en") is not None]
        if len(sub) < 20:
            continue
        X = np.array([[r["rho"], r["semantic_deficit_en"],
                       r["holonomy_energy_en"]] for r in sub])
        y = np.array([r["ratio"] for r in sub])
        fit = ols(X, y, ["rho", "semantic_deficit_en", "holonomy_energy_en"])
        report(run, f"{s} (English channel)", fit)
        i = fit["terms"].index("holonomy_energy_en")
        explor[s] = {"fit": fit, "eta_en_beta": fit["beta"][i],
                     "eta_en_p": fit["p"][i]}
    out["result_D_english_channel_exploratory"] = explor

    # Descriptive comparison of the two channels, to document the flattening.
    en = [r["holonomy_energy_en"] for r in kept
          if r.get("holonomy_energy_en") is not None]
    full = [r["holonomy_energy"] for r in kept]
    if en:
        out["channel_comparison"] = {
            "median_eta_full_chain": float(np.median(full)),
            "median_eta_english_only": float(np.median(en)),
            "median_delta_full_chain": float(np.median(
                [r["semantic_deficit"] for r in kept])),
            "median_delta_english_only": float(np.median(
                [r["semantic_deficit_en"] for r in kept
                 if r.get("semantic_deficit_en") is not None])),
        }
        cc = out["channel_comparison"]
        run.log("")
        run.log(f"  median holonomy energy: full chain "
                f"{cc['median_eta_full_chain']:.4f}, English waypoints only "
                f"{cc['median_eta_english_only']:.4f}")

    # ---------------------------------------------------------------- #
    #  4. Verdict                                                        #
    # ---------------------------------------------------------------- #
    # The clause is written over three schemes. If fewer than two could be
    # tested at all, the clause has not been executed and the honest verdict
    # is neither support nor refutation.
    if len(resultD) < 2:
        verdict = "UNDETERMINED"
    elif significant >= 2:
        verdict = "SUPPORTED"
    else:
        verdict = "REFUTED"
    run.log("")
    run.log(f"Preregistered verdict on Result D: {verdict} "
            f"({significant} of {len(resultD)} schemes pass)")
    if verdict == "UNDETERMINED":
        run.log("  Fewer than two schemes had enough observations to run the "
                "test; the preregistered clause was not executed.")
    if verdict == "REFUTED":
        run.log("  Per Plan_v1.md this is not a kill criterion. The article "
                "is restructured around Propositions A, B and Theorem C, and "
                "the null result is reported in full.")
    out["verdict"] = {"result_D": verdict, "n_schemes_passing": significant,
                      "n_schemes_tested": len(resultD)}

    run.write_json("analysis.json", out)
    run.finish(
        conclusions={
            "result_D_verdict": verdict,
            "n_schemes_passing": significant,
            "theorem_C": {k: {"h": v["h"],
                              "median_ratio": v["median_ratio"],
                              "median_rho_pow": v["median_rho_pow"],
                              "corr_ratio_intact_window": v["corr_ratio_rho_ctx"]}
                          for k, v in thmC.items()},
        },
        limitations=[
            "Round-trip machine translation is one attack family; the "
            "verdict does not transfer to learned paraphrasers untested here.",
            "Chains within a scheme share the same 36 prompts, so "
            "observations are not fully independent; the standard errors "
            "are therefore optimistic.",
            "A single embedder defines both delta and eta.",
            "The holonomy energy is one scalar reduction of H; a null "
            "result for it is not a null result for the path datum itself.",
        ],
        inputs={"indicators": str(path), "alpha": ALPHA, "min_z0": MIN_Z0},
        command="python exp_analysis_v1.py",
    )


if __name__ == "__main__":
    main()
