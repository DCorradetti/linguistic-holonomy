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
        if (r / "indicators.json").exists() and (r / "manifest.json").exists():
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


def ols_clustered(X: np.ndarray, y: np.ndarray, groups: list,
                  names: list[str], absorb: bool = False) -> dict:
    """
    The same least squares with a standard error clustered on the base
    text, and optionally with the base text absorbed as a fixed effect.

    The six chains of a scheme are applied to one and the same thirty
    passages, so the residuals of the six rows that share a passage are
    anything but independent; the classical standard error of `ols` is
    therefore optimistic. With `absorb=True` every variable is measured
    in deviation from its own passage's mean, and the coefficient answers
    the question one actually wants answered: holding the passage fixed,
    does a longer excursion cost more of the mark?
    """
    n = X.shape[0]
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    yy = np.array(y, dtype=float)
    g = np.asarray(groups)
    uniq = sorted(set(g.tolist()))
    if absorb:
        for u in uniq:
            m = g == u
            Z[m] -= Z[m].mean(0)
            yy[m] -= yy[m].mean()
        D, terms = Z, list(names)
    else:
        D = np.column_stack([np.ones(n), Z])
        terms = ["intercept"] + list(names)
    beta, *_ = np.linalg.lstsq(D, yy, rcond=None)
    resid = yy - D @ beta
    bread = np.linalg.pinv(D.T @ D)
    meat = np.zeros((D.shape[1], D.shape[1]))
    for u in uniq:
        m = g == u
        sc = D[m].T @ resid[m]
        meat += np.outer(sc, sc)
    G = len(uniq)
    k_eff = D.shape[1] + (G if absorb else 0)
    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k_eff, 1))
    cov = c * bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t = beta / np.where(se < 1e-300, np.nan, se)
    pv = 2.0 * stats.t.sf(np.abs(t), max(G - 1, 1))
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {"n": int(n), "n_clusters": int(G), "absorbed": bool(absorb),
            "r2_within" if absorb else "r2": float(r2),
            "terms": terms,
            "beta": [float(b) for b in beta],
            "se": [float(v) for v in se],
            "t": [float(v) for v in t],
            "p": [float(v) for v in pv]}


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
    n_weak = sum(1 for r in rows if r["z0"] < MIN_Z0)
    # A p-value that underflows to exactly one sends the normal-equivalent
    # z to minus infinity. It means the statistic fell below its null mean
    # by more than double precision can express, which is a way of saying
    # the mark is gone; but it is not a number one can regress on.
    n_inf = sum(1 for r in rows
                if r["z0"] >= MIN_Z0 and not np.isfinite(r["ratio"]))
    run.log(f"{len(kept)} of {len(rows)} chains retained; {n_weak} were "
            f"never detected to begin with (z0 < {MIN_Z0}) and {n_inf} have "
            f"a residual statistic that underflowed to minus infinity")

    schemes = sorted({r["scheme"] for r in kept})
    out: dict = {"settings": {"alpha": ALPHA, "min_z0": MIN_Z0,
                              "n_strata": N_STRATA,
                              "n_total": len(rows), "n_kept": len(kept),
                              "n_dropped_weak": n_weak,
                              "n_dropped_nonfinite": n_inf}}

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

        # The intact-window law was proved for substitutions that preserve
        # length. Round-trip translation does not preserve it, and the
        # statistic of every scheme here is normalised by the square root
        # of the number of scored positions, so the residual ratio the law
        # predicts on a text of changed length is
        #
        #     |I| / sqrt(T'_0 T'_att) = rho_ctx * sqrt(T'_0 / T'_att).
        #
        # This is not a new hypothesis: it is the same theorem, with the
        # normalisation of the z-score carried through honestly.
        n0 = np.array([float(r["n_tokens_0"]) for r in sub])
        n1 = np.array([float(r["n_tokens_final"]) for r in sub])
        scale = np.sqrt(np.maximum(n0 - h, 1.0) / np.maximum(n1 - h, 1.0))
        pred_len = rho_ctx * scale

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
            # length-corrected form
            "median_length_ratio": float(np.median(n1 / np.maximum(n0, 1.0))),
            "median_pred_length_corrected": float(np.median(pred_len)),
            "corr_ratio_pred_length_corrected":
                float(np.corrcoef(pred_len, ratio)[0, 1]),
            "mean_abs_err_vs_pred_length_corrected":
                float(np.mean(np.abs(ratio - pred_len))),
            "frac_ratio_above_one": float(np.mean(ratio > 1.0)),
            # The error distribution has a tail: machine translation
            # occasionally degenerates into repetitive output, on which the
            # independence hypothesis behind the law fails outright. The
            # median error is therefore reported beside the mean.
            "median_abs_err_vs_rho_ctx":
                float(np.median(np.abs(ratio - rho_ctx))),
            "median_abs_err_vs_pred_length_corrected":
                float(np.median(np.abs(ratio - pred_len))),
            "q90_length_ratio": float(np.quantile(n1 / np.maximum(n0, 1.0),
                                                  0.9)),
            "max_length_ratio": float(np.max(n1 / np.maximum(n0, 1.0))),
            "n_residual_below_minus3": int(np.sum(
                np.array([r["z_final"] for r in sub]) < -3.0)),
        }
        t = thmC[s]
        run.log(f"  {s:8s} h={h}  n={t['n']:4d}  rho={t['median_rho']:.3f}  "
                f"rho^(h+1)={t['median_rho_pow']:.3f}  "
                f"measured intact-window={t['median_rho_ctx']:.3f}  "
                f"observed z-ratio={t['median_ratio']:.3f}")
        run.log(f"           corr(ratio, intact-window)={t['corr_ratio_rho_ctx']:+.3f}  "
                f"corr(ratio, rho)={t['corr_ratio_rho']:+.3f}")
        run.log(f"           length ratio={t['median_length_ratio']:.3f}  "
                f"length-corrected prediction="
                f"{t['median_pred_length_corrected']:.3f}  "
                f"corr={t['corr_ratio_pred_length_corrected']:+.3f}  "
                f"MAE {t['mean_abs_err_vs_rho_ctx']:.3f} -> "
                f"{t['mean_abs_err_vs_pred_length_corrected']:.3f}  "
                f"(fraction of chains with residual above the original: "
                f"{t['frac_ratio_above_one']:.3f})")
        run.log(f"           median errors {t['median_abs_err_vs_rho_ctx']:.3f}"
                f" -> {t['median_abs_err_vs_pred_length_corrected']:.3f}  "
                f"length ratio q90={t['q90_length_ratio']:.2f} "
                f"max={t['max_length_ratio']:.2f}  "
                f"chains with residual z below -3: "
                f"{t['n_residual_below_minus3']}")
    out["theorem_C"] = thmC

    # The law speaks about one text carried along attacks of varying
    # severity, not about a comparison between texts of different entropy.
    # Pooling the two questions is what produces the sign anomaly of the
    # context-free scheme: between passages the association is negative,
    # within a passage it is strongly positive, and it is the second that
    # the theorem is about.
    run.log("")
    run.log("Theorem C within the base text (each passage against its own "
            "six chains)")
    within = {}
    for s in schemes:
        sub = [r for r in kept if r["scheme"] == s]
        per, per_len = [], []
        for uid in sorted({r["uid"] for r in sub}):
            grp = [r for r in sub if r["uid"] == uid]
            if len(grp) < 4:
                continue
            rc = np.array([r["rho_ctx"] for r in grp])
            rt = np.array([r["ratio"] for r in grp])
            n0 = np.array([float(r["n_tokens_0"]) for r in grp])
            n1 = np.array([float(r["n_tokens_final"]) for r in grp])
            hh = grp[0]["h"]
            pl = rc * np.sqrt(np.maximum(n0 - hh, 1.0) /
                              np.maximum(n1 - hh, 1.0))
            if rc.std() > 1e-9 and rt.std() > 1e-9:
                per.append(float(np.corrcoef(rc, rt)[0, 1]))
            if pl.std() > 1e-9 and rt.std() > 1e-9:
                per_len.append(float(np.corrcoef(pl, rt)[0, 1]))
        if not per:
            continue
        arr, arr_len = np.array(per), np.array(per_len)
        within[s] = {
            "n_texts": len(per),
            "mean_corr_intact_window": float(arr.mean()),
            "median_corr_intact_window": float(np.median(arr)),
            "n_positive": int((arr > 0).sum()),
            "mean_corr_prediction_length_corrected":
                float(arr_len.mean()) if len(arr_len) else float("nan"),
            "n_positive_length_corrected":
                int((arr_len > 0).sum()) if len(arr_len) else 0,
            "pooled_corr_intact_window": thmC[s]["corr_ratio_rho_ctx"],
        }
        w = within[s]
        run.log(f"  {s:8s} over {w['n_texts']} passages: mean within-text "
                f"corr(intact-window, residual) = "
                f"{w['mean_corr_intact_window']:+.3f} "
                f"(median {w['median_corr_intact_window']:+.3f}, positive in "
                f"{w['n_positive']} of {w['n_texts']}); pooled over passages "
                f"it is {w['pooled_corr_intact_window']:+.3f}")
    out["theorem_C_within_text"] = within

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

    # The clause was written over the regression above and is settled by it.
    # It did not say which standard error to use, and the observations are
    # clustered by passage; both the clustered version and the version that
    # absorbs the passage entirely are therefore reported. They are checks
    # on the preregistered test, not a substitute for it.
    run.log("")
    run.log("Robustness of Result D: standard errors clustered on the base "
            "text, and the same regression with the base text absorbed")
    robust = {}
    for s in schemes:
        sub = [r for r in kept if r["scheme"] == s]
        if len(sub) < 20:
            continue
        X = np.array([[r["rho"], r["semantic_deficit"], r["holonomy_energy"]]
                      for r in sub])
        y = np.array([r["ratio"] for r in sub])
        grp = [r["uid"] for r in sub]
        nm = ["rho", "semantic_deficit", "holonomy_energy"]
        cl = ols_clustered(X, y, grp, nm, absorb=False)
        fe = ols_clustered(X, y, grp, nm, absorb=True)
        i_cl = cl["terms"].index("holonomy_energy")
        i_fe = fe["terms"].index("holonomy_energy")
        robust[s] = {"clustered": cl, "fixed_effects": fe,
                     "eta_beta_clustered": cl["beta"][i_cl],
                     "eta_p_clustered": cl["p"][i_cl],
                     "eta_beta_fixed_effects": fe["beta"][i_fe],
                     "eta_p_fixed_effects": fe["p"][i_fe]}
        run.log(f"  {s:8s} clustered ({cl['n_clusters']} passages): "
                f"eta beta={cl['beta'][i_cl]:+.4f} "
                f"se={cl['se'][i_cl]:.4f} p={cl['p'][i_cl]:.3e}")
        run.log(f"           passage absorbed: eta beta="
                f"{fe['beta'][i_fe]:+.4f} se={fe['se'][i_fe]:.4f} "
                f"p={fe['p'][i_fe]:.3e}  (within R2="
                f"{fe['r2_within']:.3f})")
    out["result_D_robustness"] = robust

    # delta and eta both grow with the number of pivots, so they are far
    # from orthogonal; entering the same regression with opposite signs they
    # form a suppression pair, and their separate coefficients should not be
    # read as separate effects. What the clause tests is only whether eta
    # adds anything at all once (rho, delta) are held.
    coll = {}
    for s in schemes:
        sub = [r for r in kept if r["scheme"] == s]
        d = np.array([r["semantic_deficit"] for r in sub])
        e = np.array([r["holonomy_energy"] for r in sub])
        per = []
        for uid in sorted({r["uid"] for r in sub}):
            g = [r for r in sub if r["uid"] == uid]
            if len(g) < 4:
                continue
            dd = np.array([r["semantic_deficit"] for r in g])
            ee = np.array([r["holonomy_energy"] for r in g])
            if dd.std() > 1e-12 and ee.std() > 1e-12:
                per.append(float(np.corrcoef(dd, ee)[0, 1]))
        coll[s] = {"corr_delta_eta_pooled": float(np.corrcoef(d, e)[0, 1]),
                   "mean_corr_delta_eta_within_text":
                       float(np.mean(per)) if per else float("nan"),
                   "n_texts": len(per)}
        run.log(f"  {s:8s} corr(delta, eta) = "
                f"{coll[s]['corr_delta_eta_pooled']:+.3f} pooled, "
                f"{coll[s]['mean_corr_delta_eta_within_text']:+.3f} within "
                f"the base text")
    out["collinearity_delta_eta"] = coll

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
    # A chain with only two English waypoints has no holonomy in that
    # channel at all: the loop rotation is the direct rotation and H is the
    # identity, exactly. Single round trips are therefore excluded here, not
    # because their value is inconvenient but because it is structurally
    # zero and would enter the regression as a constant.
    explor = {}
    for s in schemes:
        sub = [r for r in kept if r["scheme"] == s
               and r.get("holonomy_energy_en") is not None
               and r.get("n_english_waypoints", 0) >= 3]
        if len(sub) < 20:
            run.log(f"  {s}: only {len(sub)} chains with three or more "
                    f"English waypoints, skipped")
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
    multi = [r for r in kept if r.get("n_english_waypoints", 0) >= 3
             and r.get("holonomy_energy_en") is not None]
    full = [r["holonomy_energy"] for r in kept]
    if multi:
        out["channel_comparison"] = {
            "median_eta_full_chain": float(np.median(full)),
            "median_eta_full_chain_multi_pivot": float(np.median(
                [r["holonomy_energy"] for r in multi])),
            "median_eta_english_only_multi_pivot": float(np.median(
                [r["holonomy_energy_en"] for r in multi])),
            "n_multi_pivot": len(multi),
            "n_single_round_trip": len(kept) - len(multi),
            "median_delta_full_chain": float(np.median(
                [r["semantic_deficit"] for r in kept])),
            "median_delta_english_only": float(np.median(
                [r["semantic_deficit_en"] for r in kept
                 if r.get("semantic_deficit_en") is not None])),
        }
        cc = out["channel_comparison"]
        run.log("")
        run.log(f"  median holonomy energy over the {cc['n_multi_pivot']} "
                f"multi-pivot chains: full chain "
                f"{cc['median_eta_full_chain_multi_pivot']:.4f}, English "
                f"waypoints only "
                f"{cc['median_eta_english_only_multi_pivot']:.4f}")
        run.log(f"  ({cc['n_single_round_trip']} single round trips have two "
                f"English waypoints and therefore no holonomy in that "
                f"channel by construction)")

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
            "result_D_robustness": {k: {
                "eta_beta_clustered": v["eta_beta_clustered"],
                "eta_p_clustered": v["eta_p_clustered"],
                "eta_beta_fixed_effects": v["eta_beta_fixed_effects"],
                "eta_p_fixed_effects": v["eta_p_fixed_effects"]}
                for k, v in robust.items()},
            "theorem_C_within_text": within,
            "collinearity_delta_eta": coll,
            "theorem_C": {k: {"h": v["h"], "n": v["n"],
                              "median_rho": v["median_rho"],
                              "median_ratio": v["median_ratio"],
                              "median_rho_pow": v["median_rho_pow"],
                              "median_intact_window": v["median_rho_ctx"],
                              "median_length_ratio": v["median_length_ratio"],
                              "median_prediction_length_corrected":
                                  v["median_pred_length_corrected"],
                              "corr_ratio_intact_window": v["corr_ratio_rho_ctx"],
                              "corr_ratio_prediction_length_corrected":
                                  v["corr_ratio_pred_length_corrected"],
                              "mean_abs_err_vs_intact_window":
                                  v["mean_abs_err_vs_rho_ctx"],
                              "mean_abs_err_vs_prediction_length_corrected":
                                  v["mean_abs_err_vs_pred_length_corrected"],
                              "median_abs_err_vs_intact_window":
                                  v["median_abs_err_vs_rho_ctx"],
                              "median_abs_err_vs_prediction_length_corrected":
                                  v["median_abs_err_vs_pred_length_corrected"],
                              "n_residual_below_minus3":
                                  v["n_residual_below_minus3"]}
                          for k, v in thmC.items()},
        },
        limitations=[
            "Round-trip machine translation is one attack family; the "
            "verdict does not transfer to learned paraphrasers untested here.",
            "The six chains of a scheme are applied to one and the same "
            "set of base texts, so the observations are clustered by "
            "prompt and the classical standard errors reported here are "
            "optimistic.",
            "A single embedder defines both delta and eta.",
            "The holonomy energy is one scalar reduction of H; a null "
            "result for it is not a null result for the path datum itself.",
            "Machine translation occasionally degenerates into repetitive "
            "output, on which the independence hypothesis behind the "
            "intact-window law fails outright; such chains are reported, "
            "not removed.",
            "The intact-window count is read off the longest common "
            "subsequence of the two token strings and does not check that "
            "a surviving window is still contiguous in the attacked text, "
            "so it is an upper bound on the number of intact seeding "
            "windows the detector actually sees.",
        ],
        inputs={"indicators": str(path), "alpha": ALPHA, "min_z0": MIN_Z0},
        command="python exp_analysis_v1.py",
    )


if __name__ == "__main__":
    main()
