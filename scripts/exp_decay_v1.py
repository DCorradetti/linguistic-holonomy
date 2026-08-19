"""
exp_decay_v1.py -- validation of Theorem C, the context-width decay law.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, sub-goal S3 (numerical part), Theorem C, C1, C2
Version  : v1
Purpose  : Test, against the exact detectors of `wm_schemes.py`, the
           prediction

               E[z_att] = rho^(h+1) . z_0,

           and test the hypothesis under which it was derived by
           replacing i.i.d. edits with structured ones at the same
           retention rate.
Input    : none (synthetic language model, fixed seeds)
Output   : run_exp_decay_v1_<stamp>/decay.json in `7. Results/`

Why a synthetic generator
-------------------------
Theorem C is a statement about a detector applied to a token stream
under an edit process.  It does not mention a language model, and using
a real one would introduce confounds (entropy varying with content,
tokenizer artefacts, repetition) without making the test sharper.  The
generator here emits from a Zipf distribution with a freshly permuted
support at each step, which fixes the entropy at a known value and lets
us vary rho and h cleanly.  The corresponding experiment on real
watermarked text is `exp_corpus_v1.py`.

Edit models
-----------
  iid       every position independently replaced with probability 1-rho
  block     one contiguous run of (1-rho)T positions replaced
  periodic  every k-th position replaced, k = 1/(1-rho)

All three have the same retention rate.  Theorem C is derived under the
first; the other two probe how much the arrangement of the edits, at
fixed rate, matters.  This is the surface-channel counterpart of the
path-dependence claim that the geometric part of the article makes in
the semantic channel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loop_geometry import context_retention_rate                 # noqa: E402
from runio import Run                                            # noqa: E402
from wm_schemes import (WMConfig, apply_greenlist_bias, detect,   # noqa: E402
                        exp_sample)

VOCAB = 4000
T_TOKENS = 400
N_SEQ = 120
HS = [0, 1, 2, 3]
RHOS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
KEY = 20260819


def zipf_probs(rng: np.random.Generator, v: int, s: float = 1.0) -> np.ndarray:
    """A Zipf law over a freshly permuted support: fixed entropy, no memory."""
    w = 1.0 / np.power(np.arange(1, v + 1), s)
    w = w / w.sum()
    return w[rng.permutation(v)]


def generate(cfg: WMConfig, rng: np.random.Generator, T: int) -> list[int]:
    toks: list[int] = []
    for _ in range(T):
        p = zipf_probs(rng, cfg.vocab_size)
        if cfg.family == "greenlist":
            logits = np.log(np.clip(p, 1e-300, None))
            biased = apply_greenlist_bias(cfg, logits, toks)
            biased -= biased.max()
            q = np.exp(biased)
            q /= q.sum()
            toks.append(int(rng.choice(cfg.vocab_size, p=q)))
        else:
            toks.append(exp_sample(cfg, p, toks))
    return toks


def edit(toks: list[int], rho: float, model: str,
         rng: np.random.Generator, vocab: int) -> list[int]:
    """Replace a (1 - rho) fraction of positions, arranged in three ways."""
    T = len(toks)
    n_edit = int(round((1.0 - rho) * T))
    out = list(toks)
    if n_edit == 0:
        return out
    if model == "iid":
        idx = rng.choice(T, size=n_edit, replace=False)
    elif model == "block":
        start = int(rng.integers(0, max(1, T - n_edit + 1)))
        idx = np.arange(start, start + n_edit)
    elif model == "periodic":
        step = max(1, T // n_edit)
        idx = np.arange(0, T, step)[:n_edit]
    else:
        raise ValueError(model)
    for i in idx:
        out[int(i)] = int(rng.integers(0, vocab))
    return out


def excess(cfg: WMConfig, det: dict) -> float:
    """
    The detector statistic on a scale where the null is zero.
    For the green-list family the z-score already is such a scale; for
    the exponential family we use the excess S - T of the Gamma
    statistic, which has null mean zero.
    """
    if cfg.family == "greenlist":
        return float(det["z"])
    return float(det["S"] - det["n_scored"])


def main() -> None:
    run = Run("exp_decay_v1")
    rows = []

    schemes = [("KGW", h) for h in HS] + [("EXP", 1)]
    for name, h in schemes:
        cfg = WMConfig(name, KEY, VOCAB, h=h, gamma=0.25, delta_bias=2.0)
        rng = np.random.default_rng(KEY + 17 * h + (0 if name == "KGW" else 999))
        base = [generate(cfg, rng, T_TOKENS) for _ in range(N_SEQ)]
        e0 = np.array([excess(cfg, detect(cfg, t)) for t in base])
        run.log(f"{name} h={h}: baseline statistic = {e0.mean():.2f} "
                f"(sd {e0.std():.2f})")

        for model in ("iid", "block", "periodic"):
            for rho in RHOS:
                att = [edit(t, rho, model, rng, VOCAB) for t in base]
                ea = np.array([excess(cfg, detect(cfg, t)) for t in att])
                ratio = float(ea.mean() / e0.mean())
                pred = float(rho ** (h + 1))
                ctx = float(np.mean([context_retention_rate(b, a, h)
                                     for b, a in zip(base, att)]))
                rows.append({
                    "scheme": name, "h": h, "edit_model": model, "rho": rho,
                    "statistic_0": float(e0.mean()),
                    "statistic_att": float(ea.mean()),
                    "ratio_observed": ratio,
                    "ratio_predicted": pred,
                    "abs_error": abs(ratio - pred),
                    "context_retention_measured": ctx,
                })
            sub = [r for r in rows if r["scheme"] == name and r["h"] == h
                   and r["edit_model"] == model]
            mae = float(np.mean([r["abs_error"] for r in sub]))
            run.log(f"  {model:9s} mean |observed - rho^(h+1)| = {mae:.4f}")

    # --- headline table: the i.i.d. case, where the theorem applies -------
    run.log("")
    run.log("Theorem C under i.i.d. edits (observed / predicted):")
    run.log("  scheme  h   rho    observed  predicted   ctx-retention")
    for r in rows:
        if r["edit_model"] == "iid":
            run.log(f"  {r['scheme']:7s} {r['h']}  {r['rho']:.2f}   "
                    f"{r['ratio_observed']:8.4f}  {r['ratio_predicted']:8.4f}   "
                    f"{r['context_retention_measured']:8.4f}")

    iid = [r for r in rows if r["edit_model"] == "iid" and r["rho"] < 1.0]
    mae_iid = float(np.mean([r["abs_error"] for r in iid]))

    # --- the hypothesis probe: same rho, different arrangement ------------
    run.log("")
    run.log("Same retention rate, different arrangement of the edits (KGW h=1):")
    run.log("  rho     iid      block   periodic")
    probe = {}
    for rho in RHOS:
        vals = {}
        for model in ("iid", "block", "periodic"):
            m = [r for r in rows if r["scheme"] == "KGW" and r["h"] == 1
                 and r["edit_model"] == model and r["rho"] == rho]
            vals[model] = m[0]["ratio_observed"] if m else float("nan")
        probe[str(rho)] = vals
        run.log(f"  {rho:.2f}  {vals['iid']:7.4f}  {vals['block']:7.4f}  "
                f"{vals['periodic']:7.4f}")
    spread = max(
        abs(v["block"] - v["iid"]) for k, v in probe.items() if float(k) < 1.0
    )
    run.log(f"  --> largest gap between block and i.i.d. at equal rho: {spread:.4f}")

    run.write_json("decay.json", {"rows": rows, "arrangement_probe": probe,
                                  "settings": {"vocab": VOCAB, "T": T_TOKENS,
                                               "n_seq": N_SEQ, "hs": HS,
                                               "rhos": RHOS, "key": KEY}})
    run.finish(
        conclusions={
            "theorem_C_mae_iid": mae_iid,
            "theorem_C_supported": bool(mae_iid < 0.05),
            "max_block_vs_iid_gap_at_equal_rho": float(spread),
            "arrangement_matters": bool(spread > 0.05),
        },
        limitations=[
            "Substitution only: insertions and deletions shift the seeding "
            "window and are outside the model of Theorem C.",
            "The generator is synthetic; entropy is held fixed by "
            "construction and does not vary with content as it does in real text.",
            "Replacement tokens are drawn uniformly, which is the most "
            "favourable case for the null calibration.",
        ],
        inputs={"vocab": VOCAB, "T": T_TOKENS, "n_seq": N_SEQ, "key": KEY},
        command="python exp_decay_v1.py",
    )


if __name__ == "__main__":
    main()
