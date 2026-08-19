"""
exp_geometry_v1.py -- numerical validation of Propositions A and B.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, sub-goal S4; kill criterion K1
Version  : v1
Purpose  : Verify to machine precision, over random chains and over the
           embedding dimensions actually used in the article, that
             (A) I - sym(R_U) is positive semidefinite, its Sylvester
                 signature is (2p, 0, n - 2p), and it therefore carries
                 exactly the information of the rank;
             (B) H = R_direct^{-1} R_U fixes the base point, the
                 semantic deficit depends only on R_direct, and the two
                 data are independent (realisability).
Input    : none (pseudo-random chains, fixed seed)
Output   : run_exp_geometry_v1_<stamp>/geometry.json in `7. Results/`

A numerical check is not a proof; the proofs are in
`3. Certificates/Article_LLW/`.  What this run establishes is that the
proofs describe the objects the rest of the pipeline actually computes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loop_geometry import (angle_spectrum, cosine_distance, holonomy,  # noqa: E402
                           loop_rotation, quadratic_form_matrix,
                           signature, unit)
from runio import Run                                                  # noqa: E402

DIMS = [8, 16, 64, 384]          # 384 is the dimension of the embedder used later
N_TRIALS = 300
CHAIN_LENGTHS = [2, 3, 4, 6, 8]


def random_chain(rng, n: int, L: int, step: float = 0.7) -> list[np.ndarray]:
    """A random walk in R^n, standing for an arbitrary chain of states."""
    V = [rng.normal(size=n)]
    for _ in range(L):
        V.append(V[-1] + step * rng.normal(size=n))
    return V


def check_propositions(rng, n: int, L: int) -> dict:
    V = random_chain(rng, n, L)
    R_U, R_dir = loop_rotation(V)
    H = R_dir.T @ R_U
    M = quadratic_form_matrix(R_U)
    ev = np.linalg.eigvalsh(M)
    sig = signature(M)
    theta = angle_spectrum(R_U)
    v0h, vLh = unit(V[0]), unit(V[-1])
    return {
        # Proposition A
        "min_eig": float(ev.min()),
        "n_minus": int(sig[1]),
        "n_plus_minus_2p": int(sig[0] - 2 * len(theta)),
        # Proposition A: the quadratic form reproduces the semantic deficit
        "quadform_err": float(abs(float(v0h @ M @ v0h) - cosine_distance(V[0], V[-1]))),
        # Proposition B
        "fix_err": float(np.linalg.norm(H @ v0h - v0h)),
        "delta_from_Rdir_err": float(abs(cosine_distance(V[0], V[-1])
                                         - (1.0 - float(v0h @ (R_dir @ v0h))))),
        "factorisation_err": float(np.linalg.norm(R_dir @ H - R_U)),
        "orthogonality_err": float(np.linalg.norm(R_U.T @ R_U - np.eye(n))),
        "det_err": float(abs(np.linalg.det(R_U) - 1.0)),
    }


def realisability(rng, n: int, k_max: int = 6) -> list[dict]:
    """
    Proposition B, item 3.  Chains with a *prescribed common endpoint*
    but paths that wander through an increasing number of extra
    dimensions.  The semantic deficit must be constant along the family
    while the holonomy energy grows: the two data are independent.
    """
    v0 = np.zeros(n); v0[0] = 1.0
    vL = np.zeros(n); vL[0] = np.cos(0.5); vL[1] = np.sin(0.5)
    rows = []
    for k in range(k_max + 1):
        mids = []
        for j in range(k):
            m = np.zeros(n)
            m[0] = np.cos(0.9)
            m[2 + (j % (n - 2))] = np.sin(0.9)
            mids.append(m)
        V = [v0] + mids + [vL]
        H = holonomy(V)
        R_U, _ = loop_rotation(V)
        rows.append({
            "n_detours": k,
            "semantic_deficit": cosine_distance(V[0], V[-1]),
            "holonomy_energy": float(np.sqrt(sum(a * a for a in angle_spectrum(H)))),
            "loop_energy": float(np.sqrt(sum(a * a for a in angle_spectrum(R_U)))),
            "sylvester": list(signature(quadratic_form_matrix(R_U))),
        })
    return rows


def gauss_bonnet(rng, n_trials: int = 200) -> dict:
    """
    Proposition B, item 4, in the lowest dimension where it has classical
    content.  On the two-sphere the loop rotation is parallel transport
    along the geodesic polygon, so for a chain of three points the
    holonomy angle must equal the spherical excess of the geodesic
    triangle they span, which by Gauss-Bonnet is its area.

    One caveat is intrinsic and not a defect of the implementation: the
    angle recovered from the eigenvalues of a rotation is its principal
    value in (0, pi].  The holonomy therefore determines the area only
    modulo 2 pi and up to reflection, so the identity must be tested
    against the principal reduction of the excess.  For triangles of
    area at most pi -- that is, at most a hemisphere -- the reduction is
    the identity and the holonomy angle is the area outright.

    If the identification of the loop rotation with parallel transport
    were wrong, this equality would fail even for small triangles.
    """
    def interior_angle(p, q, r):
        u = unit(q - (q @ p) * p)
        w = unit(r - (r @ p) * p)
        return math.acos(max(-1.0, min(1.0, float(u @ w))))

    def principal(a: float) -> float:
        a = a % (2.0 * math.pi)
        return a if a <= math.pi else 2.0 * math.pi - a

    worst, worst_small, n_small, rows = 0.0, 0.0, 0, []
    for _ in range(n_trials):
        P = [unit(rng.normal(size=3)) for _ in range(3)]
        R_U, R_dir = loop_rotation(P)
        spec = angle_spectrum(R_dir.T @ R_U)
        hol = spec[0] if spec else 0.0
        excess = (interior_angle(P[0], P[1], P[2])
                  + interior_angle(P[1], P[2], P[0])
                  + interior_angle(P[2], P[0], P[1]) - math.pi)
        err = abs(principal(excess) - hol)
        worst = max(worst, err)
        if abs(excess) <= math.pi:            # at most a hemisphere
            n_small += 1
            worst_small = max(worst_small, abs(abs(excess) - hol))
        if len(rows) < 6:
            rows.append({"spherical_excess": float(excess),
                         "principal_excess": float(principal(excess)),
                         "holonomy_angle": float(hol), "abs_error": float(err)})
    return {"worst_abs_error": float(worst),
            "worst_abs_error_area_at_most_pi": float(worst_small),
            "n_triangles_area_at_most_pi": int(n_small),
            "n_trials": n_trials, "sample": rows}


def signature_blindness(rng, n: int = 64, n_trials: int = 200) -> dict:
    """
    Proposition A, operational consequence.  Among chains sharing one
    and the same Sylvester signature, how much do the angle spectra
    still differ?  If the signature were a fine invariant this spread
    would be small.
    """
    buckets: dict[tuple, list[tuple[float, float]]] = {}
    for _ in range(n_trials):
        L = int(rng.integers(2, 9))
        V = random_chain(rng, n, L)
        R_U, R_dir = loop_rotation(V)
        sig = signature(quadratic_form_matrix(R_U))
        energy = float(np.sqrt(sum(a * a for a in angle_spectrum(R_U))))
        eta = float(np.sqrt(sum(a * a for a in angle_spectrum(R_dir.T @ R_U))))
        buckets.setdefault(tuple(sig), []).append((energy, eta))
    out = {}
    for sig, vals in buckets.items():
        if len(vals) >= 5:
            en = [v[0] for v in vals]
            et = [v[1] for v in vals]
            out[str(list(sig))] = {
                "count": len(vals),
                "loop_energy_min": float(np.min(en)),
                "loop_energy_max": float(np.max(en)),
                "loop_energy_ratio": float(np.max(en) / max(np.min(en), 1e-12)),
                "holonomy_energy_min": float(np.min(et)),
                "holonomy_energy_max": float(np.max(et)),
            }
    return out


def main() -> None:
    run = Run("exp_geometry_v1")
    rng = np.random.default_rng(20260819)

    worst: dict[str, float] = {}
    per_dim = {}
    for n in DIMS:
        agg: dict[str, float] = {}
        for L in CHAIN_LENGTHS:
            for _ in range(N_TRIALS // len(CHAIN_LENGTHS)):
                res = check_propositions(rng, n, L)
                for k, v in res.items():
                    if k == "min_eig":
                        agg[k] = min(agg.get(k, 0.0), v)
                    else:
                        agg[k] = max(agg.get(k, 0.0), abs(v))
        per_dim[str(n)] = agg
        run.log(f"n = {n:4d}  min_eig = {agg['min_eig']:.3e}  "
                f"n_minus = {agg['n_minus']:.0f}  "
                f"|n+ - 2p| = {agg['n_plus_minus_2p']:.0f}  "
                f"fix_err = {agg['fix_err']:.3e}  "
                f"factorisation_err = {agg['factorisation_err']:.3e}")
        for k, v in agg.items():
            if k == "min_eig":
                worst[k] = min(worst.get(k, 0.0), v)
            else:
                worst[k] = max(worst.get(k, 0.0), v)

    run.log("")
    run.log("Proposition B, realisability: prescribed endpoint, growing detour")
    real_rows = realisability(rng, n=64)
    for r in real_rows:
        run.log(f"  detours={r['n_detours']}  delta={r['semantic_deficit']:.8f}  "
                f"eta={r['holonomy_energy']:.4f}  sylvester={r['sylvester']}")
    deltas = [r["semantic_deficit"] for r in real_rows]
    etas = [r["holonomy_energy"] for r in real_rows]
    delta_spread = float(max(deltas) - min(deltas))
    eta_spread = float(max(etas) - min(etas))
    run.log(f"  --> delta spread = {delta_spread:.3e}   eta spread = {eta_spread:.4f}")

    run.log("")
    run.log("Proposition B item 4, Gauss-Bonnet check on the two-sphere:")
    gb = gauss_bonnet(rng)
    for r in gb["sample"]:
        run.log(f"  excess = {r['spherical_excess']:9.6f}  principal = "
                f"{r['principal_excess']:9.6f}  holonomy = "
                f"{r['holonomy_angle']:9.6f}  error = {r['abs_error']:.2e}")
    run.log(f"  --> worst error over {gb['n_trials']} triangles: "
            f"{gb['worst_abs_error']:.2e}")
    run.log(f"  --> worst error over the {gb['n_triangles_area_at_most_pi']} "
            f"triangles of area <= pi: {gb['worst_abs_error_area_at_most_pi']:.2e}")

    run.log("")
    run.log("Proposition A, signature blindness (n = 64):")
    blind = signature_blindness(rng)
    for sig, st in sorted(blind.items()):
        run.log(f"  signature {sig:16s} count={st['count']:3d}  "
                f"loop energy in [{st['loop_energy_min']:.3f}, "
                f"{st['loop_energy_max']:.3f}]  ratio={st['loop_energy_ratio']:.2f}  "
                f"holonomy energy in [{st['holonomy_energy_min']:.3f}, "
                f"{st['holonomy_energy_max']:.3f}]")

    run.write_json("geometry.json", {
        "per_dimension": per_dim,
        "worst_case": worst,
        "realisability": real_rows,
        "gauss_bonnet": gb,
        "signature_blindness": blind,
        "settings": {"dims": DIMS, "trials_per_dim": N_TRIALS,
                     "chain_lengths": CHAIN_LENGTHS, "seed": 20260819},
    })

    tol = 1e-9
    passed = (worst["min_eig"] > -tol and worst["n_minus"] == 0
              and worst["n_plus_minus_2p"] == 0 and worst["fix_err"] < 1e-8
              and worst["delta_from_Rdir_err"] < 1e-8
              and worst["factorisation_err"] < 1e-8
              and worst["quadform_err"] < 1e-8
              and delta_spread < 1e-12 and eta_spread > 0.1
              and gb["worst_abs_error"] < 1e-8)

    run.log("")
    run.log(f"ALL CHECKS PASSED: {passed}")
    run.finish(
        conclusions={
            "propositions_A_B_verified": bool(passed),
            "worst_min_eigenvalue": worst["min_eig"],
            "worst_n_minus": worst["n_minus"],
            "worst_holonomy_fix_error": worst["fix_err"],
            "worst_factorisation_error": worst["factorisation_err"],
            "realisability_delta_spread": delta_spread,
            "realisability_eta_spread": eta_spread,
            "gauss_bonnet_worst_error": gb["worst_abs_error"],
            "kill_criterion_K1_excluded": bool(passed),
        },
        limitations=[
            "Numerical verification, not proof; the proofs are in "
            "3. Certificates/Article_LLW/.",
            "Chains are pseudo-random walks, not embeddings of real text; "
            "the propositions are basis-free so this is sufficient for them, "
            "but it says nothing about the empirical claim of the article.",
        ],
        inputs={"seed": 20260819, "dims": DIMS},
        command="python exp_geometry_v1.py",
    )


if __name__ == "__main__":
    main()
