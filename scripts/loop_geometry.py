"""
loop_geometry.py -- geometry of linguistic loops.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, sub-goals S2/S4/S8; Propositions A and B
Version  : v1
Purpose  : Minimal rotations, loop rotation, holonomy, angle spectra and
           the derived scalar indicators, following the formalism of
           Corradetti & Marrani, arXiv:2503.23311 (Eqs. 4.2-4.7), and
           the refinements proved in this article.
Input    : a chain of embedding vectors v_0, ..., v_L in R^n
Output   : R_U, R_direct, holonomy H, angle spectra, semantic deficit,
           holonomy energy, path lengths

What is new here relative to the source preprint
------------------------------------------------
  * `signature` is retained for reproduction of the preprint's
    construction, but Proposition A shows it is always of the form
    (2p, 0, n - 2p); `angle_spectrum` supersedes it.
  * `holonomy` implements the factorisation R_U = R_direct . H of
    Proposition B.  H fixes v_0, so it lives in SO(n-1), and the
    semantic deficit cannot see it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

Vec = np.ndarray


# --------------------------------------------------------------------- #
#  Basic metric utilities                                                #
# --------------------------------------------------------------------- #
def unit(x: Vec, eps: float = 1e-12) -> Vec:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x) + eps)


def cosine_distance(a: Vec, b: Vec, eps: float = 1e-12) -> float:
    """Eq. (2.3) of the source preprint: d = 1 - cos.  Zero iff parallel."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def path_length(V: Sequence[Vec]) -> float:
    """Sum of cosine distances between consecutive elements of the chain."""
    return float(sum(cosine_distance(V[i - 1], V[i]) for i in range(1, len(V))))


# --------------------------------------------------------------------- #
#  Minimal rotations, Eq. (4.3)                                          #
# --------------------------------------------------------------------- #
def minimal_rotation(x: Vec, y: Vec, eps: float = 1e-12) -> Vec:
    """
    The unique rotation carrying x_hat to y_hat and acting as the
    identity on the orthogonal complement of span(x, y):

        R = I + A + (1 + x_hat . y_hat)^{-1} A^2,
        A = y_hat x_hat^T - x_hat y_hat^T.

    Two degenerate cases are handled explicitly.  Parallel vectors give
    the identity.  Antipodal vectors make the formula singular, and we
    return the rotation by pi in a plane spanned by x_hat and an
    arbitrary orthogonal unit vector; for embeddings of natural-language
    text this branch is never taken in practice, but leaving it
    undefined would make the function partial.
    """
    xh, yh = unit(x), unit(y)
    n = xh.shape[0]
    ident = np.eye(n)
    c = float(np.clip(xh @ yh, -1.0, 1.0))
    if c > 1.0 - eps:
        return ident
    if c < -1.0 + eps:
        k = int(np.argmin(np.abs(xh)))
        e = np.zeros(n)
        e[k] = 1.0
        w = unit(e - (e @ xh) * xh)
        return ident - 2.0 * np.outer(xh, xh) - 2.0 * np.outer(w, w)
    A = np.outer(yh, xh) - np.outer(xh, yh)
    return ident + A + (1.0 / (1.0 + c)) * (A @ A)


def loop_rotation(V: Sequence[Vec]) -> tuple[Vec, Vec]:
    """
    Return (R_U, R_direct).

    R_U is the ordered product of the minimal rotations between
    consecutive states, i.e. Eq. (4.4): it remembers the whole path.
    R_direct is the single minimal rotation from v_0 to v_L: it
    remembers only the endpoints.
    """
    V = [np.asarray(v, dtype=np.float64) for v in V]
    R = np.eye(V[0].shape[0])
    for i in range(1, len(V)):
        R = minimal_rotation(V[i - 1], V[i]) @ R
    return R, minimal_rotation(V[0], V[-1])


def holonomy(V: Sequence[Vec]) -> Vec:
    """
    H = R_direct^{-1} R_U, the holonomy of the loop at base point v_0
    (Proposition B).  H fixes v_0 and therefore lies in the stabiliser
    SO(n-1).  It is exactly the part of the loop that the semantic
    deficit discards.
    """
    R_U, R_dir = loop_rotation(V)
    return R_dir.T @ R_U          # R_dir orthogonal, so the inverse is the transpose


# --------------------------------------------------------------------- #
#  Spectral invariants                                                   #
# --------------------------------------------------------------------- #
def angle_spectrum(R: Vec, tol: float = 1e-8) -> list[float]:
    """
    The rotation angles of R in SO(n), one per non-trivial 2-plane,
    sorted decreasingly.  By Proposition A this is the complete O(n)
    invariant of the loop, and it strictly refines the signature.

    Two details matter for the identity n+ = 2p to hold numerically.

    First, the threshold.  A plane counts as non-trivial for the
    quadratic form when its eigenvalue 1 - cos(theta) exceeds `tol`, so
    the angle threshold must be acos(1 - tol) and not `tol` itself;
    using `tol` on the angle would admit planes that the signature,
    computed with the same tolerance, legitimately discards.

    Second, the angle pi.  The eigenvalues of a plane rotated by pi are
    both equal to -1, so numpy reports the argument +pi twice for a
    single plane; those occurrences must be halved.
    """
    w = np.linalg.eigvals(R)
    ang = [float(a) for a in np.angle(w)]
    theta_min = math.acos(min(1.0, max(-1.0, 1.0 - tol)))
    pi_tol = 1e-7
    n_near_pi = sum(1 for a in ang if abs(abs(a) - math.pi) <= pi_tol)
    out = [a for a in ang if a > theta_min and abs(abs(a) - math.pi) > pi_tol]
    out += [math.pi] * (n_near_pi // 2)
    return sorted(out, reverse=True)


def rotation_energy(R: Vec) -> float:
    """Euclidean norm of the angle spectrum."""
    return float(math.sqrt(sum(a * a for a in angle_spectrum(R))))


def signature(M: Vec, tol: float = 1e-8) -> tuple[int, int, int]:
    """
    Sylvester signature (n+, n-, n0) of the symmetric part of M.
    Kept for reproduction of the source construction only: Proposition A
    shows that for M = I - sym(R_U) the middle entry is always zero.
    """
    w = np.linalg.eigvalsh((M + M.T) / 2.0)
    return (int((w > tol).sum()), int((w < -tol).sum()), int((np.abs(w) <= tol).sum()))


def quadratic_form_matrix(R_U: Vec) -> Vec:
    """The representing matrix I_n - sym(R_U) of Eq. (4.6)-(4.7)."""
    n = R_U.shape[0]
    return np.eye(n) - 0.5 * (R_U + R_U.T)


# --------------------------------------------------------------------- #
#  Aggregate description of one chain                                    #
# --------------------------------------------------------------------- #
@dataclass
class LoopInvariants:
    n: int
    length: int                       # number of transformations L
    semantic_deficit: float           # delta, Eq. (3.5)
    semantic_path_length: float       # cumulative motion in the embedding
    loop_energy: float                # || Theta(R_U) ||_2
    holonomy_energy: float            # eta = || Theta(H) ||_2, Proposition B
    n_planes: int                     # p, the number of non-trivial planes
    sylvester: tuple                  # (n+, n-, n0); n- is always 0
    detour_ratio: float               # eta / (delta + reg): path per unit meaning
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["sylvester"] = list(self.sylvester)
        return d


def loop_invariants(V: Sequence[Vec], reg: float = 1e-3) -> LoopInvariants:
    """Compute every geometric invariant of one embedded chain."""
    V = [np.asarray(v, dtype=np.float64) for v in V]
    if len(V) < 2:
        raise ValueError("a chain needs at least two states")
    n = V[0].shape[0]
    R_U, _ = loop_rotation(V)
    H = holonomy(V)
    theta_U = angle_spectrum(R_U)
    delta = cosine_distance(V[0], V[-1])
    eta = float(math.sqrt(sum(a * a for a in angle_spectrum(H))))
    return LoopInvariants(
        n=n,
        length=len(V) - 1,
        semantic_deficit=delta,
        semantic_path_length=path_length(V),
        loop_energy=float(math.sqrt(sum(a * a for a in theta_U))),
        holonomy_energy=eta,
        n_planes=len(theta_U),
        sylvester=signature(quadratic_form_matrix(R_U)),
        detour_ratio=float(eta / (delta + reg)),
    )


# --------------------------------------------------------------------- #
#  Surface channel: what the detector actually sees                      #
# --------------------------------------------------------------------- #
def retention_rate(tokens_a: Sequence[int], tokens_b: Sequence[int]) -> float:
    """
    Fraction of the original tokens that survive, in order, in the
    attacked text.  Computed by longest-common-subsequence matching on
    the *tokenizer's own* token ids -- not on words and not on character
    n-grams, because the seeding context of a green-list detector is
    defined on token ids.  This is the quantity rho of Theorem C.
    """
    from difflib import SequenceMatcher
    a, b = [int(t) for t in tokens_a], [int(t) for t in tokens_b]
    if not a:
        return 0.0
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return float(matched / len(a))


def context_retention_rate(tokens_a: Sequence[int], tokens_b: Sequence[int],
                           h: int) -> float:
    """
    Fraction of original positions that survive *together with their h
    preceding tokens*, i.e. the fraction of positions whose seeding
    context is intact.  Theorem C predicts that this, rather than the
    plain retention rate, is what the green-list detector responds to;
    under the i.i.d. model it equals rho^(h+1).
    """
    from difflib import SequenceMatcher
    a, b = [int(t) for t in tokens_a], [int(t) for t in tokens_b]
    if len(a) <= h:
        return 0.0
    survived = np.zeros(len(a), dtype=bool)
    for block in SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        survived[block.a:block.a + block.size] = True
    ok = 0
    for t in range(h, len(a)):
        if survived[t] and all(survived[t - j] for j in range(1, h + 1)):
            ok += 1
    return float(ok / (len(a) - h))
