# Certificate: Degeneracy of the Sylvester signature of a linguistic loop, v1

## Statement

Let `R` be an element of `SO(n)` and let

```
    R* := (R + R^T) / 2 ,        M := I_n - R* .
```

Let `theta_1, ..., theta_p` in `(0, pi]` be the rotation angles of `R`,
one for each non-trivial invariant 2-plane. Then:

**(A1)** `M` is symmetric and positive semidefinite.

**(A2)** The spectrum of `M` is

```
    { 1 - cos(theta_k) , each with multiplicity 2 }  union  { 0 with multiplicity n - 2p } .
```

**(A3)** Consequently the Sylvester signature of the quadratic form
`Q(v) = v^T M v` is

```
    sign(M) = (2p, 0, n - 2p) ,
```

so the negative index vanishes identically, `rank(M) = 2p`, and the
signature is a function of the single integer `p`. Sylvester's law of
inertia contributes no information beyond the rank.

**(A4)** The signature is a strictly coarser invariant than the
conjugacy class of `R` in `SO(n)`: the map `R -> sign(M)` factors
through `p`, whereas the conjugacy class is the full multiset
`Theta(R) = {theta_1, ..., theta_p}`. Any two rotations with the same
number of non-trivial planes and different angles are separated by
`Theta` and confused by `sign(M)`.

**(A5)** (Invariance group.) Write `Psi(V)` for the loop rotation
`R_U` built from a chain `V = (v_0, ..., v_L)` by Eq. (4.4) of [CM25].
For `A` in `O(n)` one has

```
    Psi(A v_0, ..., A v_L) = A Psi(v_0, ..., v_L) A^T ,
```

hence `Theta` is an invariant of the chain under the orthogonal group.
No analogous identity holds for general `A` in `GL(n,R)`. The signature
of a *fixed* quadratic form is of course invariant under congruence;
but the assignment chain `-> M` is not `GL(n,R)`-equivariant, because
`R_U` is constructed from the cosine metric. The natural invariance
group of the construction is therefore `O(n)`, not `GL(n,R)`.

## Notation

`I_n` is the identity; `R(theta)` denotes the planar rotation matrix
with entries `cos(theta), -sin(theta); sin(theta), cos(theta)`.
`[CM25]` is Corradetti and Marrani, arXiv:2503.23311v1, whose Eqs.
(4.4)-(4.7) define `R_U` and the representing matrix `I_n - R_U^*` of
the quadratic form associated with the semantic deficit.

## Dependencies

- Real normal form of a special orthogonal matrix (standard; see e.g.
  Artin, *Algebra*, Ch. 5, or any text on the classification of
  orthogonal transformations).
- Spectral theorem for real symmetric matrices.
- No result from this project.

## Proof

**(A1) and (A2).** By the real normal form, there exists `Q` in `O(n)`
with

```
    Q^T R Q = diag( R(theta_1), ..., R(theta_p), I_{n - 2p} ) ,
```

with `theta_k` in `(0, pi]`. Two remarks on the range of the angles.
An orthogonal matrix may have the eigenvalue `-1`; in `SO(n)` such
eigenvalues occur with even multiplicity, and each pair is a rotation
by `pi` in a 2-plane, i.e. a block `R(pi)`. Angles are therefore taken
in the half-open interval `(0, pi]`, and the value `pi` is admissible.
Conversely `theta = 0` contributes a trivial block and is absorbed into
`I_{n - 2p}`.

Since `R(theta)^T = R(-theta)`, the symmetric part of a block is

```
    ( R(theta) + R(theta)^T ) / 2 = cos(theta) I_2 ,
```

so that

```
    Q^T M Q = diag( (1 - cos theta_1) I_2 , ..., (1 - cos theta_p) I_2 , 0 ) .    (*)
```

Because `Q` is orthogonal, (*) is simultaneously a congruence and a
similarity; hence the displayed diagonal entries are the eigenvalues of
`M`, which proves (A2). For `theta_k` in `(0, pi]` we have
`cos(theta_k) < 1`, so `1 - cos(theta_k) > 0`, and all eigenvalues of
`M` are non-negative, which proves (A1).

**(A3).** Immediate from (A2): the number of strictly positive
eigenvalues is `2p`, the number of strictly negative eigenvalues is `0`,
and the kernel has dimension `n - 2p`.

**(A4).** Take `n >= 4`, `p = 2`, and the two rotations
`R = R(a) + R(b)` and `R' = R(a') + R(b')` in block form with
`{a, b}` and `{a', b'}` distinct multisets of angles in `(0, pi]`. Both
give `sign(M) = (4, 0, n - 4)`, while `Theta(R) != Theta(R')`, and `R`
and `R'` are not conjugate in `SO(n)` precisely because their angle
multisets differ. Hence `sign` is strictly coarser.

**(A5).** For `A` in `O(n)` and any non-zero `x`, `unit(Ax) = A unit(x)`.
Writing `xh = unit(x)`, `yh = unit(y)`, the minimal rotation of Eq. (4.3)
is a polynomial in `S = yh xh^T - xh yh^T` with coefficients depending
only on `xh . yh`. Since `A` is orthogonal,

```
    (A yh)(A xh)^T - (A xh)(A yh)^T = A S A^T ,
    (A xh) . (A yh) = xh . yh ,
```

so the minimal rotation transforms by conjugation, and the ordered
product defining `R_U` does too. Conjugation preserves the multiset of
eigenvalues, hence `Theta`. For `A` in `GL(n,R)` neither identity
survives: `unit(Ax)` is not `A unit(x)`, and `xh . yh` is not preserved,
so `R_U` is replaced by an unrelated rotation. []

## Verification of hypotheses

- `R_U` as defined by Eq. (4.4) of [CM25] is a product of minimal
  rotations, each of which is a rotation in a 2-plane and therefore lies
  in `SO(n)`; the product lies in `SO(n)`. The hypothesis `R` in `SO(n)`
  of the statement is thus satisfied by the object to which we apply it.
  This is checked numerically as `orthogonality_err` and `det_err` in
  the supporting run.
- The identity `Q_U(v0h, v0h) = delta_U` of Eq. (4.5) is checked
  numerically as `quadform_err`, so the matrix `M` we analyse is the
  representing matrix of the semantic deficit and not a different object.

## Computational support

Run `run_exp_geometry_v1_20260819T135822` under
`7. Results/Article_LLW/`, produced by `2. Scripts/Article_LLW/exp_geometry_v1.py`.

Over 300 random chains per dimension, with `n` in `{8, 16, 64, 384}` and
chain lengths in `{2, 3, 4, 6, 8}`:

| quantity | worst case observed |
|---|---|
| minimum eigenvalue of `M` | `-6.1e-15` |
| negative index `n_-` | `0` |
| `abs(n_+ - 2p)` | `0` |
| `abs(v^T M v - delta)` | `3.7e-12` |
| `norm(R_U^T R_U - I)` | `3.1e-12` |

The blindness of the signature is exhibited directly: among 200 random
chains in dimension 64, within the single signature class `(8, 0, 56)`
the loop energy `norm(Theta(R_U))` ranges over `[0.823, 1.561]` and the
holonomy energy over `[0.269, 0.794]`. The signature is constant on
these families; the spectral invariant is not.

A numerical subtlety that the run exposed and that the manuscript should
record in a remark: the identity `n_+ = 2p` holds only if the two counts
are taken with matched tolerances. Counting a plane as non-trivial when
`theta > tol` while counting a positive eigenvalue when
`1 - cos(theta) > tol` admits planes with `tol < theta < sqrt(2 tol)` to
the first count and not to the second. The angle threshold must be
`arccos(1 - tol)`. Additionally, a plane rotated by exactly `pi` has both
its eigenvalues equal to `-1`, so a naive argument-based count reports
the angle `pi` twice for one plane.

## Failure modes / limitations

- The statement is about `SO(n)` and says nothing about the *linguistic*
  adequacy of `R_U` as a description of a chain of reformulations.
- (A4) shows the signature is coarser; it does not show that the
  signature is useless. `p` is a genuine, if weak, invariant, and the
  manuscript should present the result as a refinement rather than a
  refutation of [CM25].
- (A5) concerns the invariance group of the construction. It does not
  contradict the assertion in [CM25] that the signature of a quadratic
  form is a `GL(n,R)`-invariant, which is true; it observes that the
  quadratic form itself is not a `GL(n,R)`-natural function of the chain.

## Public-manuscript status

Included in the main draft, Section 3, in full. The proof is four lines
once the normal form is invoked, and there is no reason to relegate it
to an appendix or to a supplement.

## Changelog

- **v1** (2026-08-19). First issue. Statement, proof, `O(n)` versus
  `GL(n,R)` remark, and the tolerance subtlety discovered during
  numerical verification.
