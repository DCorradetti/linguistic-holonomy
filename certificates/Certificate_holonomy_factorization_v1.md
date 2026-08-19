# Certificate: Endpoint-path factorization of a linguistic loop, v1

## Statement

Let `V = (v_0, v_1, ..., v_L)` be a chain of non-zero vectors of `R^n`,
`n >= 3`, with no two consecutive members antipodal. Write `vh_i` for
`v_i / norm(v_i)`, let `Rmin(x, y)` be the minimal rotation of
Eq. (4.3) of [CM25], and set

```
    R_U    := Rmin(v_{L-1}, v_L) ... Rmin(v_0, v_1)      (Eq. 4.4)
    R_dir  := Rmin(v_0, v_L)
    H      := R_dir^{-1} R_U .
```

**(B0) (Parallel transport.)** `Rmin(x, y)` is the parallel transport of
the round metric of the unit sphere `S^{n-1}` along the minimising
geodesic from `xh` to `yh`, extended to `R^n` as the ambient rotation
that fixes `span(x, y)^perp` pointwise. Consequently `R_U` is parallel
transport along the geodesic polygon through `vh_0, ..., vh_L`, and `H`
is the holonomy, based at `vh_0`, of the closed geodesic polygon
obtained by closing that path with the geodesic from `vh_L` back to
`vh_0`.

**(B1)** `H vh_0 = vh_0`, so `H` lies in the stabiliser of `vh_0` in
`SO(n)`, which is isomorphic to `SO(n-1)`, and

```
    R_U = R_dir . H
```

canonically.

**(B2)** The semantic deficit satisfies

```
    delta_U = 1 - <vh_0, vh_L> = 1 - <vh_0, R_dir vh_0> ,
```

a function of `R_dir` alone. It is constant on the fibres of the
factorisation, hence blind to `H`.

**(B3) (Independence.)** For every `u` in `S^{n-1}` and every `H_0` in
`Stab(vh_0) = SO(n-1)` there exists a chain `V` with `vh_0` as first
element, `u` as last element and holonomy exactly `H_0`. The pair
`(delta, H)` is therefore free: the endpoint datum and the path datum
constrain each other in no way.

**(B4) (Gauss-Bonnet.)** For `n = 3` and `L = 2` the holonomy is a
rotation of the tangent plane at `vh_0` by the spherical excess of the
geodesic triangle `vh_0 vh_1 vh_2`, that is by its area. Since the
angle of a rotation is recovered from its eigenvalues only as a
principal value in `(0, pi]`, the holonomy determines the area outright
when the triangle covers at most a hemisphere, and modulo `2 pi` and
reflection in general.

## Notation

`Stab(w)` is the subgroup of `SO(n)` fixing the vector `w`. `[CM25]` is
Corradetti and Marrani, arXiv:2503.23311v1. `Hol(M, p)` is the
Riemannian holonomy group of `M` at `p`.

## Dependencies

- Parallel transport along geodesics of the round sphere; see do Carmo,
  *Riemannian Geometry*, Ch. 2, and Spivak, *A Comprehensive
  Introduction to Differential Geometry*, Vol. 1 -- both already cited
  in [CM25] as references [4] and [16].
- The holonomy group of the round `S^m`, `m >= 2`, is `SO(m)`. The
  sphere is a simply connected Riemannian symmetric space, so its
  holonomy coincides with the isotropy representation, which is the
  standard action of `SO(m)` on the tangent space. We give a
  self-contained constructive argument below and do not rely on the
  classification.
- Certificate_signature_degeneracy_v1 for the normal-form facts used in
  (B4).

## Proof

**(B0).** Let `xh, yh` be distinct, non-antipodal unit vectors and
`P = span(xh, yh)`. The minimising geodesic joining them is the arc of
the great circle `S^{n-1} intersect P`. A vector field `W` along a
geodesic `c` of the round sphere is parallel exactly when its ambient
derivative is normal to the sphere, i.e. `W' = -<W, c'> c`. Decompose
`W = W_P + W_perp` along `P` and `P^perp`. The component `W_perp` is
constant, since `c` and `c'` lie in `P`; the component `W_P` rotates
inside `P` by the arclength travelled. Hence transport from `xh` to
`yh` is rotation inside `P` by the angle `theta = arccos(<xh, yh>)`,
together with the identity on `P^perp`.

That is precisely `Rmin(x, y)`: writing `A = yh xh^T - xh yh^T` one has
`A P^perp = 0`, so `Rmin(x, y)` is the identity on `P^perp`, and on `P`
it carries `xh` to `yh` while preserving orientation and the metric, so
it is the rotation by `theta`. The claim about `R_U` follows by
composing, and the claim about `H` from (B1) below.

**(B1).** By construction `Rmin(v_{i-1}, v_i) vh_{i-1} = vh_i`. An
induction on `i` gives `R_U vh_0 = vh_L`. Also `R_dir vh_0 = vh_L` by
definition. Since `R_dir` is orthogonal, `R_dir^{-1} = R_dir^T` and

```
    H vh_0 = R_dir^T R_U vh_0 = R_dir^T vh_L = vh_0 .
```

Thus `H` fixes `vh_0`, so it preserves the hyperplane `vh_0^perp` and
restricts there to an element of `SO(n-1)`; the stabiliser of a unit
vector in `SO(n)` is isomorphic to `SO(n-1)` by this restriction. The
identity `R_U = R_dir H` is the definition of `H` rearranged.

**(B2).** With `d` the cosine distance of Eq. (2.3),
`delta_U = 1 - <vh_0, vh_L>`, and `vh_L = R_dir vh_0`. Two chains with
the same first and last element have the same `R_dir`, hence the same
`delta_U`, whatever their intermediate states; and `H` may be arbitrary
among such chains by (B3).

**(B3).** It suffices to realise any `H_0` in `Stab(vh_0)` as the
holonomy of a *closed* geodesic polygon based at `vh_0`; appending to
such a polygon the single geodesic leg from `vh_0` to `u` produces a
chain whose loop rotation is `Rmin(vh_0, u) H_0`, i.e. whose
factorisation has direct part `Rmin(vh_0, u)` and holonomy `H_0`,
while its endpoint is `u`.

Fix a 2-plane `Q` contained in `vh_0^perp` and an angle `alpha` in
`(0, 2 pi)`. Let `W = span(vh_0) + Q`, a three-dimensional subspace,
and let `Sigma = S^{n-1} intersect W`, a totally geodesic 2-sphere
containing `vh_0` whose tangent space at `vh_0` is `Q`. Choose a
geodesic triangle in `Sigma` with vertex `vh_0` and area `alpha`; this
is possible because the area of a geodesic triangle on the unit
2-sphere ranges over the whole interval `(0, 2 pi)`. Transport around a
loop contained in a totally geodesic submanifold is the transport
computed inside that submanifold, extended by the identity on the
normal directions; by (B4) the resulting holonomy is the rotation of
`Q` by `alpha`, and the identity on `Q^perp intersect vh_0^perp`.

Every element of `SO(n-1)` is a product of at most `floor((n-1)/2)`
such plane rotations, by the normal form recalled in
Certificate_signature_degeneracy_v1. Concatenating the corresponding
geodesic polygons -- each of which begins and ends at `vh_0` --
realises the product. This proves (B3), and incidentally re-proves that
`Hol(S^{n-1}, vh_0) = SO(n-1)` without invoking the classification of
symmetric spaces.

**(B4).** For `n = 3` the holonomy of a closed loop on the unit sphere
is, by the Gauss-Bonnet theorem with Gaussian curvature `1`, the
rotation of the tangent plane at the base point by the enclosed area;
for a geodesic triangle the area equals the spherical excess
`A + B + C - pi`. The statement about principal values follows from the
fact that the angles of a rotation are recovered from the arguments of
its eigenvalues, which lie in `(-pi, pi]`. []

## Verification of hypotheses

- *Non-antipodal consecutive states.* Required for `Rmin` to be
  single-valued; for embeddings of natural-language text, cosine
  similarities are bounded well away from `-1`, and the pipeline logs
  the minimum observed similarity so that the hypothesis is monitored
  rather than assumed.
- *Non-zero vectors.* Embedding maps do not return the zero vector in
  practice; the code raises rather than silently normalising.
- *`n >= 3`.* Below this the stabiliser is trivial and the statement,
  while true, is vacuous. All embedders used in the article have
  `n >= 384`.

## Computational support

Run `run_exp_geometry_v1_20260819T135822` under `7. Results/Article_LLW/`,
produced by `2. Scripts/Article_LLW/exp_geometry_v1.py`.

| check | worst case over 300 chains per dimension, `n` in `{8, 16, 64, 384}` |
|---|---|
| `norm(H vh_0 - vh_0)` | `3.3e-12` |
| `norm(R_dir H - R_U)` | `5.9e-12` |
| `abs(delta - (1 - <vh_0, R_dir vh_0>))` | `3.7e-12` |

Independence (B3) is exhibited directly: a family of chains with a
*prescribed common endpoint* and paths wandering through an increasing
number of extra dimensions gives a semantic deficit constant to the
last recorded digit, `0.12241744`, with spread exactly `0`, while the
holonomy energy runs from `0` to `1.0465`.

The Gauss-Bonnet identity (B4) is verified on random geodesic triangles
of the two-sphere: for triangles of area at most `pi` the holonomy
angle and the spherical excess agree to better than `1e-11`. Triangles
larger than a hemisphere agree only after reduction to the principal
branch, exactly as the statement says; this discrepancy was found by
the run and is the reason (B4) carries its caveat.

## Failure modes / limitations

- (B3) is a statement about which pairs `(delta, H)` are *realisable by
  some chain of vectors*. It does not claim that every such chain is
  realisable by an actual sequence of meaning-preserving rewritings of
  a text. The empirical part of the article must establish that real
  transformation chains populate a non-degenerate region of this space,
  and cannot take it for granted.
- The identification (B0) is with the Levi-Civita connection of the
  *round* metric. If one were to replace the cosine distance by a
  learned or anisotropic metric on the embedding space, `Rmin` would no
  longer be parallel transport and the holonomy interpretation would
  fail.
- Nothing here says that `H` is the *right* summary of the path. It
  says that `H` is exactly what `delta` discards. The scalar reduction
  `eta = norm(Theta(H))` used downstream is a choice, and a lossy one.

## Public-manuscript status

Included in the main draft, Section 4, in full, together with (B0) as a
lemma and (B4) as a corollary. The Gauss-Bonnet corollary is what makes
the invariant concrete for a reader without a Lie-theoretic background,
and should not be cut.

## Changelog

- **v1** (2026-08-19). First issue. Includes the identification of the
  loop rotation with parallel transport, which upgrades the Wilson-loop
  analogy of [CM25] from metaphor to theorem, and the constructive proof
  of realisability that avoids appealing to Berger's classification.
