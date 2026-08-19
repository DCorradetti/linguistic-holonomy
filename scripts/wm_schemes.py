"""
wm_schemes.py -- watermark schemes and exact detectors.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, sub-goals S5/S6, Theorem C
Version  : v1
Purpose  : Reference implementations of three text-watermarking schemes
           together with their exact detectors.  The keys are ours, so
           every detection statistic in this project is ground truth
           rather than an estimate.
Input    : token id sequences (numpy int arrays) and a vocabulary size
Output   : biased logits at generation time; z-scores / p-values at
           detection time

Schemes
-------
  KGW      green-list with context width h >= 1   (Kirchenbauer et al.,
           ICML 2023).  Logit bias `delta_bias` on a pseudo-random green
           subset of the vocabulary of relative size `gamma`, the subset
           being seeded by the h preceding tokens.
  UNIGRAM  the h = 0 member of the same family (Zhao et al., ICLR 2024):
           one fixed green list for the whole text.
  EXP      Aaronson-style exponential / Gumbel sampling, the
           distortion-free family (cf. Kuditipudi et al., TMLR 2024).
           A context-seeded uniform vector xi in [0,1]^V selects
           argmax_i xi_i^(1/p_i); the marginal law of the emitted token
           is exactly p.

Design notes
------------
Green membership is decided by a per-token integer hash rather than by
materialising a vocabulary permutation.  The two constructions are
distributionally equivalent, but the hash version is O(V) with a small
constant and, more importantly, is *stateless*: the detector recomputes
membership from the context alone, exactly as a real detector must.

The mixer is splitmix64, vectorised over numpy uint64.  All arithmetic
is modular by construction, which is what we want from a hash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

# --------------------------------------------------------------------- #
#  splitmix64, vectorised                                                #
# --------------------------------------------------------------------- #
_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)
_S1, _S2, _S3 = np.uint64(30), np.uint64(27), np.uint64(31)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser.  `x` must be a uint64 array."""
    with np.errstate(over="ignore"):
        z = (x + _M1).astype(np.uint64)
        z = ((z ^ (z >> _S1)) * _M2).astype(np.uint64)
        z = ((z ^ (z >> _S2)) * _M3).astype(np.uint64)
        return (z ^ (z >> _S3)).astype(np.uint64)


_UNIT = np.float64(1.0) / np.float64(2.0 ** 64)


def _uniform_from(seed: np.uint64, ids: np.ndarray) -> np.ndarray:
    """Uniform(0,1) sample per vocabulary id, deterministic in (seed, id)."""
    with np.errstate(over="ignore"):
        mixed = splitmix64(seed ^ splitmix64(ids.astype(np.uint64)))
    return mixed.astype(np.float64) * _UNIT


def context_seed(key: int, context: Sequence[int], h: int) -> np.uint64:
    """
    Seed derived from the key and the h most recent tokens.

    For h = 0 the seed is the key alone, which is the Unigram scheme.
    For h >= 1 we use a position-weighted sum of the context, mixed
    through splitmix64; this is the `additive` variant of Kirchenbauer
    et al., chosen because it is symmetric under no permutation of the
    context and is therefore the least forgiving one to attack -- the
    conservative choice for our purposes.
    """
    acc = np.uint64(key)
    if h > 0:
        ctx = list(context)[-h:]
        with np.errstate(over="ignore"):
            for i, t in enumerate(ctx):
                acc = np.uint64(acc + np.uint64((i + 1)) * np.uint64(int(t) + 1))
    return splitmix64(np.array([acc], dtype=np.uint64))[0]


# --------------------------------------------------------------------- #
#  Scheme configuration                                                  #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class WMConfig:
    name: str               # "KGW" | "UNIGRAM" | "EXP"
    key: int
    vocab_size: int
    h: int = 1              # context width
    gamma: float = 0.25     # green fraction (green-list family only)
    delta_bias: float = 2.0  # logit bias (green-list family only)

    @property
    def family(self) -> str:
        return "exp" if self.name.upper() == "EXP" else "greenlist"


# --------------------------------------------------------------------- #
#  Green-list family                                                     #
# --------------------------------------------------------------------- #
def green_mask(cfg: WMConfig, context: Sequence[int]) -> np.ndarray:
    """Boolean mask over the vocabulary: True where the token is green."""
    ids = np.arange(cfg.vocab_size, dtype=np.uint64)
    u = _uniform_from(context_seed(cfg.key, context, cfg.h), ids)
    return u < cfg.gamma


def is_green(cfg: WMConfig, context: Sequence[int], token: int) -> bool:
    """Green membership of a single token; what the detector actually needs."""
    u = _uniform_from(context_seed(cfg.key, context, cfg.h),
                      np.array([token], dtype=np.uint64))[0]
    return bool(u < cfg.gamma)


def apply_greenlist_bias(cfg: WMConfig, logits: np.ndarray,
                         context: Sequence[int]) -> np.ndarray:
    """Add `delta_bias` to the logits of green tokens."""
    out = np.array(logits, dtype=np.float64, copy=True)
    out[green_mask(cfg, context)] += cfg.delta_bias
    return out


def detect_greenlist(cfg: WMConfig, tokens: Sequence[int]) -> dict:
    """
    Standard z-test on the green-token count.

        z = (|s|_G - gamma * T) / sqrt(T * gamma * (1 - gamma))

    Scoring starts at position h, the first position for which a full
    seeding context exists.
    """
    toks = [int(t) for t in tokens]
    start = cfg.h
    scored = len(toks) - start
    if scored <= 0:
        return {"z": 0.0, "p": 1.0, "n_scored": 0, "n_green": 0, "green_frac": float("nan")}
    n_green = sum(1 for t in range(start, len(toks))
                  if is_green(cfg, toks[max(0, t - cfg.h):t], toks[t]))
    g = cfg.gamma
    z = (n_green - g * scored) / math.sqrt(scored * g * (1.0 - g))
    return {
        "z": float(z),
        "p": float(norm.sf(z)),
        "n_scored": int(scored),
        "n_green": int(n_green),
        "green_frac": float(n_green / scored),
    }


# --------------------------------------------------------------------- #
#  Exponential / Gumbel family (distortion-free)                         #
# --------------------------------------------------------------------- #
def exp_sample(cfg: WMConfig, probs: np.ndarray,
               context: Sequence[int]) -> int:
    """
    Aaronson's rule: emit argmax_i xi_i^(1/p_i), equivalently
    argmax_i log(xi_i) / p_i.  Marginally over the key the emitted token
    is distributed exactly as p, which is what `distortion-free` means.
    """
    ids = np.arange(cfg.vocab_size, dtype=np.uint64)
    xi = _uniform_from(context_seed(cfg.key, context, cfg.h), ids)
    xi = np.clip(xi, 1e-12, 1.0 - 1e-12)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    return int(np.argmax(np.log(xi) / p))


def detect_exp(cfg: WMConfig, tokens: Sequence[int]) -> dict:
    """
    Alignment-free detector for the context-seeded EXP scheme.

    For each scored position recompute xi for the observed token and
    accumulate  r_t = -log(1 - xi_{y_t}).  Under the null (text not
    produced with this key) the xi are Uniform(0,1) and the r_t are
    i.i.d. Exp(1), so  S = sum r_t ~ Gamma(T, 1).  Watermarked text
    pushes xi towards 1 and inflates S.

    We report the exact Gamma p-value and also its normal-equivalent
    z-score, so that the statistic is on the same scale as the
    green-list z and the two families can be compared in one plot.
    """
    toks = [int(t) for t in tokens]
    start = cfg.h
    scored = len(toks) - start
    if scored <= 0:
        return {"z": 0.0, "p": 1.0, "n_scored": 0, "S": 0.0}
    S = 0.0
    for t in range(start, len(toks)):
        xi = _uniform_from(context_seed(cfg.key, toks[max(0, t - cfg.h):t], cfg.h),
                           np.array([toks[t]], dtype=np.uint64))[0]
        S += -math.log(max(1.0 - xi, 1e-300))
    p = float(gamma_dist.sf(S, a=scored, scale=1.0))
    p = min(max(p, 1e-300), 1.0)
    return {
        "z": float(norm.isf(p)),
        "p": p,
        "n_scored": int(scored),
        "S": float(S),
    }


# --------------------------------------------------------------------- #
#  Uniform front end                                                     #
# --------------------------------------------------------------------- #
def detect(cfg: WMConfig, tokens: Sequence[int]) -> dict:
    """Dispatch to the detector of the configured family."""
    if cfg.family == "greenlist":
        return detect_greenlist(cfg, tokens)
    return detect_exp(cfg, tokens)


def default_configs(vocab_size: int, key: int = 20260819) -> list[WMConfig]:
    """
    The three schemes of the Minimum Publishable Unit.

    KGW and UNIGRAM differ *only* in the context width h, which is
    precisely the parameter Theorem C predicts to govern the decay
    exponent.  Holding everything else fixed makes the comparison a
    clean test of the theorem rather than a comparison of two papers.
    """
    return [
        WMConfig("KGW", key, vocab_size, h=1, gamma=0.25, delta_bias=2.0),
        WMConfig("UNIGRAM", key, vocab_size, h=0, gamma=0.25, delta_bias=2.0),
        WMConfig("EXP", key, vocab_size, h=1),
    ]
