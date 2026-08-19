"""
exp_indicators_v1.py -- geometric invariants of the real chains.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, sub-goal S8; Propositions A and B applied to data
Version  : v1
Purpose  : Embed every waypoint of every transformation chain produced by
           `exp_corpus_v1.py`, including the waypoints in the pivot
           languages, and compute the loop invariants: semantic deficit,
           path length, loop energy, holonomy energy, number of
           non-trivial planes, Sylvester signature.
Input    : the newest run_exp_corpus_v1_* directory under `7. Results/`
           (or an explicit path via --corpus)
Output   : run_exp_indicators_v1_<stamp>/indicators.json

The embedder must be multilingual: the intermediate waypoints of a
round-trip chain are in German, French or Spanish, and an
English-only encoder would report a spurious semantic deficit at every
odd waypoint, which would corrupt exactly the path quantity we are
trying to measure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loop_geometry import cosine_distance, loop_invariants   # noqa: E402
from runio import RESULTS, Run                               # noqa: E402

EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def newest_corpus() -> Path:
    runs = sorted(RESULTS.glob("run_exp_corpus_v1_*"),
                  key=lambda p: p.name, reverse=True)
    for r in runs:
        # A run that wrote no manifest was interrupted and supports no claim,
        # even when it left a partial corpus.json behind.
        if (r / "corpus.json").exists() and (r / "manifest.json").exists():
            return r / "corpus.json"
    raise FileNotFoundError("no completed corpus run found under 7. Results/")


class Embedder:
    """Mean-pooled sentence embeddings, L2-normalised, dimension 384."""

    def __init__(self, name: str = EMBEDDER):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch: int = 16,
               max_len: int = 512) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            chunk = [t if t.strip() else "." for t in texts[i:i + batch]]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len)
            hid = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = torch.nn.functional.normalize(vec, dim=-1)
            out.append(vec.numpy().astype(np.float64))
        return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    corpus_path = Path(args.corpus) if args.corpus else newest_corpus()
    run = Run("exp_indicators_v1")
    run.log(f"corpus = {corpus_path}")
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    chains = corpus["chains"]
    run.log(f"{len(chains)} chains to process")

    emb = Embedder()
    run.log(f"embedder = {EMBEDDER}")

    # Encode every distinct waypoint once.
    index, texts = {}, []
    for c in chains:
        for j, s in enumerate(c["states"]):
            key = (c["uid"], c["chain"], j)
            index[key] = len(texts)
            texts.append(s)
    run.log(f"encoding {len(texts)} waypoints")
    V = emb.encode(texts, batch=args.batch)
    run.log(f"embeddings: {V.shape}")

    rows, min_cos = [], 1.0
    for c in chains:
        chain_vecs = [V[index[(c["uid"], c["chain"], j)]]
                      for j in range(len(c["states"]))]
        for a, b in zip(chain_vecs[:-1], chain_vecs[1:]):
            min_cos = min(min_cos, 1.0 - cosine_distance(a, b))
        inv = loop_invariants(chain_vecs)

        # A second geometric channel, over the English waypoints only.
        # The multilingual encoder is by design nearly invariant under
        # translation, so it places a German or French waypoint almost on
        # top of its English source; the very property that makes it the
        # right instrument for the semantic deficit makes it a poor one for
        # the path, because it flattens exactly the excursion we wish to
        # measure. Restricted to the English waypoints the encoder is doing
        # real work, and the sub-chain is still a genuine loop.
        en_vecs = [V[index[(c["uid"], c["chain"], j)]] for j in c["english_idx"]]
        inv_en = loop_invariants(en_vecs) if len(en_vecs) >= 2 else None

        hop = c["hops"][-1]
        hop0 = c["hops"][0]
        rows.append({
            "uid": c["uid"], "scheme": c["scheme"], "h": c["h"],
            "chain": c["chain"], "L": c["L"], "n_pivots": len(c["pivots"]),
            # detector side
            "z0": c["z0"], "z_final": c["z_final"],
            "ratio": (c["z_final"] / c["z0"]) if c["z0"] > 0 else float("nan"),
            "rho": hop["rho"], "rho_ctx": hop["rho_ctx"],
            "n_tokens_final": hop["n_tokens"],
            # Round-trip translation does not preserve length, and the
            # detector statistic is normalised by the square root of the
            # number of scored positions. The original count is therefore
            # needed to compare the observed residual with the law.
            "n_tokens_0": hop0["n_tokens"],
            # geometry side
            "semantic_deficit": inv.semantic_deficit,
            "semantic_path_length": inv.semantic_path_length,
            "loop_energy": inv.loop_energy,
            "holonomy_energy": inv.holonomy_energy,
            "n_planes": inv.n_planes,
            "sylvester": list(inv.sylvester),
            "detour_ratio": inv.detour_ratio,
            # English-waypoint channel
            "semantic_deficit_en": inv_en.semantic_deficit if inv_en else None,
            "semantic_path_length_en": inv_en.semantic_path_length if inv_en else None,
            "holonomy_energy_en": inv_en.holonomy_energy if inv_en else None,
            "loop_energy_en": inv_en.loop_energy if inv_en else None,
            "n_english_waypoints": len(en_vecs),
        })

    run.log(f"minimum cosine similarity between consecutive waypoints: "
            f"{min_cos:.4f}  (hypothesis of Certificate_holonomy "
            f"requires it to stay well above -1)")

    by_chain = {}
    for name in sorted({r["chain"] for r in rows}):
        sub = [r for r in rows if r["chain"] == name]
        by_chain[name] = {
            "n": len(sub),
            "median_delta": float(np.median([r["semantic_deficit"] for r in sub])),
            "median_eta": float(np.median([r["holonomy_energy"] for r in sub])),
            "median_rho": float(np.median([r["rho"] for r in sub])),
            "median_ratio": float(np.median([r["ratio"] for r in sub])),
        }
        run.log(f"  {name:12s} n={len(sub):4d}  delta={by_chain[name]['median_delta']:.4f}"
                f"  eta={by_chain[name]['median_eta']:.4f}"
                f"  rho={by_chain[name]['median_rho']:.4f}"
                f"  z-ratio={by_chain[name]['median_ratio']:.4f}")

    run.write_json("indicators.json", {"rows": rows, "by_chain": by_chain,
                                       "embedder": EMBEDDER,
                                       "corpus": str(corpus_path)})
    run.finish(
        conclusions={
            "n_chains": len(rows),
            "min_consecutive_cosine": float(min_cos),
            "by_chain": by_chain,
        },
        limitations=[
            "A single embedder; Plan_v1 risk R4 requires replication "
            "across at least three before the empirical claim is final.",
            "Mean pooling over a 512-token window truncates the longest "
            "waypoints.",
            "The embedder is multilingual but not equally strong in all "
            "four languages, so part of the measured path length in the "
            "pivot waypoints is encoder anisotropy rather than meaning.",
        ],
        inputs={"corpus": str(corpus_path), "embedder": EMBEDDER},
        command="python exp_indicators_v1.py",
    )


if __name__ == "__main__":
    main()
