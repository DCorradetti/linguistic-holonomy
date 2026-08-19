"""
exp_corpus_v1.py -- watermarked corpus and real transformation chains.
=======================================================================
Article  : Article_LLW
Plan ref : Plan_v1.md, sub-goals S5 and S7
Version  : v1
Purpose  : Produce watermarked text with three schemes whose keys we
           hold, subject it to real meaning-preserving transformation
           chains (round-trip machine translation through several pivot
           languages), and record the exact detector statistic at every
           English waypoint.
Input    : none (prompts are embedded below; models are pulled from the
           HuggingFace hub and cached)
Output   : run_exp_corpus_v1_<stamp>/corpus.json  in `7. Results/`

Design
------
Chains are built so that different *paths* can reach comparable
*endpoints*.  Three of them are single round trips through different
pivots; three wander through two or three pivots, or through the same
pivot twice, returning to English in between.  Every waypoint in
English is a point at which the detector can be run, so each chain
yields a decay curve and not merely a final number.

This is the experimental heart of the article: matched semantic
endpoints with dispersed path geometry.

Compute note
------------
This run is deliberately sized for a CPU-only machine.  It is the
reduced-scale version of the testbed described in Plan_v1.md; the
scale-up to a 3-8B generator and to learned paraphrasers is listed in
the manuscript as required future work, not claimed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loop_geometry import context_retention_rate, retention_rate   # noqa: E402
from runio import Run                                              # noqa: E402
from wm_schemes import (WMConfig, apply_greenlist_bias, default_configs,  # noqa: E402
                        detect, exp_sample)

GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MT = "Helsinki-NLP/opus-mt-{src}-{tgt}"

# --------------------------------------------------------------------- #
#  Prompts: five domains, deliberately open-ended so that generation     #
#  carries enough entropy for a watermark to be embedded at all.         #
# --------------------------------------------------------------------- #
PROMPTS = [
    # -- expository
    "Explain how a lighthouse keeper organised his working day in the nineteenth century.",
    "Describe the process by which a medieval manuscript was produced, from parchment to binding.",
    "Explain why coastal cities developed differently from inland cities in the early modern period.",
    "Describe how the first underground railways changed daily life in large cities.",
    "Explain the reasoning behind crop rotation as practised before industrial fertilisers.",
    "Describe how a traditional watermill converts the motion of a river into useful work.",
    "Explain how early cartographers estimated distances they could not measure directly.",
    "Describe the working life of a printer in a small provincial workshop.",
    # -- narrative
    "Tell the story of a botanist who spends a season cataloguing plants on a remote island.",
    "Write a short account of a violin maker who receives an unusual commission.",
    "Tell the story of a night watchman in a museum who notices something out of place.",
    "Write about a translator who becomes obsessed with a single untranslatable word.",
    "Tell the story of a cartographer asked to map a village that no longer exists.",
    "Write a short account of two neighbours who share a garden wall and little else.",
    "Tell the story of a clockmaker in a town where every clock shows a different time.",
    "Write about a librarian who finds an unfamiliar book that is not in any catalogue.",
    # -- argumentative
    "Argue for the view that small museums serve their communities better than large ones.",
    "Discuss whether public transport should be free, considering both sides carefully.",
    "Argue that learning a second language changes how a person thinks about the first.",
    "Discuss the claim that historical reconstruction is always partly interpretation.",
    "Argue for or against preserving industrial buildings that have lost their function.",
    "Discuss whether handwriting should still be taught systematically in schools.",
    "Argue that the study of failed inventions is as instructive as the study of successful ones.",
    "Discuss the tension between tourism and the preservation of historic town centres.",
    # -- descriptive
    "Describe a harbour at dawn, paying attention to sound and movement rather than colour.",
    "Describe a working kitchen in a restaurant during the busiest hour of the evening.",
    "Describe a mountain path in late autumn as a walker experiences it.",
    "Describe an old bookshop from the point of view of someone who has never entered one.",
    "Describe a railway station in a small town between two trains.",
    "Describe a workshop where wooden instruments are repaired.",
    # -- procedural
    "Explain, step by step, how to plan a walking route across unfamiliar countryside.",
    "Explain how to organise a small public exhibition with almost no budget.",
    "Describe how to restore a damaged wooden chair without replacing its original parts.",
    "Explain how to prepare a garden bed for planting in a cold climate.",
    "Explain how a small choir should rehearse a piece it has never performed.",
    "Describe how to catalogue a private collection of photographs for the first time.",
]

# --------------------------------------------------------------------- #
#  Chains.  Each entry is a list of pivot languages; the text returns to #
#  English after every pivot, so every waypoint is detectable.           #
# --------------------------------------------------------------------- #
CHAINS = {
    "rt_de":        ["de"],                    # L = 2
    "rt_fr":        ["fr"],                    # L = 2
    "rt_es":        ["es"],                    # L = 2
    "rt_de_fr":     ["de", "fr"],              # L = 4, two distinct pivots
    "rt_de_de":     ["de", "de"],              # L = 4, same pivot twice
    "rt_fr_de_es":  ["fr", "de", "es"],        # L = 6, long detour
}
PIVOTS = sorted({p for chain in CHAINS.values() for p in chain})


# --------------------------------------------------------------------- #
#  Generation                                                            #
# --------------------------------------------------------------------- #
def load_generator():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.float32)
    model.eval()
    return tok, model


@torch.no_grad()
def generate_watermarked(tok, model, cfg: WMConfig, prompt: str,
                         max_new_tokens: int, temperature: float,
                         seed: int) -> list[int]:
    """
    Token-by-token generation with the watermark applied to the model's
    own distribution.  The green-list family biases the logits; the EXP
    family replaces the source of randomness and leaves the marginal
    law untouched.
    """
    rng = np.random.default_rng(seed)
    chat = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
    past, cur = None, ids
    generated: list[int] = []
    valid_vocab = len(tok)

    for _ in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float().numpy().astype(np.float64)
        # the embedding matrix is padded beyond the tokenizer; those ids are
        # not decodable, so they are dropped rather than merely down-weighted
        logits = logits[:valid_vocab] / temperature

        if cfg.family == "greenlist":
            biased = apply_greenlist_bias(cfg, logits, generated)
            biased = biased - biased.max()
            p = np.exp(biased)
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
        else:
            shifted = logits - logits.max()
            p = np.exp(shifted)
            p /= p.sum()
            nxt = exp_sample(cfg, p, generated)

        generated.append(nxt)
        if nxt == tok.eos_token_id:
            break
        cur = torch.tensor([[nxt]], dtype=torch.long)

    return generated


# --------------------------------------------------------------------- #
#  Translation                                                           #
# --------------------------------------------------------------------- #
class Translators:
    """Lazily loaded Marian models, kept in a dictionary for reuse."""

    def __init__(self):
        self.cache = {}

    def get(self, src: str, tgt: str):
        key = (src, tgt)
        if key not in self.cache:
            from transformers import MarianMTModel, MarianTokenizer
            name = MT.format(src=src, tgt=tgt)
            tok = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name, dtype=torch.float32)
            model.eval()
            self.cache[key] = (tok, model)
        return self.cache[key]

    @torch.no_grad()
    def translate(self, src: str, tgt: str, texts: list[str],
                  batch: int = 8, max_len: int = 512) -> list[str]:
        tok, model = self.get(src, tgt)
        out: list[str] = []
        for i in range(0, len(texts), batch):
            chunk = [t if t.strip() else "." for t in texts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            gen = model.generate(**enc, num_beams=1, do_sample=False,
                                 max_length=max_len)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out


def split_sentences(text: str) -> list[str]:
    """
    Marian is a sentence-level model.  Splitting on sentence boundaries
    and rejoining preserves document structure and avoids truncation of
    long passages, which would otherwise be mistaken for watermark loss.
    """
    import re
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def translate_documents(tr: Translators, src: str, tgt: str,
                        docs: list[str], batch: int = 16) -> list[str]:
    """Translate a list of documents sentence-wise, then rejoin."""
    flat, owner = [], []
    for i, d in enumerate(docs):
        for s in split_sentences(d):
            flat.append(s)
            owner.append(i)
    if not flat:
        return list(docs)
    translated = tr.translate(src, tgt, flat, batch=batch)
    out = [[] for _ in docs]
    for i, s in zip(owner, translated):
        out[i].append(s)
    return [" ".join(p) for p in out]


# --------------------------------------------------------------------- #
#  Main                                                                  #
# --------------------------------------------------------------------- #
def by_name(cfgs, name: str):
    for c in cfgs:
        if c.name == name:
            return c
    raise KeyError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--key", type=int, default=20260819)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chains", type=str, default=None,
                    help="comma-separated subset of the chain names to run; "
                         "default is all of them. Chains are independent, so "
                         "running them in separate invocations and merging is "
                         "equivalent to running them together, and is what one "
                         "must do on a machine where a long process is liable "
                         "to be terminated.")
    ap.add_argument("--merge", type=str, default=None,
                    help="path to a corpus.json whose chain records are to be "
                         "carried into this run's output.")
    ap.add_argument("--resume-stage1", type=str, default=None,
                    help="path to a corpus_stage1.json from an interrupted "
                         "run; its records are reused instead of regenerated. "
                         "Generation is deterministic in "
                         "(model revision, seed, key), so a resumed record is "
                         "identical to the one a fresh run would produce.")
    ap.add_argument("--verify-resume", type=int, default=3,
                    help="how many reused records to regenerate and compare "
                         "token by token, as an audit of the determinism on "
                         "which the reuse rests. 0 disables the check.")
    args = ap.parse_args()

    chain_items = list(CHAINS.items())
    if args.chains:
        want = [c.strip() for c in args.chains.split(",") if c.strip()]
        unknown = [c for c in want if c not in CHAINS]
        if unknown:
            raise SystemExit(f"unknown chain names: {unknown}; "
                             f"available: {sorted(CHAINS)}")
        chain_items = [(c, CHAINS[c]) for c in want]

    run = Run("exp_corpus_v1")
    run.log(f"generator = {GEN_MODEL}")
    tok, model = load_generator()
    vocab = len(tok)
    cfgs = default_configs(vocab_size=vocab, key=args.key)
    run.log(f"tokenizer vocab = {vocab}; schemes = {[c.name for c in cfgs]}")

    prompts = PROMPTS[:args.n_prompts]
    records: list[dict] = []

    cached: dict[str, dict] = {}
    if args.resume_stage1:
        with open(args.resume_stage1, encoding="utf-8") as fh:
            for r in json.load(fh):
                cached[r["uid"]] = r
        run.log(f"resuming: {len(cached)} generated records available from "
                f"{args.resume_stage1}")

    # ---- stage 1: watermarked generation -------------------------------
    t0 = time.time()
    n_reused = 0
    for ci, cfg in enumerate(cfgs):
        for pi, prompt in enumerate(prompts):
            uid = f"{cfg.name}_{pi:03d}"
            if uid in cached:
                records.append(cached[uid])
                n_reused += 1
                continue
            ids = generate_watermarked(
                tok, model, cfg, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed + 1000 * ci + pi,
            )
            text = tok.decode(ids, skip_special_tokens=True)
            det = detect(cfg, ids)
            records.append({
                "uid": f"{cfg.name}_{pi:03d}",
                "scheme": cfg.name,
                "h": cfg.h,
                "prompt_id": pi,
                "prompt": prompt,
                "tokens": [int(t) for t in ids],
                "text": text,
                "z0": det["z"],
                "p0": det["p"],
                "n_tokens": len(ids),
            })
            if (pi + 1) % 6 == 0:
                done = ci * len(prompts) + pi + 1
                tot = len(cfgs) * len(prompts)
                fresh = max(1, done - n_reused)
                run.log(f"generated {done}/{tot}  "
                        f"({(time.time()-t0)/fresh:.1f}s/new text)  "
                        f"last z0={det['z']:.2f}")
        run.write_json("corpus_stage1.json", records)
    run.log(f"stage 1 complete: {len(records)} records "
            f"({n_reused} reused, {len(records)-n_reused} generated)")

    # The reuse above is legitimate only because generation is deterministic
    # in (model revision, seed, key). That is an assertion about this machine
    # and this build of torch, not a theorem, so it is checked rather than
    # assumed: a sample of the reused records is regenerated from scratch and
    # compared token by token.
    resume_audit = None
    if n_reused and args.verify_resume > 0:
        rng = np.random.default_rng(args.seed)
        pool = [r for r in records if r["uid"] in cached]
        pick = rng.choice(len(pool), size=min(args.verify_resume, len(pool)),
                          replace=False)
        agree = 0
        for k in pick:
            rec = pool[int(k)]
            ci = [c.name for c in cfgs].index(rec["scheme"])
            ids = generate_watermarked(
                tok, model, by_name(cfgs, rec["scheme"]), rec["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed + 1000 * ci + rec["prompt_id"],
            )
            same = [int(t) for t in ids] == rec["tokens"]
            agree += int(same)
            run.log(f"  resume audit {rec['uid']}: "
                    f"{'identical' if same else 'DIFFERENT'}")
        resume_audit = {"n_checked": int(len(pick)), "n_identical": agree}
        if agree != len(pick):
            raise SystemExit("resumed records are not reproducible on this "
                             "machine; rerun stage 1 from scratch")
        run.log(f"resume audit: {agree}/{len(pick)} regenerated records "
                f"identical token for token")

    z0s = {c.name: [r["z0"] for r in records if r["scheme"] == c.name] for c in cfgs}
    for k, v in z0s.items():
        run.log(f"  {k:8s} median z0 = {float(np.median(v)):6.2f}  "
                f"detected@z>4: {sum(1 for x in v if x > 4)}/{len(v)}")

    # ---- stage 2: real transformation chains ---------------------------
    # The generator is no longer needed and holds about 2 GB; releasing it
    # before six Marian models are loaded keeps the run inside memory on a
    # machine that must also host the analysis. An earlier run of this
    # script was terminated at this point for want of that precaution.
    del model
    import gc
    gc.collect()
    run.log("generator released before loading the translation models")

    tr = Translators()
    by_cfg = {c.name: c for c in cfgs}
    chains_out: list[dict] = []

    # Chains are independent of one another: they share only the stage-1
    # records, and each is a separate excursion out of English and back.
    # Running them in several invocations and merging is therefore equivalent
    # to running them together, and is what one must do on a machine where an
    # hour-long process is liable to be terminated. Only a run that wrote a
    # manifest may be merged from.
    if args.merge:
        merge_path = Path(args.merge)
        if not (merge_path.parent / "manifest.json").exists():
            raise SystemExit(f"{merge_path.parent.name} has no manifest: it "
                             f"was interrupted and supports no claim")
        with open(merge_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        if {r["uid"] for r in prev["records"]} != {r["uid"] for r in records}:
            raise SystemExit("the merged run was built on a different set of "
                             "stage-1 records")
        fresh = {name for name, _ in chain_items}
        carried = [c for c in prev["chains"] if c["chain"] not in fresh]
        chains_out.extend(carried)
        run.log(f"merging {len(carried)} chain records "
                f"({sorted({c['chain'] for c in carried})}) from "
                f"{merge_path.parent.name}")

    for chain_name, pivots in chain_items:
        run.log(f"chain {chain_name}: pivots {pivots}")
        docs = [r["text"] for r in records]
        waypoints = [list(docs)]                     # waypoint 0 = the original
        for pivot in pivots:
            mid = translate_documents(tr, "en", pivot, waypoints[-1], batch=args.batch)
            back = translate_documents(tr, pivot, "en", mid, batch=args.batch)
            waypoints.append(mid)                    # intermediate language
            waypoints.append(back)                   # back in English
            run.log(f"  ... through {pivot} done ({len(mid)} docs)")

        for i, rec in enumerate(records):
            cfg = by_cfg[rec["scheme"]]
            states = [w[i] for w in waypoints]
            # detection is only meaningful on the English waypoints
            english_idx = [0] + [2 * k + 2 for k in range(len(pivots))]
            hops = []
            for j in english_idx:
                ids_j = tok(states[j], add_special_tokens=False)["input_ids"]
                det = detect(cfg, ids_j)
                hops.append({
                    "waypoint": j,
                    "z": det["z"],
                    "p": det["p"],
                    "n_tokens": len(ids_j),
                    "rho": retention_rate(rec["tokens"], ids_j),
                    "rho_ctx": context_retention_rate(rec["tokens"], ids_j, cfg.h),
                })
            chains_out.append({
                "uid": rec["uid"],
                "scheme": rec["scheme"],
                "h": cfg.h,
                "chain": chain_name,
                "pivots": pivots,
                "L": len(states) - 1,
                "z0": rec["z0"],
                "states": states,
                "english_idx": english_idx,
                "hops": hops,
                "z_final": hops[-1]["z"],
                "rho_final": hops[-1]["rho"],
                "rho_ctx_final": hops[-1]["rho_ctx"],
            })
        run.write_json("corpus.json", {"records": records, "chains": chains_out})
        run.log(f"chain {chain_name} complete ({len(chains_out)} chain records so far)")

    # ---- summary -------------------------------------------------------
    cmd = (f"python exp_corpus_v1.py --n-prompts {len(prompts)} "
           f"--max-new-tokens {args.max_new_tokens}")
    if args.chains:
        cmd += " --chains " + ",".join(name for name, _ in chain_items)
    if args.merge:
        cmd += f' --merge "{args.merge}"'
    if args.resume_stage1:
        cmd += f' --resume-stage1 "{args.resume_stage1}"'

    concl = {}
    for c in cfgs:
        sub = [x for x in chains_out if x["scheme"] == c.name]
        concl[c.name] = {
            "median_z0": float(np.median([x["z0"] for x in sub])),
            "median_z_final": float(np.median([x["z_final"] for x in sub])),
            "median_rho_final": float(np.median([x["rho_final"] for x in sub])),
            "n_chains": len(sub),
        }
        run.log(f"  {c.name:8s} z0={concl[c.name]['median_z0']:.2f} -> "
                f"z_final={concl[c.name]['median_z_final']:.2f} "
                f"(rho={concl[c.name]['median_rho_final']:.3f})")

    run.finish(
        conclusions=concl,
        limitations=[
            "CPU-only run: generator is Qwen2.5-0.5B-Instruct, not a 3-8B model.",
            "Transformations are round-trip machine translation only; learned "
            "paraphrasers (DIPPER) and human editing are not covered.",
            "EXP detection uses the alignment-free context-seeded variant; "
            "the edit-distance aligned detector of Kuditipudi et al. is not implemented.",
            "Sentence-wise translation preserves document structure but is "
            "itself a modelling choice that slightly favours retention.",
        ],
        inputs={"generator": GEN_MODEL, "pivots": PIVOTS,
                "n_prompts": len(prompts), "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "key": args.key,
                "chains_executed": [name for name, _ in chain_items],
                "chains_merged_from": args.merge,
                "stage1_resumed_from": args.resume_stage1,
                "resume_audit": resume_audit},
        command=cmd,
    )


if __name__ == "__main__":
    main()
