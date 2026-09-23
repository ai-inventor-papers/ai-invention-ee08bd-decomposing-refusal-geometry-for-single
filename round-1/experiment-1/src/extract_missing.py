#!/usr/bin/env python3
"""Fill the extraction gaps left by the earlier draft run.

For each model tag this script ensures, on ALL 80 battery prompts:
  - full decision-point logits row [V] (float32, saved as top-256 + named-token logits
    + entropy, to keep npz small)
  - last-token residual stream [37, 2560] fp16            (already done for base/instruct)
  - per-layer attention entropy + sink mass [36]          (already done for base/instruct)

Missing pieces are computed incrementally: existing npz tensors are re-used, only the
absent ones are computed, and the model-local npz under results/activations/{tag}/
is rewritten with the union. Also materialises workspace-local copies of the gated
mlabonne weights so the model loads without touching the shared cache.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
ACT = ROOT / "results" / "activations"
MODELS_LOCAL = ROOT / "models_local"
SHARED_HF = Path("/ai-inventor/aii_data/runs/run_K2ftFsEsyOu3/.shared_cache/hf/hub")

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(ROOT / "logs" / "extract_missing.log", rotation="30 MB", level="DEBUG", backtrace=True)

MODELS: dict[str, str] = {
    "base": "Qwen/Qwen3-4B-Base",
    "instruct": "Qwen/Qwen3-4B",
    "saferl": "Qwen/Qwen3-4B-SafeRL",
    "abli": "mlabonne/Qwen3-4B-abliterated",
}
ORDER = ["abli", "saferl", "base", "instruct"]   # run the slow unextracted ones first


def snapshot_dir(repo: str) -> Path:
    tag_dir = "models--" + repo.replace("/", "--")
    snaps = sorted((SHARED_HF / tag_dir).glob("snapshots/*/"))
    if not snaps:
        raise FileNotFoundError(f"no cached snapshot for {repo}")
    return snaps[-1]


def materialise_local(repo: str, dest: Path) -> Path:
    """Copy (or reuse) a workspace-local full copy of the model snapshot."""
    if (dest / "config.json").exists():
        return dest
    src = snapshot_dir(repo)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    for f in sorted(src.iterdir()):
        if f.is_file():
            d = dest / f.name
            if not d.exists() or d.stat().st_size != f.stat().st_size:
                shutil.copyfile(f, d)
            n += 1
    logger.info(f"materialised {repo} -> {dest.name} ({n} files, {time.time() - t0:.0f}s)")
    return dest


def encode(tok, text: str, chat: bool) -> list[int]:
    if chat:
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_dict=True, enable_thinking=False)
        return list(enc["input_ids"])
    ids = tok.encode(f"Question: {text}\nAnswer:", add_special_tokens=False)
    return ids + [tok.eos_token_id]


def named_token_ids(tok) -> tuple[list[int], list[str]]:
    refusal = ["I", "I'm", "Sorry", "sorry", "As", "Unfortunately", "cannot", "can't",
               "won't", "unable", "must", "apologize", "against", "decline", "not",
               "No", "never", "However", "That", "cannot assist"]
    compliance = ["Sure", "Here", "Okay", "Certainly", "Step", "First", "Let", "Great",
                  "Absolutely", "Yes", "To", "The", "A", "Definitely", "course"]
    ids, names = [], []
    for grp, toks in (("refusal", refusal), ("compliance", compliance)):
        for s in toks:
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                ids.append(int(e[0]))
                names.append(f"{grp}|{s}")
    return ids, names


def process(tag: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_grad_enabled(False)

    d = ACT / tag
    d.mkdir(parents=True, exist_ok=True)
    battery = json.loads((ROOT / "data" / "prompt_battery.json").read_text())
    pids = [r["pid"] for r in battery]
    n = len(pids)

    H_path = d / "dev_hidden.npz"
    Ho_path = d / "heldout_hidden.npz"
    need_full = not ((d / "full_hidden.npz").exists())
    if need_full:
        # merge dev + heldout H if present
        Hs, order = [], []
        if H_path.exists():
            z = np.load(H_path)
            Hs.append(z["H"])
            order += [r["pid"] for r in battery if r["split"] == "dev"]
        if Ho_path.exists():
            z = np.load(Ho_path)
            Hs.append(z["H"])
            order += [r["pid"] for r in battery if r["split"] == "heldout"]
        have = {p: i for i, p in enumerate(order)}
        missing = [p for p in pids if p not in have]
        logger.info(f"[{tag}] have hidden for {len(order)}/{n}; missing {len(missing)}")

        chat = tag != "base"
        local = materialise_local(MODELS[tag], MODELS_LOCAL / tag)
        tok = AutoTokenizer.from_pretrained(str(local))
        model = AutoModelForCausalLM.from_pretrained(str(local), dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True,
                                                     attn_implementation="eager")
        model.eval()
        L = model.config.num_hidden_layers + 1
        D = model.config.hidden_size

        H_new = np.zeros((n, L, D), dtype=np.float16)
        attn_new = np.zeros((n, L - 1), dtype=np.float32)
        sink_new = np.zeros((n, L - 1), dtype=np.float32)
        logits_named_new = None
        topk_new = np.zeros((n, 256), dtype=np.float32)
        topk_id_new = np.zeros((n, 256), dtype=np.int64)
        ent_new = np.zeros((n,), dtype=np.float32)
        ref_ids, ref_names = named_token_ids(tok)
        logits_named_new = np.zeros((n, len(ref_ids)), dtype=np.float32)

        # fill cached rows first (H only; logits always recomputed: cheap via re-run? no - we re-run everything)
        # NOTE: recompute all 80 forward passes for consistency (12s each => ~16 min)
        t0 = time.time()
        for i, r in enumerate(battery):
            ids = torch.tensor([encode(tok, r["prompt"], chat)])
            o = model(ids, output_hidden_states=True, output_attentions=True,
                      logits_to_keep=1, use_cache=False)
            hs = torch.stack([h[0, -1, :] for h in o.hidden_states]).to(torch.float32)
            H_new[i] = hs.numpy().astype(np.float16)
            for li, a in enumerate(o.attentions):
                row = a[0, :, -1, :].to(torch.float32)
                p = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
                attn_new[i, li] = float(-(p * (p + 1e-12).log()).sum(-1).mean())
                sink_new[i, li] = float(p[:, 0].mean())
            lg = o.logits[0, -1].to(torch.float32)
            logits_named_new[i] = lg[ref_ids].numpy()
            tv, ti = torch.topk(lg, 256)
            topk_new[i] = tv.numpy()
            topk_id_new[i] = ti.numpy()
            lse = torch.logsumexp(lg, 0)
            pp = lg - lse
            ent_new[i] = float(-(pp.exp() * pp).sum())
            del o, hs, lg
            if (i + 1) % 10 == 0:
                logger.info(f"[{tag}] {i + 1}/{n} prompts ({time.time() - t0:.0f}s)")

        np.savez_compressed(d / "full_hidden.npz", H=H_new, attn_entropy=attn_new,
                            sink_mass=sink_new)
        np.savez_compressed(d / "full_logits.npz", named_logits=logits_named_new,
                            named_names=np.array(ref_names, dtype="<U32"),
                            topk_vals=topk_new, topk_ids=topk_id_new,
                            entropies=ent_new, pids=np.array(pids, dtype="<U16"))
        meta = {"tag": tag, "repo": MODELS[tag], "n_prompts": n, "layers": L, "d": D,
                "recomputed": True, "chat_template": chat,
                "seconds": time.time() - t0}
        (d / "full_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"[{tag}] full extraction done in {time.time() - t0:.0f}s")
        del model
        gc.collect()
    else:
        logger.info(f"[{tag}] full_hidden.npz already present -> skip")


if __name__ == "__main__":
    only = sys.argv[1:] or ORDER
    for t in [x for x in ORDER if x in only]:
        try:
            process(t)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[{t}] FAILED: {e}")
            raise
