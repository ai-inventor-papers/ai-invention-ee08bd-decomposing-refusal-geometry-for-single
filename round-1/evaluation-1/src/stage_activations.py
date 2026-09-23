#!/usr/bin/env python3
"""Stage B: activation + logit extraction per panel model.

For the 32 harmful (AdvBench) + 32 benign (XSTest-style) prompt sets, runs
batched teacher-forced forward passes with output_hidden_states=True and
output_attentions=True, capturing:
  * last-token residual-stream hidden states for all layers  -> fp16 npz
  * last-token attention entropy per layer                    -> npz
  * first-position logits (refusal-token margin, entropy)     -> npz

Cached under acts_cache/acts_<tag>.npz + acts_<tag>_meta.json.
One model resident at a time; ~2.2 GB peak for logits+states per model on CPU.

Run:  .venv/bin/python stage_activations.py [model_hf_id ...]
"""

from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ACTS_DIR, REFUSAL_TOKEN_STRINGS, ROOT, format_prompt, \
    has_chat_template, logger
from models import resolved_panel

PROMPTS = json.loads((ROOT / "data" / "prompts.json").read_text())
BATCH = 4


@torch.no_grad()
def extract_batch(model, tok, idss: list[list[int]]):
    """Returns H [B, L+1, d] fp16 (cpu), E [B, L] fp32, logits [B, V] fp32."""
    maxlen = max(len(x) for x in idss)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(idss), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idss), maxlen), dtype=torch.long)
    for i, x in enumerate(idss):
        ids[i, maxlen - len(x):] = torch.tensor(x, dtype=torch.long)
        mask[i, maxlen - len(x):] = 1
    out = model(input_ids=ids, attention_mask=mask,
                output_hidden_states=True, output_attentions=True)
    idx = torch.arange(len(idss))
    last_idx = mask.sum(1) - 1
    H = torch.stack([h[idx, last_idx, :].cpu() for h in out.hidden_states],
                    dim=1).to(torch.float16)
    ents = []
    for att in out.attentions:
        a = att[idx, :, last_idx, :].clamp_min(1e-12)
        a = a / a.sum(-1, keepdim=True)
        ent = -(a * a.log()).sum(-1).mean(-1)  # head-mean -> [B]
        ents.append(ent.cpu().to(torch.float32))
        del att, a, ent
    E = torch.stack(ents, dim=1).numpy()
    logits = out.logits[idx, last_idx, :].float().cpu().numpy()
    del out, ents
    return H.numpy(), E, logits


def refusal_token_ids(tok) -> tuple[np.ndarray, np.ndarray]:
    ref_ids: set[int] = set()
    for s in REFUSAL_TOKEN_STRINGS:
        for tid in tok.encode(s, add_special_tokens=False):
            ref_ids.add(tid)
    ctrl_ids: set[int] = set()
    for s in ["Sure", " Sure", "Here", " Here", "Step", " Step",
              "Of course", "1", "First", " First"]:
        for tid in tok.encode(s, add_special_tokens=False):
            ctrl_ids.add(tid)
    return np.array(sorted(ref_ids)), np.array(sorted(ctrl_ids))


def run_model(m: dict) -> None:
    tag = m["hf_id"].replace("/", "__")
    out_path = ACTS_DIR / f"acts_{tag}.npz"
    meta_path = ACTS_DIR / f"acts_{tag}_meta.json"
    if out_path.exists() and meta_path.exists():
        logger.info(f"acts cached: {m['hf_id']}")
        return
    t0 = time.time()
    logger.info(f"[acts] loading {m['hf_id']}")
    tok = AutoTokenizer.from_pretrained(m["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        m["hf_id"], dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager")
    model.eval()
    chat = m["chat"] and has_chat_template(tok)
    ref_ids, ctrl_ids = refusal_token_ids(tok)
    Hs, Es, M11, M12 = {}, {}, {}, {}
    for key in ("harmful", "benign"):
        plist = PROMPTS[f"{key}_32"]
        hs, es, m11, m12 = [], [], [], []
        for i in range(0, len(plist), BATCH):
            chunk = plist[i:i + BATCH]
            idss = [format_prompt(tok, p, chat) for p in chunk]
            H, E, logits = extract_batch(model, tok, idss)
            hs.append(H)
            es.append(E)
            lref = logits[:, ref_ids].max(axis=1)
            others = logits.copy()
            others[:, ref_ids] = -1e30
            loth = others.max(axis=1)
            m11.append(lref - loth)                     # refusal-vs-rest margin
            p = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
            m12.append(ent)
            del logits, p
            gc.collect()
        Hs[key] = np.concatenate(hs, 0)   # [32, L+1, d]
        Es[key] = np.concatenate(es, 0)
        M11[key] = np.concatenate(m11, 0)
        M12[key] = np.concatenate(m12, 0)
        del hs, es, m11, m12
        gc.collect()
    np.savez_compressed(out_path,
                        H_harm=Hs["harmful"], H_ben=Hs["benign"],
                        E_harm=Es["harmful"], E_ben=Es["benign"],
                        M11_harm=M11["harmful"], M11_ben=M11["benign"],
                        M12_harm=M12["harmful"], M12_ben=M12["benign"])
    meta = {"hf_id": m["hf_id"], "role": m["role"], "family": m["family"],
            "params_b": m["params_b"], "used_chat": chat,
            "n_layers_H": int(Hs["harmful"].shape[1]),
            "d_model": int(Hs["harmful"].shape[2]),
            "n_harm": int(Hs["harmful"].shape[0]),
            "n_ben": int(Hs["benign"].shape[0]),
            "refusal_token_ids": ref_ids.tolist(),
            "ctrl_token_ids": ctrl_ids.tolist(),
            "runtime_s": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta, indent=2))
    del model, Hs, Es, M11, M12
    gc.collect()
    logger.info(f"[acts] {m['hf_id']} done ({meta['runtime_s']}s, "
                f"L={meta['n_layers_H']}, d={meta['d_model']})")


@logger.catch(reraise=True)
def main() -> None:
    panel = resolved_panel()
    wanted = sys.argv[1:]
    if wanted:
        panel = [m for m in panel if m.hf_id in wanted]
    logger.info(f"activation pass on {len(panel)} models")
    for m in panel:
        try:
            run_model(m.__dict__)
        except Exception as e:
            logger.error(f"FAILED acts {m.hf_id}: {type(e).__name__}: {e}")
            gc.collect()
    logger.info("acts pass done")


if __name__ == "__main__":
    main()
