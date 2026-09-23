#!/usr/bin/env python3
"""Stage A: behavioral metrics on the widened panel.

For each panel model: greedy (temperature=0) short generations on the 10
harmful (AdvBench) + 10 benign (XSTest-style) generation prompts, batched with
left padding for throughput on CPU. Keyword refusal classification -> B1, B2,
B3 = B1 - B2. Cached per model under results/behavior_<tag>.json.

Run:  .venv/bin/python stage_behavior.py [model_hf_id ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger

from common import (ROOT, format_prompt, has_chat_template, keyword_refusal,
                    load_model, logger, release, save_json)
from models import resolved_panel

PROMPTS = json.loads((ROOT / "data" / "prompts.json").read_text())
MAX_NEW = 24
BATCH = 5


@torch.no_grad()
def generate_batch(model, tok, list_ids: list[list[int]], max_new: int) -> list[str]:
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lens = [len(x) for x in list_ids]
    maxlen = max(lens)
    # left-pad manually to avoid relying on tokenizer padding direction quirks
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(list_ids), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(list_ids), maxlen), dtype=torch.long)
    for i, x in enumerate(list_ids):
        ids[i, maxlen - len(x):] = torch.tensor(x, dtype=torch.long)
        mask[i, maxlen - len(x):] = 1
    gen = model.generate(input_ids=ids, attention_mask=mask,
                         max_new_tokens=max_new, do_sample=False,
                         pad_token_id=pad_id)
    out = []
    for i in range(len(list_ids)):
        new = gen[i, maxlen:]
        # strip anything before the first non-pad token (left pads spill over)
        ntok = int((new != pad_id).sum()) or len(new)
        text = tok.decode(new[-ntok:], skip_special_tokens=True)
        out.append(text)
    return out


def run_model(m: dict) -> dict:
    out_path = ROOT / "results" / f"behavior_{m['hf_id'].replace('/', '__')}.json"
    if out_path.exists():
        logger.info(f"behavior cached: {m['hf_id']}")
        return json.loads(out_path.read_text())
    t0 = time.time()
    model, tok = load_model(m["hf_id"])
    chat = m["chat"] and has_chat_template(tok)
    rec = {"hf_id": m["hf_id"], "role": m["role"], "family": m["family"],
           "params_b": m["params_b"], "used_chat": chat, "max_new": MAX_NEW,
           "responses": {}}
    try:
        for key, plist in (("harmful", PROMPTS["harmful_gen10"]),
                           ("benign", PROMPTS["benign_gen10"])):
            rec["responses"][key] = []
            for i in range(0, len(plist), BATCH):
                chunk = plist[i:i + BATCH]
                idss = [format_prompt(tok, p, chat) for p in chunk]
                texts = generate_batch(model, tok, idss, MAX_NEW)
                for p, t in zip(chunk, texts):
                    rec["responses"][key].append({
                        "prompt": p, "text": t,
                        "keyword_refusal": keyword_refusal(t)})
    except torch.cuda.OutOfMemoryError:  # pragma: no cover - cpu path
        logger.error("OOM (unexpected on cpu)")
        raise
    finally:
        release(model)
    n_h = len(rec["responses"]["harmful"])
    n_b = len(rec["responses"]["benign"])
    rec["B1"] = sum(r["keyword_refusal"] for r in rec["responses"]["harmful"]) / max(n_h, 1)
    rec["B2"] = sum(r["keyword_refusal"] for r in rec["responses"]["benign"]) / max(n_b, 1)
    rec["B3"] = rec["B1"] - rec["B2"]
    rec["runtime_s"] = round(time.time() - t0, 1)
    save_json(rec, out_path)
    logger.info(f"{m['hf_id']}: B1={rec['B1']} B2={rec['B2']} B3={rec['B3']} ({rec['runtime_s']}s)")
    return rec


@logger.catch(reraise=True)
def main() -> None:
    panel = resolved_panel()
    wanted = sys.argv[1:]
    if wanted:
        panel = [m for m in panel if m.hf_id in wanted]
    logger.info(f"behavioral pass on {len(panel)} models")
    results = {}
    for m in panel:
        try:
            results[m.hf_id] = run_model(m.__dict__)
        except Exception as e:
            logger.error(f"FAILED {m.hf_id}: {type(e).__name__}: {e}")
    logger.info(f"done: {len(results)} models")


if __name__ == "__main__":
    main()
