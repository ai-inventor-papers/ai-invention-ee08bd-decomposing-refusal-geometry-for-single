#!/usr/bin/env python3
"""Generate behavior_<tag>.json for Qwen3-4B and Qwen3-4B-abliterated.

Uses the experiment-1 cached 40/20 prompt split (data/prompt_battery.json):
held-out harmful (10) -> B1, held-out benign/dual-use (10) -> B2, B3 = B1 - B2.
Greedy generation, max_new=24, chat template (both models share the Qwen3
thinking template; a complete think block is stripped before classification).

Run:  .venv/bin/python gen_classify.py
"""

from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path

import torch
from loguru import logger

from common import (RESULTS_DIR, keyword_refusal, load_model, logger,
                    release, save_json)

ROOT = Path(__file__).resolve().parent
EXP1 = Path("/ai-inventor/aii_data/runs/run_K2ftFsEsyOu3/3_invention_loop/"
            "iter_1/gen_art/gen_art_experiment_1")
MAX_NEW = 24
BATCH = 5

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def load_battery() -> tuple[list[str], list[str]]:
    d = json.loads((EXP1 / "data" / "prompt_battery.json").read_text())
    harm = [x["prompt"] for x in d
            if x["category"] == "harmful" and x["split"] == "heldout"]
    ben = [x["prompt"] for x in d
            if x["category"] != "harmful" and x["split"] == "heldout"]
    assert len(harm) == 10 and len(ben) == 10, (len(harm), len(ben))
    return harm, ben


@torch.no_grad()
def generate_batch(model, tok, list_ids: list[list[int]], max_new: int) -> list[str]:
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    maxlen = max(len(x) for x in list_ids)
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
        ntok = int((new != pad_id).sum()) or len(new)
        out.append(tok.decode(new[-ntok:], skip_special_tokens=False))
    return out


def run(hf_id: str, tag: str, role: str) -> None:
    out_path = RESULTS_DIR / f"behavior_{tag}"
    if out_path.exists():
        logger.info(f"cached: {hf_id}")
        return
    t0 = time.time()
    harm, ben = load_battery()
    model, tok = load_model(hf_id)
    rec = {"hf_id": hf_id, "role": role, "family": "qwen3", "params_b": 4.0,
           "used_chat": True, "max_new": MAX_NEW,
           "prompt_source": "experiment-1 held-out battery (20 prompts; disjoint from 40-dev metric prompts)",
           "responses": {}}
    try:
        for key, plist in (("harmful", harm), ("benign", ben)):
            rec["responses"][key] = []
            for i in range(0, len(plist), BATCH):
                chunk = plist[i:i + BATCH]
                idss = [tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True)["input_ids"] for p in chunk]
                texts = generate_batch(model, tok, idss, MAX_NEW)
                for p, t in zip(chunk, texts):
                    final = strip_think(t)
                    rec["responses"][key].append({
                        "prompt": p, "text": final, "think_len": len(t),
                        "keyword_refusal": keyword_refusal(final)})
    finally:
        release(model)
    gc.collect()
    n_h = len(rec["responses"]["harmful"])
    n_b = len(rec["responses"]["benign"])
    rec["B1"] = sum(r["keyword_refusal"] for r in rec["responses"]["harmful"]) / n_h
    rec["B2"] = sum(r["keyword_refusal"] for r in rec["responses"]["benign"]) / n_b
    rec["B3"] = rec["B1"] - rec["B2"]
    rec["runtime_s"] = round(time.time() - t0, 1)
    save_json(rec, out_path)
    logger.info(f"{hf_id}: B1={rec['B1']} B2={rec['B2']} B3={rec['B3']} "
                f"({rec['runtime_s']}s)")


@logger.catch(reraise=True)
def main() -> None:
    jobs = [("Qwen/Qwen3-4B", "Qwen__Qwen3-4B.json", "instruct"),
            ("mlabonne/Qwen3-4B-abliterated",
             "huihui-ai__Qwen3-4B-abliterated.json", "abliterated")]
    for hf_id, fname, role in jobs:
        try:
            run(hf_id, fname, role)
        except Exception as e:
            logger.error(f"FAILED {hf_id}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()