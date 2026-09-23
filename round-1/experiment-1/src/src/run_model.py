#!/usr/bin/env python3
"""Run the full extraction + per-model metric battery for the models selected.

Usage: uv run python src/run_model.py [--models name1,name2] [--limit N] [--skip-gen] [--skip-weights]
Writes per model:
  results/activations/<short>/hidden.npy    float16 [n, L+1, d]
  results/activations/<short>/logits.npy    float16 [n, V]
  results/activations/<short>/attn_ent.npy  float32 [n, L]
  results/activations/<short>/gens.json     generated openings
  results/activations/<short>/meta.json     input texts, token counts, config
  results/metrics/<short>.json              metric battery for this model
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.extract import Extractor
from src.hw import RSSWatchdog, configure_threads, current_rss_gb, log_hardware
from src.metrics import compute_all
from src.registry import REGISTRY, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "data" / "prompts" / "battery.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated short names; default all")
    ap.add_argument("--limit", type=int, default=None, help="only first N prompts (smoke tests)")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--out-metrics", default=None, help="override metrics dir")
    return ap.parse_args()


def run_model(spec: ModelSpec, prompts: list[dict], args: argparse.Namespace) -> bool:
    act_dir = ROOT / "results" / "activations" / spec.short
    met_dir = ROOT / "results" / "metrics"
    act_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    hidden_path = act_dir / "hidden.npy"
    if hidden_path.exists() and (met_dir / f"{spec.short}.json").exists():
        logger.info(f"{spec.short}: already done, skipping")
        return True

    t0 = time.time()
    ex = Extractor(spec.repo, spec.template)
    n = len(prompts)
    hidden: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    attn: list[np.ndarray] = []
    gens: list[dict] = []
    meta: list[dict] = []
    try:
        for i, p in enumerate(prompts):
            cap = ex.capture(p["text"])
            hidden.append(cap["hidden"])
            logits.append(cap["logits"].astype(np.float16))
            attn.append(cap["attn_ent"])
            if not args.skip_gen:
                g = ex.generate_greedy(p["text"])
                gens.append({"prompt_id": p["id"], "text": g["text"], "post_think": g["post_think"]})
            else:
                gens.append({"prompt_id": p["id"], "text": "", "post_think": ""})
            meta.append(
                {
                    "prompt_id": p["id"],
                    "input_text": cap["input_text"],
                    "n_prompt_tokens": cap["n_prompt_tokens"],
                }
            )
            if (i + 1) % 10 == 0:
                logger.info(
                    f"{spec.short}: {i+1}/{n} prompts done ({time.time()-t0:.0f}s, anon {current_rss_gb():.1f}GB)"
                )
    except MemoryError:
        logger.error(f"{spec.short}: MemoryError at prompt {i} - aborting this model")
        ex.close()
        return False

    H = np.stack(hidden)  # [n, L+1, d] fp16
    LG = np.stack(logits)  # [n, V] fp16
    AT = np.stack(attn)  # [n, L] fp32
    np.save(hidden_path, H)
    np.save(act_dir / "logits.npy", LG)
    np.save(act_dir / "attn_ent.npy", AT)
    (act_dir / "gens.json").write_text(json.dumps(gens, indent=1))
    (act_dir / "meta.json").write_text(
        json.dumps(
            {
                "repo": spec.repo,
                "short": spec.short,
                "template": spec.template,
                "n_layers": ex.n_layers,
                "d_model": ex.d_model,
                "meta": meta,
            },
            indent=1,
        )
    )
    logger.info(f"{spec.short}: tensors saved ({H.shape}, {LG.shape}); computing weight metrics")

    # weight metrics need the loaded model - compute now, then free
    wres = None
    if not args.skip_weights:
        try:
            from src.metrics import wmetrics

            wres = wmetrics(ex.model, ex.n_layers)
        except Exception as e:  # noqa: BLE001
            logger.error(f"{spec.short}: weight metrics failed: {type(e).__name__}: {e}")
            wres = None
    ex.close()

    # per-model metric battery (activation/logit/text metrics from saved tensors)
    split_mask = [p["split"] for p in prompts]
    labels = [{"category": p["category"], "subtype": p["subtype"]} for p in prompts]
    metrics = compute_all(H, AT, LG.astype(np.float32), gens, labels, None, H.shape[1] - 1, split_mask)
    if wres:
        metrics.update(wres)
    out = {
        "model": spec.short,
        "repo": spec.repo,
        "lineage": spec.lineage,
        "role": spec.role,
        "template": spec.template,
        "substitution": spec.substitution,
        "n_prompts": n,
        "runtime_s": round(time.time() - t0, 1),
        "metrics": metrics,
    }
    out_path = Path(args.out_metrics) if args.out_metrics else met_dir / f"{spec.short}.json"
    out_path.write_text(json.dumps(out, indent=1))
    logger.info(f"{spec.short}: metrics written to {out_path} ({round(time.time()-t0,1)}s total)")
    del H, LG, AT, hidden, logits, attn, gens, meta, cap
    gc.collect()
    return True


@logger.catch(reraise=True)
def main() -> None:
    configure_threads()
    log_hardware()
    args = parse_args()
    prompts = json.loads(BATTERY.read_text())
    if args.limit:
        prompts = prompts[: args.limit]
    logger.info(f"battery: {len(prompts)} prompts")

    specs = REGISTRY
    if args.models:
        want = {s.strip() for s in args.models.split(",")}
        specs = [s for s in REGISTRY if s.short in want]
        missing = want - {s.short for s in specs}
        if missing:
            raise SystemExit(f"unknown models: {missing}")

    failed = []
    with RSSWatchdog():
        for spec in specs:
            try:
                ok = run_model(spec, prompts, args)
                if not ok:
                    failed.append(spec.short)
            except Exception as e:  # noqa: BLE001
                logger.error(f"{spec.short}: FAILED {type(e).__name__}: {e}")
                gc.collect()
                failed.append(spec.short)
            gc.collect()
    logger.info(f"done; failed: {failed or 'none'}")
    if failed and len(failed) == len(specs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
