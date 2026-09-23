#!/usr/bin/env python3
"""Qwen3-4B single-model safety-metric discovery via activation analysis — driver.

Question: is there a cheap internal metric (weights or activations only, single
model, no reference model) that separates safety-tuned from abliterated variants,
and where in the network does the signal live?

Panel (10 models, 3 lineages + 1 second-architecture family, each with its
safety roles — see src/registry.py):
  base / instruct / saferl / abliterated across qwen3-4b, qwen3-4b-2507,
  qwen3-1.7b, and qwen2.5-3b (instruct + abliterated-SFT pair).

Battery: 60 prompts = 30 harmful (AdvBench, Or-Bench) + 15 benign (XSTest v2,
Anthropic HH harmless) + 15 hand-written dual-use; stratified 40 dev / 20 held-out.

Pipeline (the real implementation lives in the src/ package; this driver
orchestrates it and materialises the deliverables at the workspace root):
  1. check    - verify panel artifacts (per-model metrics + activation tensors)
  2. extract  - src/run_model.py: per model, bf16 transformers forward passes
                capturing all-layer last-token residual streams, attention
                entropy, decision logits (+ weight-spectral reads); ~4 min/model
                on 2 CPUs, skipped for models already extracted
  3. analyze  - src/analyze.py: metric table, safe-vs-abliterated contrasts with
                bootstrap CIs, rank correlation vs published refusal anchors,
                held-out confirmation, zero-shot probe transfer, figures
  4. deliver  - copy results/method_out.json to the workspace root and generate
                full/mini/preview variants (mini = first 3 examples per dataset,
                preview = same with strings truncated to 200 chars)

Usage:
  uv run method.py                # full pipeline (skips finished work)
  uv run method.py --phase check  # artifact inventory only
  uv run method.py --phase analyze  # aggregation only (needs extract done)
"""

from __future__ import annotations

import os

# threads before torch import (host has many cores, container is capped at 2)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import shutil
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(ROOT / "logs" / "method.log", rotation="30 MB", level="DEBUG", backtrace=True)

EXPECTED_PANEL = [
    "Qwen2.5-3B-Instruct", "Qwen2.5-3B-abliterated", "Qwen3-1.7B", "Qwen3-1.7B-Base",
    "Qwen3-4B", "Qwen3-4B-Base", "Qwen3-4B-Instruct-2507",
    "Qwen3-4B-Instruct-2507-abliterated", "Qwen3-4B-SafeRL", "Qwen3-4B-abliterated",
]


def phase_check() -> bool:
    """Verify the panel artifacts needed by the analysis exist."""
    missing: list[str] = []
    for m in EXPECTED_PANEL:
        if not (ROOT / "results" / "metrics" / f"{m}.json").exists():
            missing.append(f"metrics/{m}.json")
        if not (ROOT / "results" / "activations" / m / "hidden.npy").exists():
            missing.append(f"activations/{m}/hidden.npy")
    if missing:
        logger.warning(f"missing panel artifacts: {missing}")
        return False
    logger.info(f"panel complete: {len(EXPECTED_PANEL)} models with metrics + activations")
    return True


def phase_extract() -> None:
    """Run the per-model extraction (idempotent; finished models are skipped)."""
    from src.run_model import main as run_model_main

    sys.argv = ["run_model.py", "--skip-gen"]
    run_model_main()


def phase_analyze() -> None:
    """Aggregate the panel into the final analysis artifacts."""
    from src.analyze import main as analyze_main

    analyze_main()


def _trunc(obj: object, limit: int = 200) -> object:
    if isinstance(obj, str):
        return obj[:limit]
    if isinstance(obj, list):
        return [_trunc(v, limit) for v in obj]
    if isinstance(obj, dict):
        return {k: _trunc(v, limit) for k, v in obj.items()}
    return obj


def phase_deliver() -> None:
    """Materialise method_out.json + full/mini/preview at the workspace root."""
    src = ROOT / "results" / "method_out.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing - run the analyze phase first")
    shutil.copyfile(src, ROOT / "method_out.json")

    data = json.loads(src.read_text())
    full = ROOT / "full_method_out.json"
    full.write_text(json.dumps(data, indent=1))
    n_total = sum(len(ds["examples"]) for ds in data["datasets"])
    logger.info(f"wrote {full.name} ({n_total} examples)")

    mini = {k: v for k, v in data.items() if k != "datasets"}
    mini["datasets"] = [
        {**ds, "examples": ds["examples"][:3]} for ds in data["datasets"]
    ]
    (ROOT / "mini_method_out.json").write_text(json.dumps(mini, indent=1))

    preview = _trunc(mini)
    (ROOT / "preview_method_out.json").write_text(json.dumps(preview, indent=1))
    logger.info("wrote mini_method_out.json, preview_method_out.json")


@logger.catch(reraise=True)
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["check", "extract", "analyze", "deliver", "all"],
                    default="all", help="pipeline stage to run (default: all)")
    args = ap.parse_args()

    if args.phase == "check":
        phase_check()
        return

    if args.phase in ("all", "extract"):
        if not phase_check():
            logger.info("panel incomplete -> running extraction (long)")
            phase_extract()
            if not phase_check():
                raise RuntimeError("extraction finished but panel still incomplete")
        elif args.phase == "all":
            logger.info("panel already extracted -> skipping extract")

    if args.phase in ("all", "analyze"):
        phase_analyze()

    if args.phase in ("all", "deliver"):
        phase_deliver()

    logger.info("method.py complete")


if __name__ == "__main__":
    main()
