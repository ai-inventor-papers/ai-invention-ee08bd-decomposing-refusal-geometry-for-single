#!/usr/bin/env python3
"""Weight-spectral metrics for all four Qwen3-4B variants, read straight from
safetensors (no model instantiation -> few hundred MB RAM, no meta-device traps).

Per layer (0..35) and per matrix family (self_attn.o_proj, mlp.down_proj):
  - top1_energy   = s1^2 / sum(s_i^2)                (dominant direction energy)
  - sigma1_sigma2 = s1 / s2                          (spectral gap)
  - participation_ratio = (sum s^2)^2 / sum s^4
  - stable_rank   = sum(s^2) / s1^2
  - column_norm_entropy over input columns
Plus unembedding row-norm statistics for refusal vs compliance token groups
(weight-level analogue of the decision-point margin).

Deterministic: exact eigvalsh of the Gram matrix (G = W W^T, 2560x2560).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SHARED_HF = Path("/ai-inventor/aii_data/runs/run_K2ftFsEsyOu3/.shared_cache/hf/hub")

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(ROOT / "logs" / "weights_all.log", rotation="30 MB", level="DEBUG", backtrace=True)

MODELS: dict[str, str] = {
    "base": "Qwen/Qwen3-4B-Base",
    "instruct": "Qwen/Qwen3-4B",
    "saferl": "Qwen/Qwen3-4B-SafeRL",
    "abli": "mlabonne/Qwen3-4B-abliterated",
}
N_LAYERS = 36
FAMILIES = {"o_proj": "self_attn.o_proj.weight", "down_proj": "mlp.down_proj.weight"}
LOCAL_COPIES = ROOT / "models_local"

REFUSAL_TOKENS = ["I", "I'm", "Sorry", "sorry", "As", "Unfortunately", "cannot", "can't",
                  "won't", "unable", "must", "apologize", "against", "decline", "not",
                  "No", "never", "However", "That"]
COMPLIANCE_TOKENS = ["Sure", "Here", "Okay", "Certainly", "Step", "First", "Let", "Great",
                     "Absolutely", "Yes", "To", "The", "A", "Definitely", "course"]


def snapshot_dir(repo: str) -> Path:
    tag_dir = "models--" + repo.replace("/", "--")
    snaps = sorted((SHARED_HF / tag_dir).glob("snapshots/*/"))
    if not snaps:
        raise FileNotFoundError(f"no cached snapshot for {repo}")
    return snaps[-1]


def sv_stats(W: np.ndarray) -> dict[str, float]:
    """Exact spectral statistics from the Gram matrix eigenvalues."""
    G = W @ W.T
    ev = np.linalg.eigvalsh(G)
    ev = np.clip(ev, 0.0, None)[::-1]           # descending squared singular values
    tot = float(ev.sum())
    if tot <= 1e-12:
        return {k: float("nan") for k in
                ("top1_energy", "sigma1_sigma2", "participation_ratio", "stable_rank")}
    s1, s2 = math.sqrt(float(ev[0])), math.sqrt(float(ev[1])) if len(ev) > 1 else 0.0
    return {
        "top1_energy": float(ev[0]) / tot,
        "sigma1_sigma2": float(s1 / max(s2, 1e-12)),
        "participation_ratio": float(tot ** 2 / (ev ** 2).sum()),
        "stable_rank": float(tot / max(float(ev[0]), 1e-12)),
        "n_layers_effective": float(tot),  # sum of squared svals (absolute scale)
    }


def col_norm_entropy(W: np.ndarray) -> float:
    cn = np.linalg.norm(W, axis=0)
    p = cn / cn.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def tensor_index(tag: str) -> dict[str, Path]:
    from safetensors import safe_open
    local = LOCAL_COPIES / tag
    src = local if (local / "config.json").exists() else snapshot_dir(MODELS[tag])
    index: dict[str, Path] = {}
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, framework="numpy") as sf:
            for k in sf.keys():
                index[k] = f
    return index


def read_tensor(index: dict[str, Path], key: str) -> np.ndarray:
    """Read a safetensors tensor as float32 numpy (bf16 handled via torch)."""
    import torch
    from safetensors import safe_open
    with safe_open(index[key], framework="pt") as sf:
        t = sf.get_tensor(key)
    return t.to(torch.float32).numpy()


def process(tag: str) -> Path:
    out_path = RESULTS / f"weights_full_{tag}.json"
    if out_path.exists():
        logger.info(f"[{tag}] exists -> skip")
        return out_path
    t0 = time.time()
    idx = tensor_index(tag)
    res: dict[str, Any] = {"tag": tag, "repo": MODELS[tag], "per_layer": {}}
    for li in range(N_LAYERS):
        entry: dict[str, Any] = {}
        for fam, suffix in FAMILIES.items():
            key = f"model.layers.{li}.{suffix}"
            if key not in idx:
                logger.warning(f"[{tag}] missing {key}")
                continue
            W = read_tensor(idx, key)
            st = sv_stats(W)
            st["column_norm_entropy"] = col_norm_entropy(W)
            entry[fam] = st
            del W
        res["per_layer"][li] = entry
        if (li + 1) % 8 == 0:
            logger.info(f"[{tag}] layers {li + 1}/{N_LAYERS} ({time.time() - t0:.0f}s)")
            gc.collect()

    # ---- unembedding row norms ----
    E = read_tensor(idx, "model.embed_tokens.weight")     # [V, d] float32 ~1.5GB
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot_dir(MODELS[tag])))
    norms = np.linalg.norm(E, axis=1)
    del E
    gc.collect()
    def row_norms(tokens: list[str]) -> list[float]:
        out = []
        for s in tokens:
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                out.append(float(norms[e[0]]))
        return out
    ref_n, com_n = row_norms(REFUSAL_TOKENS), row_norms(COMPLIANCE_TOKENS)
    res["unembed"] = {
        "mean_row_norm": float(norms.mean()),
        "std_row_norm": float(norms.std()),
        "gini_row_norm": _gini(norms),
        "mean_norm_refusal_rows": float(np.mean(ref_n)) if ref_n else None,
        "mean_norm_compliance_rows": float(np.mean(com_n)) if com_n else None,
        "norm_focus_refusal": (float((np.mean(ref_n) - np.mean(com_n)) / max(float(norms.mean()), 1e-9))
                               if ref_n and com_n else None),
        "n_refusal_rows": len(ref_n), "n_compliance_rows": len(com_n),
    }
    res["seconds"] = time.time() - t0
    out_path.write_text(json.dumps(res, indent=1))
    logger.info(f"[{tag}] weights done in {time.time() - t0:.0f}s -> {out_path.name}")
    return out_path


def _gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    c = np.cumsum(x)
    return float((n + 1 - 2 * (c / c[-1]).sum()) / n)


if __name__ == "__main__":
    only = sys.argv[1:] or list(MODELS)
    for t in only:
        try:
            process(t)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[{t}] FAILED: {e}")
            raise
