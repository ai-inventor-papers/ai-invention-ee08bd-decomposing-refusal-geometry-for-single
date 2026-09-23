#!/usr/bin/env python3
"""Weight-spectral metrics (b1..b3) — reads ONLY model weights, single model.

The invariant from the commissioning brief: >=3 shipped metrics must read hidden
states or weights of a single model. This module provides the weight-reading
metrics without any reference model:

  b1_o_proj_top1_sv_ratio : mean over layers of sigma_1 / sum(sigma_i) of the
        attention output projection (W_O). Safety finetunes concentrate output
        energy into fewer directions here (low-rank behavioural edits).
  b2_o_proj_spectral_gap  : mean over layers of sigma_1 / sigma_2 of W_O.
  b3_mlp_down_weight_kurtosis : max over layers of excess kurtosis of W_down
        (MLP down-projection) entries. Abliteration-style direction edits leave
        heavy-tailed, low-rank damage in downstream projections; RL finetuning
        spreads weight changes.

Memory: streams one weight matrix at a time via safetensors metadata mmap; the
full model is never materialised.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ACT = RES / "activations"

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
from scripts.extract import MODEL_PANEL  # noqa: E402  (panel definition lives there)


def _find_repo_dir(repo: str) -> Path:
    cache = Path("/ai-inventor/aii_data/runs/run_K2ftFsEsyOu3/.shared_cache/hf/hub")
    slug = "models--" + repo.replace("/", "--")
    snaps = sorted((cache / slug / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError(f"no local snapshot for {repo}")
    return snaps[-1]


def _singular_values(mat: np.ndarray) -> np.ndarray:
    """Singular values via eigvalsh of the smaller Gram matrix (fast on CPU)."""
    m = mat.astype(np.float32)
    if m.shape[0] > m.shape[1]:
        g = m.T @ m
    else:
        g = m @ m.T
    ev = np.linalg.eigvalsh(g.astype(np.float64))
    s = np.sqrt(np.clip(ev, 0.0, None))
    return np.sort(s)[::-1]


def compute_weight_metrics(tag: str) -> dict:
    cfg = MODEL_PANEL[tag]
    snap = _find_repo_dir(cfg["repo"])
    idx = snap / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
    else:
        wm = {}
        for f in snap.glob("*.safetensors"):
            with safe_open(f, framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    wm[k] = f.name

    n_layers = 36
    sv_ratio = np.zeros(n_layers)
    sv_gap = np.zeros(n_layers)
    kurt = np.zeros(n_layers)
    found = 0
    for li in range(n_layers):
        for name, out in (
            (f"model.layers.{li}.self_attn.o_proj.weight", "o"),
            (f"model.layers.{li}.mlp.down_proj.weight", "d"),
        ):
            fname = wm.get(name)
            if fname is None:
                continue
            with safe_open(snap / fname, framework="pt", device="cpu") as fh:
                w = fh.get_tensor(name)
            m = w.to(torch.float32).numpy()
            del w
            if out == "o":
                s = _singular_values(m)
                sv_ratio[li] = float(s[0] / max(s.sum(), 1e-30))
                sv_gap[li] = float(s[0] / max(s[1], 1e-30))
                del s
            else:
                x = m.flatten()
                mu, sd = float(x.mean()), float(x.std())
                kurt[li] = float(((x - mu) ** 4).mean() / max(sd**4, 1e-30) - 3.0) if sd > 0 else 0.0
                del x
            found += 1
            del m
            gc.collect()
    if found < 72:
        logger.warning(f"[{tag}] only {found}/72 weight tensors found")
    return dict(
        b4_o_proj_top1_sv_ratio_mean=float(sv_ratio.mean()),
        b5_o_proj_spectral_gap_mean=float(sv_gap.mean()),
        b6_mlp_down_weight_kurtosis_max=float(kurt.max()),
        b4_o_proj_top1_sv_ratio_max=float(sv_ratio.max()),
        layer_curves=dict(
            b4_o_proj_top1_sv_ratio_per_layer={li + 1: float(v) for li, v in enumerate(sv_ratio)},
            b5_o_proj_spectral_gap_per_layer={li + 1: float(v) for li, v in enumerate(sv_gap)},
            b6_mlp_down_weight_kurtosis_per_layer={li + 1: float(v) for li, v in enumerate(kurt)},
        ),
    )


def main() -> None:
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODEL_PANEL)
    out: dict[str, dict] = {}
    path = RES / "weight_metrics.json"
    if path.exists():
        out = json.loads(path.read_text())
    for tag in tags:
        logger.info(f"weight metrics for {tag}")
        try:
            out[tag] = compute_weight_metrics(tag)
        except Exception as e:
            logger.error(f"[{tag}] weight metrics failed: {type(e).__name__}: {e}")
            raise
        path.write_text(json.dumps(out, indent=2))
        logger.info(f"[{tag}] saved -> {path}")
    logger.info("weight metrics done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"weight metrics crashed: {type(e).__name__}: {e}")
        sys.exit(1)
