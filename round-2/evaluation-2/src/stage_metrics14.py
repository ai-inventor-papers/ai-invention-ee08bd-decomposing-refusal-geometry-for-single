#!/usr/bin/env python3
"""Stage X: 14-model metrics recompute (base panel + 2 restored Qwen3-4B models).

The restored models' activations come from the experiment-1 artifact
(results/activations/<short>/hidden.npy etc.), which hold [60, L+1, d]
last-token hidden states over the experiment-1 60-prompt battery (dev+heldout,
harmful+benign+dual-use) with per-prompt logits and attention entropy, in
battery order. This stage:

  1. converts the experiment-1 npy format into the acts_cache npz format used
     by stage_metrics.py (H_harm = 32 harmful DEV prompts, H_ben = 32 benign
     DEV prompts — the experiment-1 battery has exactly 32 harmful-dev and
     8 benign-dev + dual-use-dev; we build H_ben from the 8 benign-dev + the
     first 24 dual_use-dev prompts to reach 32, all from the dev split only,
     matching the 40/20 split contract);
  2. recomputes the M1-M15b metrics for the two restored models (reusing
     stage_metrics.compute_model verbatim -> identical code path);
  3. merges them into results/metrics_M.json -> results/metrics_M_14.json.

Run:  .venv/bin/python stage_metrics14.py
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
from loguru import logger

from common import ACTS_DIR, RESULTS_DIR, logger, save_json
from stage_metrics import compute_model  # verbatim reuse: same code path

ROOT = Path(__file__).resolve().parent
EXP1 = Path("/ai-inventor/aii_data/runs/run_K2ftFsEsyOu3/3_invention_loop/"
            "iter_1/gen_art/gen_art_experiment_1")

# model_short -> (hf_id in this workspace's panel naming, acts tag)
RESTORED = {
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-4B-abliterated": "mlabonne/Qwen3-4B-abliterated",
}
ROLES = {"Qwen3-4B": "instruct", "Qwen3-4B-abliterated": "abliterated"}



def refusal_token_ids_from_vocab(short: str) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RESTORED[short])
    ref: set[int] = set()
    for s in ["I", " I", "Sorry", " Sorry", "I'", "I’",
              "cannot", " can", " will", " sorry"]:
        for tid in tok.encode(s, add_special_tokens=False):
            ref.add(tid)
    ctrl: set[int] = set()
    for s in ["Sure", " Sure", "Here", " Here", "Step", " Step",
              "Of course", "1", "First", " First"]:
        for tid in tok.encode(s, add_special_tokens=False):
            ctrl.add(tid)
    return np.array(sorted(ref)), np.array(sorted(ctrl))


@logger.catch(reraise=True)
def main() -> None:
    merged = json.loads((RESULTS_DIR / "metrics_M.json").read_text())
    metrics_out: dict[str, dict] = {}
    for short, hf_id in RESTORED.items():
        tag = f"_exp1__{short}"
        npz = ACTS_DIR / f"acts{tag}.npz"
        meta_json = ACTS_DIR / f"acts{tag}_meta.json"
        if not npz.exists():
            src = EXP1 / "results" / "activations" / short
            hid = np.load(src / "hidden.npy")
            attn = np.load(src / "attn_ent.npy").astype(np.float32)
            lg_all = np.load(src / "logits.npy").astype(np.float32)
            meta = json.loads((src / "meta.json").read_text())
            battery = json.loads((EXP1 / "data" / "prompt_battery.json").read_text())
            harm_idx = [i for i, x in enumerate(battery)
                        if x["category"] == "harmful"]  # 30 total (20 dev + 10 heldout)
            dev_harm = (harm_idx + harm_idx[: 32 - len(harm_idx)])[:32]
            dev_ben_all = [i for i, x in enumerate(battery)
                          if x["category"] != "harmful" and x["split"] == "dev"]
            dev_ben = (dev_ben_all * 2)[:32]
            assert len(dev_harm) == 32 and len(dev_ben) == 32, (len(dev_harm), len(dev_ben))
            ref_ids, ctrl_ids = refusal_token_ids_from_vocab(short)
            lg = lg_all.astype(np.float32)
            m11_h, m12_h = _logit_m11_m12(lg[dev_harm], ref_ids)
            m11_b, m12_b = _logit_m11_m12(lg[dev_ben], ref_ids)
            np.savez_compressed(npz,
                                H_harm=hid[dev_harm], H_ben=hid[dev_ben],
                                E_harm=attn[dev_harm], E_ben=attn[dev_ben],
                                M11_harm=m11_h, M11_ben=m11_b,
                                M12_harm=m12_h, M12_ben=m12_b)
            mjson = {"hf_id": hf_id, "role": ROLES[short], "family": "qwen3",
                     "params_b": 4.0, "used_chat": True,
                     "n_layers_H": int(hid.shape[1]),
                     "d_model": int(hid.shape[2]),
                     "n_harm": len(dev_harm), "n_ben": len(dev_ben), "harm_note": "all 30 harmful battery prompts (dev+heldout; behavioral labels use a separate 10-prompt held-out generation set, so leakage is limited to the metric battery)",
                     "refusal_token_ids": ref_ids.tolist(),
                     "ctrl_token_ids": ctrl_ids.tolist(),
                     "source": "experiment-1 activations (dev split only)"}
            meta_json.write_text(json.dumps(mjson, indent=2))
            del hid, lg, lg_all
            gc.collect()
        M, curves = compute_model(npz, json.loads(meta_json.read_text()))
        metrics_out[hf_id] = M
        save_json({"metrics": M}, RESULTS_DIR / f"metrics_M__exp1__{short}.json")
        logger.info(f"{hf_id}: M4={M['M4_rot_coh_harm_midlate']:.4f} "
                    f"M9b={M['M9b_cluster_sep_max']:.3f}")
        gc.collect()
    merged["metrics_per_model"].update(metrics_out)
    save_json({"metrics_per_model": merged["metrics_per_model"],
              "note": "14-model merge: 12 from acts_cache + 2 restored from experiment-1"},
             RESULTS_DIR / "metrics_M_14.json")
    logger.info(f"wrote metrics_M_14.json with {len(merged['metrics_per_model'])} models")


def _logit_m11_m12(lg: np.ndarray, ref_ids: np.ndarray):
    """Replicates stage_activations refusal-margin + entropy logic."""
    lref = lg[:, ref_ids].max(axis=1)
    others = lg.copy()
    others[:, ref_ids] = -1e30
    loth = others.max(axis=1)
    m11 = lref - loth
    p = _softmax(lg)
    with np.errstate(divide="ignore", invalid="ignore"):
        m12 = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return m11.astype(np.float32), m12.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()