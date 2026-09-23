#!/usr/bin/env python3
"""Stage E: correlate every metric with behavioral ground truth, bootstrap
CIs, group comparisons, verdict. Writes results/correlations.json.

Ground truth per model (declared precedence):
  1. Sibling experiment artifact's behavioral anchor for a matching tag
     (gen_art_experiment_1/results/metrics_per_model.json, z_* fields).
  2. This artifact's own greedy-generation keyword labels (B1/B2/B3), which
     exist for the whole panel.
B3 = B1 - B2 is the primary target (harmfulness-adjusted refusal).

Model resampling: 1000-resample nonparametric bootstrap over models for
Spearman rho CIs. Prompt resampling: 1000-resample bootstrap over the 10+10
generation prompts for B3 uncertainty (reported separately, per plan).
Group stats: safe-role (instruct/saferl/base... non-abliterated) vs
abliterated AUROC per metric; per-family and size-matched subgroup rhos.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import spearmanr

from common import RESULTS_DIR, logger, save_json

ROOT = Path(__file__).resolve().parent
SIB_METRICS = Path(
    "/ai-inventor/aii_data/runs/run_K2ftFsEsyOu3/3_invention_loop/iter_1/"
    "gen_art/gen_art_experiment_1/results/metrics_per_model.json")
N_BOOT = 1000
SEED = 17

# sibling tag -> hf_id mapping (sibling MODEL_TAGS)
SIB_TAGS = {
    "base": "Qwen/Qwen3-4B-Base",
    "instruct": "Qwen/Qwen3-4B-Instruct-2507",   # sibling used Qwen3-4B; record note
    "saferl": "Qwen/Qwen3-4B-SafeRL",
    "abliterated": "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated",
}


def cohens_kappa(a: list[int], b: list[int]) -> float:
    a = np.asarray(a); b = np.asarray(b)
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    if pe >= 1:
        return 1.0
    return float((po - pe) / (1 - pe))


def boot_rho_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), (float("nan"), float("nan"))
    r0 = spearmanr(x, y).statistic
    stats = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        stats.append(spearmanr(x[idx], y[idx]).statistic)
    if not stats:
        return float(r0), (float("nan"), float("nan"))
    return float(r0), (float(np.percentile(stats, 2.5)),
                       float(np.percentile(stats, 97.5)))


def prompt_boot_b3(rec: dict, n_boot: int, seed: int):
    """Resample the 10 harmful and 10 benign generation prompts; report the
    sd of B3 and the probability that B3 ranking flips are driven by prompt
    choice (as a CI on B3 itself)."""
    rng = np.random.default_rng(seed + 1)
    h = [r["kw_refusal"] for r in rec["responses"]["harmful"]]
    b = [r["kw_refusal"] for r in rec["responses"]["benign"]]
    h, b = np.asarray(h), np.asarray(b)
    stats = []
    for _ in range(n_boot):
        hi = rng.integers(0, len(h), len(h))
        bi = rng.integers(0, len(b), len(b))
        stats.append(h[hi].mean() - b[bi].mean())
    return float(np.mean(h) - np.mean(b)), \
        (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def auroc_safe_vs_abl(values: dict[str, float], roles: dict[str, str],
                      families: dict[str, str]) -> float:
    """AUROC that the metric separates non-abliterated (safe-ish) models from
    abliterated models, counting each (safe, abliterated) within-family pair;
    >0.5 = higher metric value means safer."""
    wins, tot = 0.0, 0
    for s, a in combinations(values, 2):
        if {roles[s], roles[a]} == {"abliterated"}:
            continue
        if (roles[s] == "abliterated") != (roles[a] == "abliterated"):
            if families[s] != families[a]:
                continue  # within-family only
            sv, av = values[s], values[a]
            tot += 1
            if sv > av:
                wins += 1
            elif sv == av:
                wins += 0.5
    return wins / tot if tot else float("nan")


@logger.catch(reraise=True)
def main() -> None:
    mets = json.loads((RESULTS_DIR / "metrics_M.json").read_text())["metrics_per_model"]
    behaviors = {}
    for f in RESULTS_DIR.glob("behavior_*.json"):
        rec = json.loads(f.read_text())
        behaviors[rec["hf_id"]] = rec

    # ---- ground truth assembly ----
    gt = {}
    gt_source = {}
    sib = json.loads(SIB_METRICS.read_text())["metrics_per_model"] if SIB_METRICS.exists() else {}
    for tag, rec in behaviors.items():
        gt[tag] = {"B1": rec["B1"], "B2": rec["B2"], "B3": rec["B3"]}
        gt_source[tag] = "own_keywords"
    for stag, hf in SIB_TAGS.items():
        if stag in sib and hf in gt:
            z = sib[stag]
            gt[hf]["sibling_B1"] = z.get("z_refusal_rate_harmful")
            gt[hf]["sibling_B2"] = z.get("z_false_refusal_rate_benign")
            gt[hf]["sibling_B3"] = (z.get("z_refusal_rate_harmful", 0)
                                    - z.get("z_false_refusal_rate_benign", 0))
            gt_source[hf] = "sibling+own"

    common = sorted(set(mets) & set(gt))
    logger.info(f"models with metrics+gt: {len(common)}")
    roles = {t: behaviors[t]["role"] for t in common}
    families = {t: behaviors[t]["family"] for t in common}
    sizes = {t: behaviors[t]["params_b"] for t in common}

    metric_names = sorted(next(iter(mets.values())).keys())
    rows = []
    for mn in metric_names:
        vals = np.array([mets[t].get(mn, np.nan) for t in common], float)
        ok = np.isfinite(vals)
        v, t = vals[ok], [c for c, o in zip(common, ok) if o]
        y = np.array([gt[c]["B3"] for c in t], float)
        r, (lo, hi) = boot_rho_ci(v, y, N_BOOT, SEED)
        y1 = np.array([gt[c]["B1"] for c in t], float)
        r1, _ = boot_rho_ci(v, y1, N_BOOT, SEED)
        y2 = np.array([gt[c]["B2"] for c in t], float)
        r2, _ = boot_rho_ci(v, y2, N_BOOT, SEED)
        # AUROC safe vs abliterated
        vv = {c: mets[c][mn] for c in t if np.isfinite(mets[c][mn])}
        au = auroc_safe_vs_abl(vv, roles, families)
        # per-family subgroup rho (qwen3 only if >=4 models)
        sub = {"all": (r, lo, hi, len(t))}
        for fam in sorted(set(families[c] for c in t)):
            tf = [c for c in t if families[c] == fam]
            if len(tf) >= 4:
                vf = np.array([mets[c][mn] for c in tf], float)
                yf = np.array([gt[c]["B3"] for c in tf], float)
                rf, (lf, hf_) = boot_rho_ci(vf, yf, N_BOOT, SEED)
                sub[fam] = (rf, lf, hf_, len(tf))
        rows.append({"metric": mn, "rho_B3": r, "ci95": [lo, hi], "n": len(t),
                     "rho_B1": r1, "rho_B2": r2,
                     "auroc_safe_vs_abl": au, "subgroup_rho": sub})

    # ---- group comparison: internal (M1-M10, M13-M15) vs logit-only (M11, M12) ----
    GROUPS = {
        "compression (M1-M3)": [m for m in metric_names if m.startswith(("M1_", "M2_", "M3_", "M9"))],
        "coherence (M4-M6)": [m for m in metric_names if m.startswith(("M4_", "M5_", "M6_", "M13"))],
        "coupling (M7-M8)": [m for m in metric_names if m.startswith(("M7_", "M8_"))],
        "norm/attention (M10, M14)": [m for m in metric_names if m.startswith(("M10", "M14"))],
        "logit-only (M11-M12)": [m for m in metric_names if m.startswith(("M11", "M12"))],
        "probe anchor (M15)": [m for m in metric_names if m.startswith("M15")],
    }
    group_best = {}
    for g, mlist in GROUPS.items():
        cand = [r for r in rows if r["metric"] in mlist and np.isfinite(r["rho_B3"])]
        if cand:
            best = max(cand, key=lambda r: r["rho_B3"])
            group_best[g] = {"metric": best["metric"], "rho_B3": best["rho_B3"],
                             "ci95": best["ci95"], "auroc": best["auroc_safe_vs_abl"]}

    # ---- prompt-unit bootstrap uncertainty of B3 ----
    b3_boot = {}
    for tag, rec in behaviors.items():
        med, ci = prompt_boot_b3(rec, N_BOOT, SEED)
        b3_boot[tag] = {"B3": med, "ci95": list(ci)}

    # ---- pairwise model separation: best metric per (safe, abliterated) pair ----
    pair_table = {}
    for r in rows:
        if not np.isfinite(r["auroc_safe_vs_abl"]):
            continue
        pair_table.setdefault("metric_aurocs", {})[r["metric"]] = r["auroc_safe_vs_abl"]

    save_json({
        "ground_truth": gt,
        "ground_truth_source": gt_source,
        "correlations": rows,
        "group_best": group_best,
        "b3_prompt_bootstrap": b3_boot,
        "n_boot": N_BOOT,
        "note": "sibling tag 'instruct' used Qwen/Qwen3-4B (thinking model); "
                "this artifact's instruct is Qwen3-4B-Instruct-2507 - sibling "
                "ground truth used only as a cross-check, own labels are primary",
    }, RESULTS_DIR / "correlations.json")
    logger.info("wrote correlations.json")


if __name__ == "__main__":
    main()
