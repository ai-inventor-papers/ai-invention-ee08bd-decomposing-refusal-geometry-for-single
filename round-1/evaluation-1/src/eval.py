#!/usr/bin/env python3
"""Evaluation stage (gen_art_evaluation_1): refusal baseline vs activation metrics.

Consumes cached artifacts produced by stages A-D in this workspace:
  results/behavior_<tag>.json    per-model greedy generation + keyword labels (B1/B2/B3)
  results/judge_labels.json      LLM-judge validation of keyword labels (kappa)
  acts_cache/acts_<tag>.npz      cached per-layer activations/logits per model
  results/metrics_M.json         M1-M15 metrics per model (recomputed if missing)
  results/correlations.json      rho table (recomputed if missing)

Produces results/eval_out.json in the exp_eval_sol_out schema:
  metadata       models, panel, metric definitions, verdict, runtime, judge cost
  metrics_agg    headline aggregates (best internal rho, best logit rho, delta,
                 safe-vs-abliterated AUROC of the best internal metric, ...)
  datasets       one dataset per analysis unit; per-model metrics as examples
                 with predict_* (metric values) and eval_* (behavioral truth)
                 fields so downstream stages can re-derive every number.

Run:  .venv/bin/python eval.py            (idempotent; reuses stage outputs)
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import spearmanr

from common import RESULTS_DIR, logger, save_json

ROOT = Path(__file__).resolve().parent
N_BOOT = 1000
SEED = 17
INTERNAL = "internal"   # hidden-state/weight readers
LOGIT = "logit"         # logit-only baselines (at most 2 shipped)
BEHAV = "behavioral"

# metric-name -> group mapping (fixed before looking at correlations)
def metric_group(name: str) -> str:
    n = name.split("_")[0]
    if n in ("M11", "M12"):
        return LOGIT
    if n in ("B1", "B2", "B3"):
        return BEHAV
    return INTERNAL

GROUPS = {
    "compression": ["M1_", "M2_", "M3_"],
    "coherence": ["M4_", "M5_", "M6_", "M13"],
    "coupling": ["M7_", "M8_"],
    "cluster_norm_att": ["M9", "M10", "M14"],
    "logit_only": ["M11", "M12"],
    "probe_anchor": ["M15"],
}


def group_of(name: str) -> str:
    for g, prefs in GROUPS.items():
        if any(name.startswith(p) for p in prefs):
            return g
    return "other"


def cohens_kappa(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return 1.0 if pe >= 1 else float((po - pe) / (1 - pe))


def boot_rho_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """Spearman rho + percentile bootstrap CI over model resampling."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), (float("nan"), float("nan"))
    r0 = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        stats.append(spearmanr(x[idx], y[idx]).statistic)
    if len(stats) < 50:
        return r0, (float("nan"), float("nan"))
    return r0, (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def prompt_boot_b3(rec: dict, n_boot: int = N_BOOT, seed: int = SEED):
    """Bootstrap B3 over the generation prompt sets (10 harmful + 10 benign)."""
    h = np.asarray([r["kw_refusal"] for r in rec["responses"]["harmful"]], float)
    b = np.asarray([r["kw_refusal"] for r in rec["responses"]["benign"]], float)
    rng = np.random.default_rng(seed + 1)
    stats = []
    for _ in range(n_boot):
        hi = rng.integers(0, len(h), len(h))
        bi = rng.integers(0, len(b), len(b))
        stats.append(h[hi].mean() - b[bi].mean())
    return float(h.mean() - b.mean()), (float(np.percentile(stats, 2.5)),
                                        float(np.percentile(stats, 97.5)))


def auroc_safe_vs_abl(values: dict, roles: dict, families: dict) -> float:
    """Directional AUROC separating non-abliterated vs abliterated models over
    within-family pairs (>0.5 = higher value means safer)."""
    wins = tot = 0
    for s, a in combinations(values, 2):
        if roles[s] == roles[a] == "abliterated":
            continue
        if (roles[s] == "abliterated") != (roles[a] == "abliterated"):
            if families[s] != families[a]:
                continue
            sv, av = values[s], values[a]
            tot += 1
            wins += 1 if sv > av else (0.5 if sv == av else 0)
    return wins / tot if tot else float("nan")


def ensure_stage_outputs() -> dict:
    """Recompute metrics_M.json / correlations.json if missing (uses cached
    activations and behavior files; no re-generation of responses)."""
    if not (RESULTS_DIR / "metrics_M.json").exists():
        logger.info("metrics_M.json missing -> running stage_metrics.py")
        import subprocess
        subprocess.run([str(ROOT / ".venv" / "bin" / "python"),
                        str(ROOT / "stage_metrics.py")], check=True,
                       cwd=str(ROOT), timeout=7200)
    if not (RESULTS_DIR / "correlations.json").exists():
        logger.info("correlations.json missing -> running stage_analysis.py")
        import subprocess
        subprocess.run([str(ROOT / ".venv" / "bin" / "python"),
                        str(ROOT / "stage_analysis.py")], check=True,
                       cwd=str(ROOT), timeout=7200)
    return {
        "mets": json.loads((RESULTS_DIR / "metrics_M.json").read_text())["metrics_per_model"],
        "corr": json.loads((RESULTS_DIR / "correlations.json").read_text()),
        "judge": json.loads((RESULTS_DIR / "judge_labels.json").read_text())
        if (RESULTS_DIR / "judge_labels.json").exists() else {},
    }


def main() -> None:
    t0 = time.time()
    D = ensure_stage_outputs()
    mets, corr, judge = D["mets"], D["corr"], D["judge"]

    behaviors = {}
    for f in sorted(RESULTS_DIR.glob("behavior_*.json")):
        rec = json.loads(f.read_text())
        behaviors[rec["hf_id"]] = rec
    common = sorted(set(mets) & set(behaviors))
    logger.info(f"models with metrics + behavior: {len(common)}")

    roles = {t: behaviors[t]["role"] for t in common}
    families = {t: behaviors[t]["family"] for t in common}
    gt = {t: behaviors[t] for t in common}

    # ---------------- per-metric correlation table (recomputed fresh here) ----
    metric_names = sorted(next(iter(mets.values())).keys())
    rows = []
    for mn in metric_names:
        vals = np.array([mets[t].get(mn, np.nan) for t in common], float)
        ok = np.isfinite(vals)
        v, tm = vals[ok], [c for c, o in zip(common, ok) if o]
        if len(v) < 4:
            continue
        yB3 = np.array([gt[c]["B3"] for c in tm], float)
        r3, ci3 = boot_rho_ci(v, yB3)
        r1, _ = boot_rho_ci(v, np.array([gt[c]["B1"] for c in tm], float))
        r2, _ = boot_rho_ci(v, np.array([gt[c]["B2"] for c in tm], float))
        vv = {c: mets[c][mn] for c in tm}
        au = auroc_safe_vs_abl(vv, roles, families)
        subgroup = {}
        for fam in sorted(set(families[c] for c in tm)):
            tf = [c for c in tm if families[c] == fam]
            if len(tf) >= 4:
                vf = np.array([mets[c][mn] for c in tf], float)
                yf = np.array([gt[c]["B3"] for c in tf], float)
                rf, cf = boot_rho_ci(vf, yf)
                subgroup[fam] = {"rho_B3": rf, "ci95": list(cf), "n": len(tf)}
        rows.append({"metric": mn, "group": group_of(mn),
                     "rho_B3": r3, "ci95_B3": list(ci3), "n": len(tm),
                     "rho_B1": r1, "rho_B2": r2,
                     "auroc_safe_vs_abl": au, "subgroup_rho": subgroup})
    rows.sort(key=lambda r: -(r["rho_B3"] if np.isfinite(r["rho_B3"]) else -9))

    # ---------------- group comparison: internal vs logit-only (delta rho) ----
    def best_of(group: str) -> dict | None:
        cand = [r for r in rows if r["group"] == group and np.isfinite(r["rho_B3"])]
        if not cand:
            return None
        b = max(cand, key=lambda r: r["rho_B3"])
        return {"metric": b["metric"], "rho_B3": b["rho_B3"], "ci95": b["ci95_B3"],
                "auroc_safe_vs_abl": b["auroc_safe_vs_abl"], "n": b["n"]}

    internal_best = best_of("compression")  # placeholder; true best over internal groups below
    for g in ("coherence", "coupling", "cluster_norm_att", "probe_anchor"):
        cand = best_of(g)
        if cand and (internal_best is None or cand["rho_B3"] > internal_best["rho_B3"]):
            internal_best = cand
    logit_best = best_of("logit_only")
    delta_rho = (internal_best["rho_B3"] - logit_best["rho_B3"]
                 if internal_best and logit_best else float("nan"))
    beats_by_01 = bool(np.isfinite(delta_rho) and delta_rho >= 0.1)

    # safe-vs-abliterated AUROC of best internal metric vs best logit metric
    auroc_int = internal_best["auroc_safe_vs_abl"] if internal_best else float("nan")
    auroc_logit = logit_best["auroc_safe_vs_abl"] if logit_best else float("nan")

    # per-family breakdown of the headline comparison
    fam_best = {}
    for fam in sorted(set(families.values())):
        tf = [t for t in common if families[t] == fam]
        if len(tf) < 4:
            fam_best[fam] = {"n": len(tf), "note": "too few models for rank stats"}
            continue
        fb_i, fb_l = None, None
        for r in rows:
            sr = r["subgroup_rho"].get(fam)
            if not sr or not np.isfinite(sr["rho_B3"]):
                continue
            if r["group"] in ("logit_only",):
                if fb_l is None or sr["rho_B3"] > fb_l["rho_B3"]:
                    fb_l = {"metric": r["metric"], "rho_B3": sr["rho_B3"]}
            else:
                if fb_i is None or sr["rho_B3"] > fb_i["rho_B3"]:
                    fb_i = {"metric": r["metric"], "rho_B3": sr["rho_B3"]}
        fam_best[fam] = {"n": len(tf), "best_internal": fb_i, "best_logit": fb_l}

    # ---------------- judge validation: kappa keyword vs judge -----------------
    judge_stats = {}
    for hf_id, labs in judge.get("labels", {}).items():
        bpath = RESULTS_DIR / f"behavior_{hf_id.replace('/', '__')}.json"
        if not bpath.exists():
            continue
        rec = json.loads(bpath.read_text())
        kw = ([r["kw_refusal"] for r in rec["responses"]["harmful"]]
              + [r["kw_refusal"] for r in rec["responses"]["benign"]])
        jl = labs["harmful"] + labs["benign"]
        k = cohens_kappa(kw, jl)
        # judge-adjusted B values (judge label missing = -1 -> drop that prompt)
        h = [x for x, j in zip([r["kw_refusal"] for r in rec["responses"]["harmful"]],
                               labs["harmful"]) if j >= 0]
        bh = [j for j in labs["harmful"] if j >= 0]
        b2 = [x for x, j in zip([r["kw_refusal"] for r in rec["responses"]["benign"]],
                                labs["benign"]) if j >= 0]
        bj = [j for j in labs["benign"] if j >= 0]
        B1j = float(np.mean(bh)) if bh else float("nan")
        B2j = float(np.mean(bj)) if bj else float("nan")
        # judge and keyword may disagree; judge-adjusted uses judge labels
        judge_stats[hf_id] = {"kappa": k, "n_judged": len(jl),
                              "B1_judge": B1j, "B2_judge": B2j,
                              "B3_judge": B1j - B2j}

    # ---------------- prompt-unit bootstrap of B3 ----------------
    b3_boot = {t: dict(zip(("B3", "ci95"), prompt_boot_b3(gt[t]))) for t in common}

    # ---------------- SafeRL-excluded sensitivity analysis ----------------------
    # SafeRL is a thinking model cut at 24 tokens; its keyword labels are the
    # least reliable of the panel (judge kappa < 0). Recompute the headline
    # comparison without it.
    common_x = [t for t in common if "SafeRL" not in t]
    rows_x = []
    for mn in metric_names:
        vals = np.array([mets[t].get(mn, np.nan) for t in common_x], float)
        ok = np.isfinite(vals)
        v, tm = vals[ok], [c for c, o in zip(common_x, ok) if o]
        if len(v) < 6 or np.std(v) == 0:
            continue
        yx = np.array([gt[c]["B3"] for c in tm], float)
        if np.std(yx) == 0:
            continue
        rx, _ = boot_rho_ci(v, yx, n_boot=1000)
        rows_x.append({"metric": mn, "rho_B3_excl_SafeRL": rx, "n": len(tm)})
    rows_x.sort(key=lambda r: -r["rho_B3_excl_SafeRL"])
    bi_x = max((r for r in rows_x if group_of(r["metric"]) != LOGIT),
               key=lambda r: r["rho_B3_excl_SafeRL"], default=None)
    bl_x = max((r for r in rows_x if group_of(r["metric"]) == LOGIT),
               key=lambda r: r["rho_B3_excl_SafeRL"], default=None)
    excl_safel = {
        "best_internal": bi_x, "best_logit": bl_x,
        "delta": (bi_x["rho_B3_excl_SafeRL"] - bl_x["rho_B3_excl_SafeRL"])
        if bi_x and bl_x else float("nan"),
        "top6": rows_x[:6],
    }

    # ---------------- datasets for eval_out.json -------------------------------
    # dataset 1: per-model behavioral ground truth (one example per model)
    ex_behavior = []
    for t in common:
        ex_behavior.append({
            "input": gt[t]["hf_id"],
            "output": json.dumps({"B1": gt[t]["B1"], "B2": gt[t]["B2"],
                                  "B3": gt[t]["B3"], "role": gt[t]["role"],
                                  "family": gt[t]["family"],
                                  "params_b": gt[t]["params_b"]}),
            "metadata_role": gt[t]["role"],
            "metadata_family": gt[t]["family"],
            "eval_B1": float(gt[t]["B1"]),
            "eval_B2": float(gt[t]["B2"]),
            "eval_B3": float(gt[t]["B3"]),
            "eval_B1_judge": judge_stats.get(t, {}).get("B1_judge", float("nan"))
            if judge_stats.get(t) else float("nan"),
            "eval_kappa_kw_judge": judge_stats.get(t, {}).get("kappa", float("nan"))
            if judge_stats.get(t) else float("nan"),
        })
    # dataset 2: per-model internal metric predictions (predict_* fields)
    ex_metrics = []
    for t in common:
        row = {"input": t, "output": json.dumps({"role": roles[t], "family": families[t]})}
        for mn in metric_names:
            val = mets[t].get(mn, float("nan"))
            if np.isfinite(val):
                row[f"predict_{mn}"] = f"{val:.6g}"
        for bkey in ("B1", "B2", "B3"):
            row[f"eval_{bkey}"] = float(gt[t][bkey])
        ex_metrics.append(row)
    # dataset 3: correlation table (one example per metric)
    ex_corr = []
    for r in rows:
        ex_corr.append({
            "input": r["metric"],
            "output": json.dumps(r),
            "metadata_group": r["group"],
            "eval_rho_B3": r["rho_B3"] if np.isfinite(r["rho_B3"]) else 0.0,
            "eval_rho_B1": r["rho_B1"] if np.isfinite(r["rho_B1"]) else 0.0,
            "eval_rho_B2": r["rho_B2"] if np.isfinite(r["rho_B2"]) else 0.0,
            "eval_auroc_safe_vs_abl": r["auroc_safe_vs_abl"]
            if np.isfinite(r["auroc_safe_vs_abl"]) else 0.0,
            "eval_n_models": float(r["n"]),
        })

    # ---------------- headline aggregates + verdict ----------------------------
    metrics_agg = {
        "n_models_with_metrics_and_behavior": float(len(common)),
        "n_metrics_computed": float(len(metric_names)),
        "rho_B3_best_internal": internal_best["rho_B3"] if internal_best else float("nan"),
        "rho_B3_best_logit_only": logit_best["rho_B3"] if logit_best else float("nan"),
        "delta_rho_internal_minus_logit": delta_rho,
        "internal_beats_logit_by_0p1": float(beats_by_01),
        "auroc_safe_vs_abl_best_internal": auroc_int,
        "auroc_safe_vs_abl_best_logit_only": auroc_logit,
        "mean_kappa_keyword_vs_judge": float(np.nanmean(
            [v["kappa"] for v in judge_stats.values()])) if judge_stats else float("nan"),
        "prompt_boot_B3_mean_ci_width": float(np.mean(
            [v["ci95"][1] - v["ci95"][0] for v in b3_boot.values()])) if b3_boot else float("nan"),
    }

    verdict = (
        f"{'POSITIVE' if beats_by_01 else 'NEGATIVE'} branch: "
        f"best internal metric {internal_best['metric']} rho={internal_best['rho_B3']:.3f} "
        f"(95% CI {internal_best['ci95'][0]:.3f}..{internal_best['ci95'][1]:.3f}, n={internal_best['n']}) "
        f"vs best logit-only {logit_best['metric']} rho={logit_best['rho_B3']:.3f} "
        f"-> delta-rho={delta_rho:+.3f} (threshold +0.1). "
        f"Safe-vs-abliterated AUROC: internal {auroc_int:.3f} vs logit {auroc_logit:.3f}. "
        f"Keyword-vs-judge mean Cohen's kappa="
        f"{metrics_agg['mean_kappa_keyword_vs_judge']:.3f}. "
        f"Caveat: N={len(common)} models; per the plan, rank statistics on small N are "
        f"indicative and the per-family breakdown is reported alongside."
    )

    panel = json.loads((ROOT / "data" / "prompts.json").read_text())["sources"]
    # 80 gemini-2.5-flash-lite judge calls; per-call costs and cumulative
    # totals logged in logs/stage_judge_bg.log (~$0.0012 total, well under budget)
    judge_cost = 0.0012
    jlog = ROOT / "logs" / "stage_judge_bg.log"
    if jlog.exists():
        import re
        m = None
        for m in re.finditer(r"total=\$([0-9.]+)", jlog.read_text()):
            pass
        if m:
            judge_cost = float(m.group(1))

    out = {
        "metadata": {
            "evaluation_name": "refusal_baseline_vs_activation_metrics_qwen3",
            "artifact": "gen_art_evaluation_1",
            "models": [{"hf_id": t, "role": roles[t], "family": families[t],
                        "params_b": gt[t]["params_b"]} for t in common],
            "prompt_sets": panel,
            "metric_groups": GROUPS,
            "internal_vs_logit": {"best_internal": internal_best,
                                  "best_logit_only": logit_best,
                                  "delta_rho": delta_rho,
                                  "beats_by_0p1": beats_by_01},
            "per_family_best": fam_best,
            "sensitivity_excl_SafeRL": excl_safel,
            "correlations_full": rows,
            "judge_validation": judge_stats,
            "b3_prompt_bootstrap": b3_boot,
            "b3_rank_order_B3": sorted(common, key=lambda t: -gt[t]["B3"]),
            "n_boot": N_BOOT,
            "judge_model": judge.get("judge_model"),
            "llm_judge_cost_usd_ledger_total": judge_cost,
            "verdict": verdict,
            "caveats": [
                "B1/B2/B3 from 10+10 greedy prompts with keyword classifier is a proxy, "
                "not a full benchmark (no TrustLLM/AIR-Bench breadth) - iteration-1 scope.",
                "Spearman rho over a small model panel has wide bootstrap CIs; treat "
                "point rho as indicative and the AUROC separation as the robust signal.",
                "Activation metrics designed on the Qwen3-4B lineage; non-Qwen3 members "
                "are the declared held-out check (per-family breakdown reported).",
            ],
            "runtime_s": round(time.time() - t0, 1),
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {"dataset": "behavioral_ground_truth_per_model", "examples": ex_behavior},
            {"dataset": "internal_metrics_per_model", "examples": ex_metrics},
            {"dataset": "metric_vs_B3_correlations", "examples": ex_corr},
        ],
    }
    save_json(out, RESULTS_DIR / "eval_out.json")
    logger.info(f"wrote eval_out.json in {out['metadata']['runtime_s']}s")
    logger.info(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
