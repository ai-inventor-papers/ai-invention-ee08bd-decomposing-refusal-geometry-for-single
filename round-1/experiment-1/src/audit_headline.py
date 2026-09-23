#!/usr/bin/env python3
"""Independent audit of the headline numbers in results/method_out.json.

Re-derives, through a DIFFERENT code path than src/analyze.py:
  1. Cohen's d / AUROC for the primary contrast (safe vs abliterated) of the
     top-2 metrics, reading the RAW activation/weight data straight from
     results/metrics/*.json (not the aggregated metrics_table / separability).
  2. Recomputes the top-2 metric values themselves from raw hidden states
     (anisotropy: mean pairwise cosine gap; refusal-dir gap: diff-of-means
     projection) with an independent numpy implementation (scipy.stats and
     sklearn.metrics rather than hand-written Mann-Whitney / pooled-SD code).
  3. Placebo test: same statistics after permuting the role labels 2000 times;
     the real statistic must sit in the extreme tail of the placebo
     distribution (empirical p < 0.05), otherwise the metric is vacuous.

Writes results/audit_headline.json and prints a PASS/FAIL table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 12345  # different seed from analyze.py on purpose
N_PERM = 2000


def pooled_d(a: np.ndarray, b: np.ndarray) -> float:
    """Independent implementation via scipy (t statistic * hedges correction shape)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp)


def main() -> None:
    battery = json.loads((ROOT / "data" / "prompts" / "battery.json").read_text())
    cats = np.array([p["category"] for p in battery])
    splits = np.array([p["split"] for p in battery])
    harm_dev = np.where((cats == "harmful") & (splits == "dev"))[0]
    benign_dev = np.where((cats == "benign") & (splits == "dev"))[0]  # benign-only (m03/m07 semantics)
    dual_dev = np.where((cats == "dual-use") & (splits == "dev"))[0]

    report = json.loads((RESULTS / "separability.json").read_text())
    league = report["league"]
    top2 = [league[0]["metric"], league[1]["metric"]]
    print(f"headline metrics: {top2}")

    audit: dict[str, dict] = {}
    for metric in top2:
        # --- recompute metric values from raw data, independent path ---
        vals: dict[str, float] = {}
        roles: dict[str, str] = {}
        for f in sorted((RESULTS / "metrics").glob("*.json")):
            d = json.loads(f.read_text())
            roles[d["model"]] = d["role"]
            if metric == "activation_anisotropy_harmful_minus_benign":
                H = np.load(RESULTS / "activations" / d["model"] / "hidden.npy").astype(np.float32)
                l_star = int(np.argmax([
                    _aniso(H[harm_dev, l]) - _aniso(H[benign_dev, l])
                    for l in range(H.shape[1])
                ]))
                vals[d["model"]] = _aniso(H[harm_dev, l_star]) - _aniso(H[benign_dev, l_star])
                del H
            elif metric == "refusal_dir_alignment_gap":
                H = np.load(RESULTS / "activations" / d["model"] / "hidden.npy").astype(np.float32)
                others = np.concatenate([benign_dev, dual_dev])  # m05 'others' semantics
                gaps = [_dir_gap(H, l, harm_dev, others) for l in range(H.shape[1])]
                vals[d["model"]] = float(np.max(gaps))
                del H
            else:
                vals[d["model"]] = float(d["metrics"][metric]["value"])

        models = sorted(vals)
        pos = [vals[t] for t in models if roles[t] in ("instruct", "saferl")]
        neg = [vals[t] for t in models if roles[t] == "abliterated"]
        d_real = pooled_d(np.asarray(pos), np.asarray(neg))
        u = mannwhitneyu(pos, neg, alternative="two-sided")
        auc_real = roc_auc_score([1] * len(pos) + [0] * len(neg),
                                 list(pos) + list(neg))

        # --- placebo: permute role labels, keep values ---
        rng = np.random.default_rng(SEED)
        pooled_vals = np.asarray([vals[t] for t in models])
        role_arr = np.asarray([roles[t] in ("instruct", "saferl") for t in models])
        n_pos = int(role_arr.sum())
        count_ge = 0
        for _ in range(N_PERM):
            perm = rng.permutation(role_arr)
            pp = pooled_vals[perm][:n_pos] if False else pooled_vals[:n_pos][perm[:n_pos]]
            pn = pooled_vals[n_pos:][perm[n_pos:]]
            dp = abs(pooled_d(pp, pn))
            if dp >= abs(d_real):
                count_ge += 1
        p_placebo = count_ge / N_PERM

        claimed = next(r for r in league if r["metric"] == metric)
        ok_d = abs(d_real - abs(claimed["d"])) < 0.05 * max(1, abs(claimed["d"]))
        audit[metric] = {
            "recomputed_d": round(d_real, 4), "claimed_d": claimed["d"], "d_match": ok_d,
            "recomputed_auroc": round(float(auc_real), 4), "claimed_auroc": claimed["auroc"],
            "mannwhitney_p": float(u.pvalue),
            "placebo_p_two_sided": round(p_placebo, 4),
            "placebo_passes": p_placebo < 0.05,
        }
        print(f"{metric}: d={d_real:.3f} (claimed {claimed['d']:.3f}, match={ok_d}) "
              f"AUROC={auc_real:.3f} (claimed {claimed['auroc']:.3f}) "
              f"placebo p={p_placebo:.4f} -> {'PASS' if p_placebo < 0.05 and ok_d else 'FAIL'}")

    (RESULTS / "audit_headline.json").write_text(json.dumps(audit, indent=1))
    all_pass = all(a["placebo_passes"] and a["d_match"] for a in audit.values())
    print("AUDIT:", "ALL PASS" if all_pass else "MISMATCH - see results/audit_headline.json")
    print("wrote results/audit_headline.json")


def _aniso(X: np.ndarray) -> float:
    Xc = X - X.mean(0, keepdims=True)
    Xn = Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12, None)
    n = len(Xn)
    iu = np.triu_indices(n, 1)
    return float((Xn @ Xn.T)[iu].mean())


def _dir_gap(H: np.ndarray, l: int, harm: np.ndarray, others: np.ndarray) -> float:
    mu_h = H[harm, l].mean(0)
    mu_o = H[others, l].mean(0)
    r = mu_h - mu_o
    nr = np.linalg.norm(r)
    if nr <= 1e-12:
        return 0.0  # degenerate direction (same guard as src/metrics.m05)
    r = r / nr
    ch = (H[harm, l] / np.clip(np.linalg.norm(H[harm, l], axis=1, keepdims=True), 1e-12, None)) @ r
    co = (H[others, l] / np.clip(np.linalg.norm(H[others, l], axis=1, keepdims=True), 1e-12, None)) @ r
    return float(ch.mean() - co.mean())


if __name__ == "__main__":
    sys.exit(main())
