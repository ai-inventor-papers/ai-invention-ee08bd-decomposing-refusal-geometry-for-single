#!/usr/bin/env python3
"""Stage D: compute M1–M15 activation/logit metrics from cached npz extracts.

Metric definitions are FIXED here, before any correlation is computed (the
artifact plan's pre-registration requirement). Layer indexing: hidden_states
index 0 is the embedding output; model layers are 1..L-1; we report 1-based
layer ids. The mid-late window is layers round(L'/3) .. round(3L'/4) over the
L' = L-1 actual transformer layers, per the plan.

  M1  mean effective rank (Roy & Vetterli) of the harmful last-token
      activation matrix [32 x d] over the mid-late window
  M2  same on benign prompts
  M3  harmful/benign effective-rank ratio (M1/M2)
  M4  mean adjacent-layer cosine of the top right-singular vector (harmful)
  M5  same (benign)
  M6  benign-subtracted rotation coherence: M4 - M5
  M7  coupling product: mean over mid-late layers of z(effrank_traj) *
      z(coherence_traj), harmful prompts (both trajectories smoothed)
  M8  Pearson correlation across layers between the eff-rank trajectory and
      the local rotation angle trajectory (harmful, mid-late window)
  M9  cluster separation: per layer ||mean_harm - mean_ben|| / pooled sd,
      averaged over mid-late layers
  M10 residual norm ratio trajectory harm/ben per layer; value = mean over
      mid-late window; also M10b dip depth = max - min inside window
  M11 first-position refusal-token logit margin (harmful mean) [logit-only]
  M12 first-position output entropy (harmful mean) [logit-only]
  M13 mean cosine between top-1 left singular vectors of harmful vs benign
      matrices per layer (refusal-direction presence), mid-late mean
  M14 mean attention entropy (last token, head-mean) on harmful prompts,
      mid-late layers
  M15 per-layer logistic-probe AUROC harmful vs benign (5-fold CV,
      token-level labels = prompt class), max over layers and mid-late mean

Run:  .venv/bin/python stage_metrics.py
Output: results/metrics_M.json  {hf_id: {metric: value}}, plus layer curves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import ACTS_DIR, RESULTS_DIR, logger, save_json

OUT_PATH = RESULTS_DIR / "metrics_M.json"
CURVES_PATH = RESULTS_DIR / "metrics_M_layer_curves.json"


def eff_rank(s: np.ndarray) -> float:
    e = np.asarray(s, dtype=np.float64) ** 2
    tot = e.sum()
    if tot <= 0:
        return float("nan")
    p = e / tot
    p = p[p > 1e-30]
    if p.size <= 1:
        return 1.0
    return float(np.exp(-(p * np.log(p)).sum()))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def midlate_window(n_layers_model: int) -> range:
    """n_layers_model = number of actual transformer layers (len(H)-1)."""
    lo = max(1, round(n_layers_model / 3))
    hi = min(n_layers_model, round(3 * n_layers_model / 4))
    return range(lo, hi + 1)


def probe_auroc(H: np.ndarray, y: np.ndarray) -> float:
    """5-fold CV logistic probe AUROC on [n x d] with binary y."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < 4 or n_neg < 4:
        return float("nan")
    X = (H - H.mean(0)) / (H.std(0) + 1e-8)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    try:
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        return float(roc_auc_score(y, oof))
    except Exception as e:
        logger.warning(f"probe failed: {e}")
        return float("nan")


def compute_model(npz_path: Path, meta: dict) -> tuple[dict, dict]:
    z = np.load(npz_path)
    H_harm = z["H_harm"].astype(np.float64)   # [32, L+1, d]
    H_ben = z["H_ben"].astype(np.float64)
    E_harm = z["E_harm"].astype(np.float64)   # [32, L]
    n = meta["n_layers_H"]                    # H includes embedding at 0
    L = n - 1                                 # transformer layers count
    win = midlate_window(L)
    d_model = meta["d_model"]
    y = np.concatenate([np.ones(H_harm.shape[0]), np.zeros(H_ben.shape[0])])

    curves: dict[str, dict] = {
        "M1_effrank_harm": {}, "M2_effrank_ben": {},
        "M4_coh_harm": {}, "M5_coh_ben": {}, "M9_sep": {},
        "M10_normratio": {}, "M13_pc_cos": {}, "M15_probe": {},
        "M14_attent": {}}
    er_h, er_b, coh_h, coh_b, sep_l, nr_l, pc_l, pr_l, at_l = \
        [], [], [], [], [], [], [], [], []

    prev_vh_h = prev_vh_b = None
    for l in range(1, n):  # transformer layers only (1-based)
        A = H_harm[:, l, :]
        B = H_ben[:, l, :]
        Ua, sa, Vta = np.linalg.svd(A, full_matrices=False)
        Ub, sb, Vtb = np.linalg.svd(B, full_matrices=False)
        er_h.append(eff_rank(sa))
        er_b.append(eff_rank(sb))
        coh_h.append(float(np.nan))
        coh_b.append(float(np.nan))
        if prev_vh_h is not None:
            coh_h[-1] = cosine(prev_vh_h, Vta[0])
            coh_b[-1] = cosine(prev_vh_b, Vtb[0])
        prev_vh_h, prev_vh_b = Vta[0], Vtb[0]
        mu_a, mu_b = A.mean(0), B.mean(0)
        sd = np.sqrt(A.var(0).mean() + B.var(0).mean()) + 1e-8
        sep_l.append(float(np.linalg.norm(mu_a - mu_b) / sd))
        na = np.linalg.norm(A, axis=1).mean()
        nb = np.linalg.norm(B, axis=1).mean()
        nr_l.append(na / (nb + 1e-8))
        pc_l.append(cosine(Ua[:, 0], Ub[:, 0]))
        pr_l.append(probe_auroc(np.concatenate([A, B], 0), y))
        at_l.append(float(E_harm[:, l - 1].mean()))

    def w(vals, win=win, offset=1):
        return [vals[l - offset] for l in win]

    z_er = np.array(w(er_h))
    z_co = np.array([c for c in w(coh_h) if not np.isnan(c)])
    er_w = np.array(w(er_h))
    co_w = np.array(w(coh_h))
    # M7: coupling product of z-scored smoothed trajectories (harmful)
    def zsc(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        sd = v.std()
        return (v - v.mean()) / sd if sd > 1e-9 else v * 0.0
    # smooth with a 3-layer moving average, then z-score both
    def smooth(v: np.ndarray) -> np.ndarray:
        k = np.ones(3) / 3
        return np.convolve(v, k, mode="same")
    a_t = zsc(smooth(er_w))
    c_t = zsc(np.nan_to_num(smooth(co_w), nan=0.0))
    m7 = float(np.nanmean(a_t * c_t))
    # M8: Pearson correlation of raw trajectories
    ok = ~np.isnan(co_w)
    m8 = float(np.corrcoef(er_w[ok], co_w[ok])[0, 1]) if ok.sum() > 3 else float("nan")

    M = {
        "M1_effrank_harm_midlate": float(np.mean(w(er_h))),
        "M2_effrank_ben_midlate": float(np.mean(w(er_b))),
        "M3_effrank_ratio_hb": float(np.mean(w(er_h)) / (np.mean(w(er_b)) + 1e-9)),
        "M4_rot_coh_harm_midlate": float(np.nanmean(w(coh_h))),
        "M5_rot_coh_ben_midlate": float(np.nanmean(w(coh_b))),
        "M6_rot_coh_diff": float(np.nanmean(w(coh_h)) - np.nanmean(w(coh_b))),
        "M7_coupling_product": m7,
        "M8_coupling_corr": m8,
        "M9_cluster_sep_midlate": float(np.mean(w(sep_l))),
        "M9b_cluster_sep_max": float(np.max(sep_l)),
        "M10_norm_ratio_midlate": float(np.mean(w(nr_l))),
        "M10b_norm_dip_depth": float(np.max(w(nr_l)) - np.min(w(nr_l))),
        "M11_logit_margin_harm": float(z["M11_harm"].mean()),
        "M11b_logit_margin_hb": float(z["M11_harm"].mean() - z["M11_ben"].mean()),
        "M12_entropy_harm": float(z["M12_harm"].mean()),
        "M13_pc_cosine_midlate": float(np.mean(w(pc_l))),
        "M14_attent_entropy_midlate": float(np.mean(w(at_l))),
        "M15_probe_auroc_max": float(np.nanmax(pr_l)),
        "M15b_probe_auroc_midlate": float(np.nanmean(w(pr_l))),
    }
    curves = {
        "M1_effrank_harm": {str(l): er_h[l - 1] for l in range(1, n)},
        "M2_effrank_ben": {str(l): er_b[l - 1] for l in range(1, n)},
        "M4_rot_coh_harm": {str(l): coh_h[l - 1] for l in range(2, n)},
        "M9_sep": {str(l): sep_l[l - 1] for l in range(1, n)},
        "M10_normratio": {str(l): nr_l[l - 1] for l in range(1, n)},
        "M13_pc_cos": {str(l): pc_l[l - 1] for l in range(1, n)},
        "M15_probe": {str(l): pr_l[l - 1] for l in range(1, n)},
        "M14_attent": {str(l): at_l[l - 1] for l in range(1, n)},
    }
    del z, H_harm, H_ben
    return M, curves


@logger.catch(reraise=True)
def main() -> None:
    metas = sorted(ACTS_DIR.glob("acts_*_meta.json"))
    all_M, all_curves = {}, {}
    for mp in metas:
        meta = json.loads(mp.read_text())
        tag = meta["hf_id"]
        npz = ACTS_DIR / mp.name.replace("_meta.json", ".npz")
        if not npz.exists():
            continue
        try:
            M, curves = compute_model(npz, meta)
        except Exception as e:
            logger.error(f"metrics failed for {tag}: {type(e).__name__}: {e}")
            continue
        all_M[tag] = M
        all_curves[tag] = curves
        logger.info(f"metrics done: {tag} (M9b={M['M9b_cluster_sep_max']:.3f}, "
                    f"M15max={M['M15_probe_auroc_max']:.3f})")
    save_json({"metrics_per_model": all_M}, OUT_PATH)
    save_json({"layer_curves": all_curves}, CURVES_PATH)
    logger.info(f"wrote {OUT_PATH} for {len(all_M)} models")


if __name__ == "__main__":
    main()
