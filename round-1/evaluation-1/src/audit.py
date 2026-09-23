#!/usr/bin/env python3
"""Independent audit of headline numbers in eval_out.json (TODO 4).

Recomputes, via a DIFFERENT code path than eval.py/stage_analysis.py:
  * B1/B2/B3 per model directly from behavior_*.json response lists
  * Spearman rho (best internal vs best logit, incl. and excl. SafeRL)
    by hand-rolled average-rank + Pearson, no scipy.spearman
  * shuffle/placebo control: headline metrics vs permuted B3 (200 shuffles)
Reads only results/*.json raw fields; never imports eval.py.
"""
import json, glob
import numpy as np


def rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a)
    r = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def my_spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


mets = json.load(open("results/metrics_M.json"))["metrics_per_model"]
beh = {}
for f in glob.glob("results/behavior_*.json"):
    r = json.load(open(f))
    kw_h = [int(bool(r2["kw_refusal"] or r2.get("keyword_refusal")))
            for r2 in r["responses"]["harmful"]]
    kw_b = [int(bool(r2["kw_refusal"] or r2.get("keyword_refusal")))
            for r2 in r["responses"]["benign"]]
    assert abs(sum(kw_h) / len(kw_h) - r["B1"]) < 1e-9, r["hf_id"]
    assert abs(sum(kw_b) / len(kw_b) - r["B2"]) < 1e-9, r["hf_id"]
    beh[r["hf_id"]] = dict(B1=sum(kw_h) / len(kw_h), B2=sum(kw_b) / len(kw_b),
                           B3=sum(kw_h) / len(kw_h) - sum(kw_b) / len(kw_b))
print(f"B1/B2/B3 independently recomputed for {len(beh)} models: all match")

common = sorted(set(mets) & set(beh))


def headline(exclude_safel=False):
    mods = [m for m in common if not (exclude_safel and "SafeRL" in m)]
    # round to 6dp: B3 floats carry 1e-16 noise that flips tie detection in
    # rank comparison (0.30000000000000004 vs 0.3); rounding restores the
    # canonical tie structure scipy sees
    y = np.round(np.array([beh[m]["B3"] for m in mods]), 6)
    res = {}
    for mn in sorted(next(iter(mets.values())).keys()):
        v = np.array([mets[m].get(mn, np.nan) for m in mods], float)
        ok = np.isfinite(v)
        if ok.sum() < 6 or np.std(v[ok]) == 0:
            continue
        res[mn] = my_spearman(v[ok], y[ok])
    internal = max((k for k in res if k.split("_")[0] not in ("M11", "M12")), key=res.get)
    logit = max((k for k in res if k.split("_")[0] in ("M11", "M12")), key=res.get)
    return internal, res[internal], logit, res[logit], res[internal] - res[logit]


i1, r1, l1, rl1, d1 = headline(False)
i2, r2, l2, rl2, d2 = headline(True)
print(f"ALL:     best internal {i1} rho={r1:.4f} | best logit {l1} rho={rl1:.4f} | delta={d1:+.4f}")
print(f"-SafeRL: best internal {i2} rho={r2:.4f} | best logit {l2} rho={rl2:.4f} | delta={d2:+.4f}")

rng = np.random.default_rng(0)
mods = [m for m in common if "SafeRL" not in m]
y0 = np.array([beh[m]["B3"] for m in mods])
for mn in (i1, l1):
    v = np.array([mets[m][mn] for m in mods], float)
    null = [my_spearman(v, y0[rng.permutation(len(y0))]) for _ in range(200)]
    obs = my_spearman(v, y0)
    p = float(np.mean(np.abs(null) >= abs(obs)))
    print(f"placebo {mn}: obs={obs:+.3f} null |rho| p95={np.percentile(np.abs(null), 95):.3f} perm-p={p:.3f}")
