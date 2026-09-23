# gen_art_evaluation_2 — Judge-validated 14-model re-analysis of the coherence safety signal

Fixes every methodological issue from iteration-1's evaluation: judge validation expanded
from 4 to 14 models (~840 `google/gemini-2.5-flash-lite` calls, $0.014 actual cost), the
two dropped Qwen3-4B models restored to the behavioral panel, all correlations recomputed
against judge-based net refusal with exact-seed permutation audits, BH correction, honest
n=7 published anchors, and MMLU-Redux capability-confound checks.

## Layout

- `eval.py` — main evaluation driver (run this); writes `results/eval_out.json`,
  `results/eval_out_audit.json`, and the workspace-root copy `eval_out.json`
  (exp_eval_sol_out schema).
- `audit.py` — INDEPENDENT audit through a different code path (hand-rolled Spearman,
  pandas, sklearn kappa); writes `results/audit_independent.json`. Includes a permutation
  placebo that must FAIL on shuffled labels. `ALL CHECKS PASSED`.
- `gen_classify.py` — restores Qwen/Qwen3-4B and mlabonne/Qwen3-4B-abliterated to the
  behavioral panel: greedy generation on the experiment-1 20 held-out prompts (10 harmful
  / 10 benign-dual-use) -> `results/behavior_*.json`.
- `stage_judge2.py` — async 3-seed judge pass (seeds 0/1/2, majority vote) over all 280
  responses x 3 seeds; writes `results/judge_labels_14.json`; cost ledger in
  `logs/judge_cost_ledger.json` (hard stop $5).
- `stage_metrics14.py` — recomputes the 19 M-metrics for the 2 restored models from the
  experiment-1 cached activations (`/ai-inventor/.../gen_art_experiment_1/results/
  activations/{Qwen3-4B,Qwen3-4B-abliterated}/hidden.npy`) reusing `stage_metrics.compute_model`
  verbatim; writes `results/metrics_M_14.json`.
- `common.py`, `models.py`, `stage_behavior.py`, `stage_metrics.py`, `stage_activations.py`,
  `stage_analysis.py` — iteration-1 infrastructure, reused unchanged.
- `results/` — all outputs (kept); `acts_cache/` — per-model activation npz cache.
- `full_eval_out.json`, `mini_eval_out.json`, `preview_eval_out.json` — schema-formatted
  variants of `eval_out.json`.

## Run

```bash
uv venv .venv --python=3.12 && uv sync
.venv/bin/python eval.py      # idempotent; reuses all stage outputs
.venv/bin/python audit.py     # independent re-derivation + placebo audit
```

## Headline results (n=14 judge-validated models)

- Best internal metric: `M10_norm_ratio_midlate` rho=0.693 [bootstrap CI 0.140..0.959,
  p_perm=0.0076, p_BH=0.048] vs best logit-only `M11b_logit_margin_hb` rho=0.380
  -> delta-rho=+0.313 (threshold +0.1 passed).
- Iteration-1 headline `M4_rot_coh_harm_midlate` REVERSES on judge labels: rho=-0.751
  (n=14) — the earlier rho=0.738 was computed against keyword labels with kappa 0.084.
- Honest n=7 published-refusal anchor: rho=-0.811; sd-ratio 0.975 => NOT range
  restriction, the reversal is a genuine orthogonality/negativity finding.
- MMLU-Redux capability confound: rho=-0.643 (n=7); tradeoff index +0.108 — the
  internal safety signal is not a capability proxy.
- Keyword-vs-judge mean Cohen's kappa=0.198 (non-thinking-truncated models 0.274);
  thinking-truncated models are flagged in `metadata.judge_validation`.
- Non-Qwen subpanel and within-Qwen3 subpanel rhos are reported in
  `metadata.sensitivity_subpanels` (cross-family generalization stays underpowered).

## Restoring removed files

- `.venv/` is regenerable: `uv sync` (pyproject.toml pins every version).

- `__pycache__/` regenerates automatically on the next run.

## Manifest

See `.aii/manifest.yaml` for keep/delete decisions on heavy paths.
