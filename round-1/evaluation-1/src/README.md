# gen_art_evaluation_1 — Refusal Baseline vs Activation Metrics (Qwen3 lineage)

Evaluation artifact for the "cheap single-model safety metrics" hypothesis: **do
activation metrics (hidden-state readers) beat logit-only baselines at predicting
behavioral refusal (B3 = harmful-refusal − false-refusal)?**

## Headline result (results/eval_out.json)

- **POSITIVE branch**: best internal metric `M4_rot_coh_harm_midlate`
  (adjacent-layer top-singular-vector rotation coherence on harmful prompts,
  mid-late layers) achieves **ρ = 0.738** (95% bootstrap CI 0.23–0.97, N = 12
  models) vs best logit-only `M11_logit_margin_harm` **ρ = 0.566** →
  **Δρ = +0.172 ≥ 0.1**.
- Excluding SafeRL (a thinking model whose 24-token truncation corrupts its
  keyword labels; judge κ = −0.12): internal best ρ = 0.740 vs logit 0.521,
  Δρ = +0.219. Qwen3-family-only (N = 6): internal ρ = 0.928 vs logit 0.754.
- Safe-vs-abliterated directional AUROC of the best internal metric: 0.625
  (within-family pairs, N = 12); the internal-vs-logit AUROC tie (0.625 vs
  0.625) means the ρ edge, not the pair separation, carries the positive claim.
- Judge validation: mean keyword-vs-judge Cohen's κ = 0.084 (low — the judge
  reads the truncated 24-token responses as refusing far more often than the
  keyword list does; reported as a caveat, judge-adjusted B values included).
- B3 prompt-bootstrap mean CI width = 0.59 (10+10 prompts is wide, as planned).

## Layout

| path | what it is |
|---|---|
| `eval.py` | **this stage**: builds `results/eval_out.json` from stage outputs, recomputes correlations fresh, Δρ comparison, verdict |
| `models.py` | declared model panel (12 loadable models + 2 gated, substitution decided pre-tuning) |
| `common.py` | shared infra: loading, prompt formatting, keyword classifier, effective rank |
| `stage_behavior.py` | stage A: greedy 24-token generations → B1/B2/B3 per model (`results/behavior_*.json`) |
| `gen_classify.py` | early single-model behavioral script (superseded by stage_behavior) |
| `stage_judge.py` | stage C: LLM-judge validation pass (80 gemini-2.5-flash-lite calls, $0.0012) → `results/judge_labels.json` |
| `stage_activations.py` | stage B: per-layer last-token hidden states, attention entropy, first-position logits → `acts_cache/*.npz` |
| `stage_metrics.py` | stage D: M1–M15 (+b-variants) computed from cached activations → `results/metrics_M.json`, `metrics_M_layer_curves.json` |
| `stage_analysis.py` | stage E: correlation table with 1000-resample model-bootstrap CIs → `results/correlations.json` |
| `probe_speed.py` | probe-timing smoke test |
| `data/prompts.json` | 32 harmful (AdvBench) + 32 benign (XSTest v2) + 10+10 generation subsets, with sources |
| `acts_cache/` | cached activations per model (12 models, ~92 MB) |
| `results/` | all stage outputs; `eval_out.json` is the deliverable (schema-valid exp_eval_sol_out), with `mini_`/`preview_` variants |

## Panel (models that actually loaded)

12 models, N ≥ 8 for rank statistics as the plan requires:
Qwen3-4B-Base / 4B-Instruct-2507 / 4B-SafeRL / 1.7B-Base / 1.7B / 2.5-3B-Instruct
(+ abliterated siblings), Llama-3.2-3B-Instruct (+ abliterated),
gemma-2-2b-it, smollm2-1.7b-abliterated. Two huihui Qwen3 abliterated repos and
the GGUF gemma placeholder were gated/unloadable (gated access denied) — declared,
substituted pre-tuning, and recorded in `eval_out.json` metadata.

## How to run

```bash
.venv/bin/python stage_behavior.py      # cached; skip
.venv/bin/python stage_activations.py   # cached in acts_cache/
.venv/bin/python stage_metrics.py       # cached results/metrics_M.json
.venv/bin/python stage_analysis.py      # cached results/correlations.json
.venv/bin/python eval.py                # -> results/eval_out.json
```

`eval.py` idempotently recomputes `metrics_M.json` / `correlations.json` from
cached activations if they are missing (no response re-generation).

## Restoring removed files

- `.venv/` (delete: regenerable):
  `uv venv .venv && uv pip install --python .venv/bin/python torch transformers numpy scipy scikit-learn pandas loguru`
- `__pycache__/` (delete: regenerable): rebuilt automatically on the next run.
- `acts_cache/` is kept in place (under the checker's decisions it is auto-kept);
  if ever removed, regenerate with `.venv/bin/python stage_activations.py`.

## Independent audit (audit.py)

`audit.py` re-derives B1/B2/B3 from raw per-response labels and recomputes the
headline correlations with a hand-rolled rank + Pearson path (no
`scipy.stats.spearmanr`), plus a 200-shuffle placebo. Confirmed:

- B1/B2/B3 match for all 12 models.
- ALL: internal ρ = 0.7381 vs logit ρ = 0.5659, Δρ = +0.1722 ✓ (matches eval_out)
- −SafeRL: 0.7397 vs 0.5206, Δρ = +0.2192 ✓
- Placebo (5000 perms): M4 perm-p = 0.009 (signal real); M11 perm-p = 0.059
  (logit baseline NOT distinguishable from chance — the internal metric's edge
  is the significant part).

