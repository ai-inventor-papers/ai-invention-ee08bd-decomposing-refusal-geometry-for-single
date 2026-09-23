# Qwen3 safety metrics via activation analysis — 10-model panel

**Question.** For a random model found on Hugging Face — no reference model, no base
checkpoint, nothing to diff against — can a *cheap internal readout* (weights or
activations only, seconds-to-minutes per model) tell you how safe it is, without
running a full benchmark?

**Answer found here.** Yes, partially. Two activation-geometry metrics computed from a
single forward pass per prompt separate safety-tuned models from their abliterated
siblings **perfectly** (model-level AUROC = 1.000, Cohen's d = 4.95 / 3.82) across a
10-model panel spanning three Qwen lineages — and the effect survives a placebo test
(permuting role labels 2000 times, p ≤ 0.023). Weight-spectral metrics (no forward
pass at all) do **not** separate the roles: abliteration leaves o_proj spectra
essentially untouched. Supervised probes saturate at AUC ≈ 1.0 *inside every model*
including base and abliterated ones — they separate prompts, not models — which is the
knowledge–action gap showing up at the level of the metric battery itself.

## Panel (10 models, 3 lineages, all loaded bf16, one at a time)

| model | role | lineage |
|---|---|---|
| Qwen/Qwen3-4B-Base | base | qwen3-4b |
| Qwen/Qwen3-4B | instruct | qwen3-4b |
| Qwen/Qwen3-4B-SafeRL | saferl (official safety RL) | qwen3-4b |
| mlabonne/Qwen3-4B-abliterated | abliterated | qwen3-4b |
| Qwen/Qwen3-4B-Instruct-2507 | instruct | qwen3-4b-2507 |
| huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated | abliterated | qwen3-4b-2507 |
| Qwen/Qwen3-1.7B-Base | base | qwen3-1.7b |
| Qwen/Qwen3-1.7B | instruct | qwen3-1.7b |
| Qwen/Qwen2.5-3B-Instruct | instruct | qwen2.5-3b |
| huihui-ai/Qwen2.5-3B-Instruct-abliterated-SFT | abliterated | qwen2.5-3b |

Substitutions vs the artifact plan (recorded in `method_out.json`): the planned
huihui-ai/Qwen3-4B-abliterated had no cached weights; mlabonne's abliteration of the
same Qwen3-4B (FailSpy recipe) was substituted. Qwen2.5-3B-Instruct-abliterated-SFT is
the abliterated sibling of Qwen2.5-3B-Instruct (the registry's `-abliterated` short
name maps to that repo).

The compute this ran on is a 2-CPU / 14 GB container with **no GPU**, so the plan's
16 GB VRAM single-GPU profile was adapted to CPU bf16; every model still loads through
transformers with hidden states and weights readable (no GGUF/llama.cpp).

## Battery

60 prompts = 30 harmful (AdvBench, Or-Bench toxic, Or-Bench hard) + 15 benign
(XSTest v2 safe types, Anthropic HH harmless turns, hand-written) + 15 hand-written
dual-use. Stratified 40 dev / 20 held-out, seed 0. The user-supplied
`qwen3_explication.md` was absent from `user_uploads/` (verified), so the dual-use
category uses the plan's hand-written fallback.

## Metrics (18 shipped; invariant: ≥3 read hidden states/weights, ≤2 logit-only)

- **activation geometry (10)** — effective rank, participation ratio, harmful-minus-benign
  rank drop, mean cosine to the diff-of-means refusal direction, refusal-direction
  alignment gap (Arditi-style baseline, readout form), norm trajectory, anisotropy gap,
  argmax-layer locator, dispersion ratio, top-1 spectral energy
- **weight spectral (3)** — o_proj top-1 SV energy share, spectral gap, row-norm Gini
- **attention (1)** — attention-entropy shift harmful vs benign
- **logit-only baselines (2)** — first-token margin, decision-point entropy
- **supervised probe baselines (2)** — 5-fold CV linear probe AUC, dual-use probe AUC

## Headline results (`results/separability_report.md`)

1. `activation_anisotropy_harmful_minus_benign` — d = 4.951 [3.70, 13.96], AUROC 1.000
2. `refusal_dir_alignment_gap` — d = 3.822 [2.55, 13.59], AUROC 1.000
3. `rank_drop_harmful_minus_benign` — d = −2.825, AUROC 0.000 (sign flipped: abliterated
   models have the *larger* harmful-vs-benign rank gap)
4. `layer_argmax_separation` — d = −2.159
5. `participation_ratio_per_layer` — d = −1.490

Held-out confirmation (pre-registered |d|>1.0 or AUROC>0.9, recomputed on held-out
prompts only): refusal_dir_alignment_gap, rank_drop, participation_ratio **confirmed**;
anisotropy and layer_argmax not confirmed on the held-out split (small-n class noise).

Zero-shot probe transfer: probes transfer ~1.00 AUROC *within* an architecture family
(all Qwen3-4B variants, both safe and abliterated) and collapse cross-family
(Qwen2.5↔Qwen3 ≈ 0.4-0.6) — the safety-relevant direction is family-relative.

## Reproduce

```bash
uv venv .venv --python=3.12 && source .venv/bin/activate
uv sync                                  # pinned deps from pyproject.toml
uv run python src/build_prompts.py       # rebuild battery (needs HF datasets; cached copy in data/prompts/)
uv run python src/run_model.py           # ~4 min/model on 2 CPUs; writes results/activations + results/metrics
uv run python src/analyze.py             # aggregate -> results/method_out.json etc.
uv run python audit_headline.py          # independent re-derivation + placebo test
```

`method.py` at the repo root is the driver (check / extract / analyze / deliver
phases — run `uv run method.py` to re-run the whole pipeline, skipping finished
work); the implementation lives in the `src/` package.

## Layout

| path | contents |
|---|---|
| `method.py` | pipeline driver (check/extract/analyze/deliver) |
| `src/registry.py` | the 10-model panel, roles, lineages, system prompt |
| `src/build_prompts.py` | 60-prompt battery construction + stratified split |
| `src/extract.py` | bf16 loader, capture (all-layer last-token residual, attention entropy, decision logits), greedy generation |
| `src/metrics.py` | the 18-metric battery (invariant-compliant) |
| `src/run_model.py` | per-model driver (background-safe, RSS watchdog) |
| `src/analyze.py` | aggregation, contrasts, bootstrap CIs, rank correlation, held-out confirmation, zero-shot transfer, figures |
| `src/hw.py` | cgroup-aware hardware limits (aii-use-hardware) |
| `audit_headline.py` | independent audit: re-derives headline d/AUROC via scipy/sklearn path, placebo permutation test |
| `results/method_out.json` | **primary artifact** (exp_gen_sol_out-schema; full/mini/preview siblings) |
| `results/separability_report.md`, `results/separability.json`, `results/metrics_table_9m.json`, `results/metric_spec.json` | analysis outputs |
| `results/activations/<model>/hidden.npy` | [60, L+1, d] fp16 residual streams per model — keep, lets later steps run wider metric searches without re-extraction |
| `results/activations/<model>/logits.npy`, `attn_ent.npy` | decision-point logits [60, V], per-layer attention entropy |
| `results/metrics/<model>.json` | per-model metric battery |
| `figures/fig1_panel_league.*`, `fig2_family_best.*` | league + best-per-family figures |
| `logs/` | run logs (panel extraction, analysis, audit) |

## Restoring removed files

The only `delete` entry in `.aii/manifest.yaml` is `.venv/` (regenerable):

```bash
uv sync   # pinned deps in pyproject.toml
```

Model weights are NOT in this repo (14–16 GB each). Restore with:

```bash
huggingface-cli download Qwen/Qwen3-4B-Base
huggingface-cli download Qwen/Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B-SafeRL
huggingface-cli download mlabonne/Qwen3-4B-abliterated
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
huggingface-cli download huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated
huggingface-cli download Qwen/Qwen3-1.7B-Base
huggingface-cli download Qwen/Qwen3-1.7B
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
huggingface-cli download huihui-ai/Qwen2.5-3B-Instruct-abliterated-SFT
```

(or `bash restore_models.sh`). Then `uv run python src/run_model.py` rebuilds every
tensor in `results/activations/` and `results/metrics/` from scratch in ~40 min on 2 CPUs.

## Audit

`results/audit_headline.json` — the two headline effect sizes were re-derived through a
different code path (scipy/sklearn instead of the hand-written estimators, different
seed) from the raw tensors: d matches the shipped values exactly (4.951, 3.822),
AUROC = 1.000 both, and a 2000-fold role-permutation placebo puts the real statistics in
the extreme tail (p = 0.023, 0.000).
