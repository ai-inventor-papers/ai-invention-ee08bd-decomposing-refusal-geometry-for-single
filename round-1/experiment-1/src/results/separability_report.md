# Separability report - single-model safety metrics, 10-model panel

Panel roles: Qwen2.5-3B-Instruct (instruct/qwen2.5-3b), Qwen2.5-3B-abliterated (abliterated/qwen2.5-3b), Qwen3-1.7B (instruct/qwen3-1.7b), Qwen3-1.7B-Base (base/qwen3-1.7b), Qwen3-4B (instruct/qwen3-4b), Qwen3-4B-Base (base/qwen3-4b), Qwen3-4B-Instruct-2507 (instruct/qwen3-4b-2507), Qwen3-4B-Instruct-2507-abliterated (abliterated/qwen3-4b-2507), Qwen3-4B-SafeRL (saferl/qwen3-4b), Qwen3-4B-abliterated (abliterated/qwen3-4b)

## Primary contrast: safety-tuned (instruct/saferl) vs abliterated, pooled

| rank | metric | family | d | d 95% CI | AUROC | rho vs refusal anchor | p (BH) |
|---|---|---|---|---|---|---|---|
| 1 | `activation_anisotropy_harmful_minus_benign` | activation_geometry | 4.951 | [3.696, 13.963] | 1.000 | 0.603148 | 0.259596 |
| 2 | `refusal_dir_alignment_gap` | activation_geometry | 3.822 | [2.545, 13.593] | 1.000 | 0.781631 | 0.097544 |
| 3 | `rank_drop_harmful_minus_benign` | activation_geometry | -2.825 | [-6.486, -2.297] | 0.000 | -0.713931 | 0.108757 |
| 4 | `layer_argmax_separation` | activation_geometry | -2.159 | [-57.062, -1.056] | 0.033 | -0.751567 | 0.097544 |
| 5 | `participation_ratio_per_layer` | activation_geometry | -1.490 | [-5.981, -0.369] | 0.133 | -0.369274 | 0.45383 |
| 6 | `eff_rank_per_layer` | activation_geometry | -1.395 | [-5.496, -0.247] | 0.133 | -0.436975 | 0.45383 |
| 7 | `first_token_margin_harmful_vs_benign` | logit_only_baseline | 1.167 | [0.265, 5.508] | 0.800 | 0.535448 | 0.354234 |
| 8 | `logit_entropy_at_refusal_decision` | logit_only_baseline | -1.101 | [-6.690, 0.314] | 0.200 | -0.504675 | 0.364904 |
| 9 | `attention_entropy_shift_harmful_vs_benign` | attention | 0.716 | [-0.923, 3.596] | 0.667 | 0.313883 | 0.45383 |
| 10 | `spectral_gap_attn_out` | weight_spectral | 0.658 | [-0.755, 2.157] | 0.800 | 0.326192 | 0.45383 |
| 11 | `cross_prompt_dispersion_ratio` | activation_geometry | 0.581 | [-2.732, 4.977] | 0.667 | 0.332347 | 0.45383 |
| 12 | `mean_cos_to_refusal_dir` | activation_geometry | 0.509 | [-2.139, 5.531] | 0.667 | 0.387738 | 0.45383 |
| 13 | `spectral_energy_top1` | activation_geometry | -0.424 | [-2.880, 0.910] | 0.400 | -0.301574 | 0.45383 |
| 14 | `weight_row_norm_gini_o_proj` | weight_spectral | -0.066 | [-2.043, 1.752] | 0.333 | -0.375429 | 0.45383 |
| 15 | `top1_sv_energy_ratio_o_proj` | weight_spectral | 0.057 | [-1.986, 2.200] | 0.733 | 0.166174 | 0.646368 |
| 16 | `norm_trajectory_harmful_minus_benign` | activation_geometry | 0.037 | [-3.921, 3.545] | 0.600 | 0.209256 | 0.599228 |
| 17 | `cv_linear_probe_auc` | supervised_probe_baseline | 0.000 | [0.000, 0.000] | 0.500 | — | — |
| 18 | `dual_use_probe_auc` | supervised_probe_baseline | 0.000 | [0.000, 0.000] | 0.500 | — | — |

Bootstrap resampling unit = model (n=10; positive class = instruct+saferl, negative class = abliterated). rho uses the 7 models with published refusal numbers.

## Held-out confirmation (pre-registered: |d|>1.0 or AUROC>0.9)

- `activation_anisotropy_harmful_minus_benign`: confirmed=False (activation metric; dev-calibrated argmax layer, held-out prompts)
- `refusal_dir_alignment_gap`: confirmed=True (activation metric; dev-calibrated argmax layer, held-out prompts)
- `rank_drop_harmful_minus_benign`: confirmed=True (activation metric; dev-calibrated argmax layer, held-out prompts)
- `layer_argmax_separation`: confirmed=False (activation metric; dev-calibrated argmax layer, held-out prompts)
- `participation_ratio_per_layer`: confirmed=True (activation metric; dev-calibrated argmax layer, held-out prompts)

## Zero-shot probe transfer (fit on model X dev, eval on Y held-out)

```
fit\eval                               Qwen2.5-3B-I  Qwen2.5-3B-a    Qwen3-1.7B  Qwen3-1.7B-B      Qwen3-4B  Qwen3-4B-Bas  Qwen3-4B-Ins  Qwen3-4B-Ins  Qwen3-4B-Saf  Qwen3-4B-abl
Qwen2.5-3B-Instruct                            1.00          0.99          0.59          0.52 — — — — — —
Qwen2.5-3B-abliterated                         0.99          0.99          0.46          0.58 — — — — — —
Qwen3-1.7B                                     0.39          0.42          0.99          1.00 — — — — — —
Qwen3-1.7B-Base                                0.91          0.67          0.97          1.00 — — — — — —
Qwen3-4B                              — — — —          1.00          0.99          1.00          0.95          1.00          1.00
Qwen3-4B-Base                         — — — —          0.99          1.00          0.99          0.89          1.00          0.94
Qwen3-4B-Instruct-2507                — — — —          1.00          0.97          1.00          0.99          1.00          1.00
Qwen3-4B-Instruct-2507-abliterated    — — — —          1.00          1.00          1.00          0.99          1.00          1.00
Qwen3-4B-SafeRL                       — — — —          1.00          0.97          1.00          0.97          1.00          1.00
Qwen3-4B-abliterated                  — — — —          0.98          0.98          0.95          0.97          0.98          1.00
```