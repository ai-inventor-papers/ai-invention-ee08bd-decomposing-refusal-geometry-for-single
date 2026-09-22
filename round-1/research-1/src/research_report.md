# Single-Model Safety Metric Literature and Ground-Truth Survey

## Summary

This research artifact surveys (1) the Qwen3-4B model family including Base, Instruct, SafeRL, and abliterated variants on HuggingFace with exact repo IDs and properties; (2) prior single-model and activation-based safety evaluation methods - AMS, RAS/SafeVec, Two-Signal Audit, Arditi refusal direction, Wei brittleness, effective-rank audit, SCSV, RepBend, and SAE-based approaches - with structured comparison tables; (3) published benchmark scores from the Qwen3Guard technical report and SafeRL model card (WildJailbreak safety rates, ArenaHard-v2, AIME, GPQA, LCB); (4) standard behavioral benchmark infrastructure (XSTest, HarmBench, JBB-Behaviors, TrustLLM, AdvBench, OR-Bench, WildGuard); (5) concrete candidate metric formulas drawn from the prior work (15+ formulas covering refusal direction sigma, activation cosine gap, weight spectral ratios, effective rank, SVD-rank diagnostics, SAE feature activations, logit margin baselines); and (6) recommendations for prompt sets and gap analysis showing what the proposed coupling metric (compression x coherence) adds over existing published signals. Key findings: no existing method provides a reference-free, single-model, few-prompt safety score - all prior work either requires a reference model, extensive prompt sets, or behavioral generation. The Qwen3-4B-SafeRL model card publishes WildJailbreak safety rates but NOT TrustLLM/AIR-Bench/HarmBench/XSTest scores, so two-refusal-rate ground truth must be constructed from published leaderboard entries. The abliterated variants (huihui-ai, mlabonne) exist and share the chat template with Qwen3-4B-Instruct and SafeRL, enabling direct activation comparison. Abliteration is widely documented as a technique that removes refusal behavior by orthogonalizing the residual stream against a refusal direction [37], with mlabonne publishing an extensive blog and model series demonstrating the approach.

## Research Findings

## 1. Model Family Verification (STEP 0)

### Qwen3-4B Lineage (Official)

Four models in the Qwen3-4B lineage are verified on HuggingFace:

- **Qwen/Qwen3-4B-Base** [1]: 4.0B params, 36 layers, 32 Q / 8 KV heads, 32K context, Apache 2.0. Pretrained only (no chat template). Released May 14, 2025.
- **Qwen/Qwen3-4B** [2]: Same architecture as Base, with instruction tuning and chat template. Non-thinking and thinking modes supported.
- **Qwen/Qwen3-4B-Instruct-2507** [3]: Updated non-thinking-mode version.
- **Qwen/Qwen3-4B-SafeRL** [4]: Safety-aligned via RL with Qwen3Guard-Gen reward signal. Released October 16, 2025 (arXiv: 2510.14276). Shares chat template with Qwen3-4B.

All three non-Base models (Qwen3-4B, Qwen3-4B-SafeRL, and abliterated variants) share the same chat template, enabling direct activation comparison. The Base model uses a different format.

### Abliterated Variants

Two independently produced abliterated variants exist, created using the abliteration technique documented by mlabonne [37]:

- **huihui-ai/Qwen3-4B-abliterated** [5]: Uses the remove-refusals-with-transformers technique. BF16 weights. A v2 variant also exists.
- **mlabonne/Qwen3-4B-abliterated** [6]: Uses a newer abliteration technique with accumulation and better evaluations. F32 weights. 0.6B through 30B-A3B size variants also available.

### Comparison Model Lineages

Verified pairs/triplets with safety-tuned + abliterated siblings:

- **Llama-3.1-8B**: meta-llama/Llama-3.1-8B-Instruct (reference) <-> huihui-ai/Meta-Llama-3.1-8B-Instruct-abliterated [7]
- **Gemma-2-9B**: google/gemma-2-9b-it (reference) <-> IlyaGusev/gemma-2-9b-it-abliterated [8]
- **Mistral-7B**: mistralai/Mistral-7B-Instruct-v0.3 <-> evolveon/Mistral-7B-Instruct-v0.3-abliterated [9] and huihui-ai/Mistral-7B-Instruct-v0.3-abliterated [10]
- **Qwen3-8B**: Qwen/Qwen3-8B <-> huihui-ai/Qwen3-8B-abliterated [11]

The AMS paper [12] validates across 14 model configurations spanning 4 architecture families (Llama, Gemma, Qwen, Mistral), confirming these families are well-populated.

## 2. Qwen3-4B Ground Truth (STEP 1)

### Published Scores from Qwen3-4B-SafeRL Model Card [4]

The model card [4] publishes a performance table with these non-thinking mode scores:
- Qwen3-4B: Safety Rate (Qwen3-235B judge) = 47.5, Safety Rate (WildGuard) = 64.7, Refusal (WildGuard) = 12.9, ArenaHard-v2 = 9.5, AIME25 = 19.1, LCB-v6 = 26.4, GPQA = 41.7
- Qwen3-4B-SafeRL: Safety Rate (Qwen3-235B judge) = 86.5, Safety Rate (WildGuard) = 98.1, Refusal (WildGuard) = 5.3, ArenaHard-v2 = 10.7, AIME25 = 18.2, LCB-v6 = 27.7, GPQA = 40.8

**Key observation**: SafeRL dramatically improves safety (47.5->86.5 on Qwen3-235B judge, 64.7->98.1 on WildGuard) while preserving or slightly improving capability scores. **Critical caveat**: The safety judge is Qwen3-235B (Qwen3Guard-Gen), which was also the training reward signal for SafeRL [4]. The user explicitly prohibits using Qwen3Guard as a judge for SafeRL. Ground truth must therefore rely on WildGuard-judged safety rate (98.1) and WildGuard refusal rate (5.3), plus external benchmarks.

### Qwen3-4B Capability Scores

From the Qwen3 technical report [13] and blog [14]: Qwen3-4B matches Qwen2.5-7B-Instruct performance. The blog states 'even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct' [14].

### Missing Ground Truth

No published TrustLLM, AIR-Bench, HarmBench, or XSTest scores were found for Qwen3-4B-SafeRL specifically. The Qwen3Guard technical report [15] evaluates Qwen3Guard models (not SafeRL) on HarmBench, XSTest, ToxicChat, etc. The WildJailbreak benchmark (used in SafeRL training) provides safety rates but the full prompt set is not publicly documented.

## 3. Prior Single-Model / Activation-Based Safety Work (STEP 2)

### 3.1 Refusal Direction (Arditi et al. 2024)

The foundational paper [16]: Refusal is mediated by a **single direction** in the residual stream across 13 open-source chat models up to 72B. Erasing this direction prevents refusal; adding it elicits refusal on harmless inputs. This finding directly motivates the abliteration technique used by huihui-ai [5] and mlabonne [6, 37].

- **Signal**: Difference-of-means direction from harmful vs. benign activations
- **Reference dependence**: Yes (requires base or aligned reference)
- **Prompts needed**: ~50 harmful + 50 benign
- **Venue**: NeurIPS 2024

### 3.2 AMS - Activation-Based Model Scanner (Messenger 2026)

[12] Measures geometric structure (cluster separation) and direction vectors in activation space. 71% LOOCV accuracy on 14 models. The harmful-content concept signal predicts compliance with Pearson r=-0.546 (p=0.043). Four-class taxonomy: (i) training removal (cluster collapse), (ii) weight-orthogonalization abliteration, (iii) rotation-without-collapse abliteration, (iv) behavioral fine-tuning (undetectable by activation-only probing).

- **Key numbers**: Llama-3.1-Instruct: cluster separation 4-8; Llama-abliterated: 3.33, direction cosine sim 0.30; Gemma-2-9b-abliterated: 4.54, direction cosine sim 0.84
- **Venue**: IEEE Access 2026

### 3.3 RAS / SafeVec (Huang et al. 2026)

[17] White-box evaluation procedure that, given a safety-aligned reference model, extracts layer-wise refusal directions and scores a target model by cosine similarity. Calibrated 0-100 RAS score. Tested on Llama-3.1-8B, Gemma-3-4B, Qwen2.5-7B. **Requires reference model + calibration set - NOT reference-free.**

### 3.4 Two-Signal Audit (Hurtado 2026)

[18] Combines two complementary signals: a reference-anchored activation refusal-gap ratio and a weight-recovery energy. On a 273-checkpoint registry spanning Qwen, DeepSeek-distilled Qwen, Llama, and Gemma, their z-sum separates 57 public abliterations from 37 benign edits at AUROC 0.95. Leave-one-family-out balanced accuracy 0.89. Signals are negatively correlated (r=-0.41).

### 3.5 Brittleness of Safety Alignment (Wei et al. 2024)

[19] Safety alignment is a low-rank concentrated edit: pruning ~5% of parameters removes safety without impacting utility. ICML 2024.

### 3.6 Effective-Rank Audit (Nakamura 2026)

[20] Formalizes single-refusal-direction as a continuous quantity. On Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen-2.5-7B-Instruct: the effective rank rho_epsilon is {0.0029, 0.0048, 0.0044}. The Arditi direction is recovered at |cos| in {0.77, 0.86, 0.50}. The paper finds that rho_epsilon is a diagnostic for fragility, not a target whose mechanical inflation buys robustness.

### 3.7 SCSV (Gu et al. 2025)

[21] Safety-critical singular vectors localization in weight matrices. ACL 2025.

### 3.8 RepBend (Yousefpour et al. 2025)

[22] Representation bending via activation steering differences. ACL 2025.

### 3.9 SAE-Based Safety Work

SAFER [23], GSAE [24], Understanding Refusal with SAEs [25], Safe-SAIL [26].

### 3.10 Abliteration Off-Target Effects (Fafula 2026)

[27] Abliteration has off-target effects - not a scalpel.

### 3.11 Comparative Abliteration Analysis

[28] Large-scale comparison using JBB and HarmBench.

## 4. Behavioral Ground-Truth Sources (STEP 3)

| Benchmark | Description | Prompts | Venue | Notes |
|-----------|-------------|---------|-------|-------|
| XSTest [29] | Over-refusal on 250 safe prompts | 450 | NAACL 2024 | Measures false refusal |
| HarmBench [30] | Standardized red-teaming | 33 LLMs | 2024 | Automated classifiers |
| JBB [31] | 100 harmful behaviors | 100 | NeurIPS 2024 | Official leaderboard |
| TrustLLM [32] | Six dimensions | 30+ datasets | ICML 2024 | Comprehensive |
| AdvBench [33] | 520 harmful behaviors | 520 | 2023 | Saturated/leakage risk |
| OR-Bench [34] | Over-refusal | Multiple | 2024 | Complementary |
| AIR-Bench 2024 [35] | Regulation-aligned safety | Multiple | Stanford | No Qwen3-SafeRL scores |

## 5. Metric-Design Input Extraction (STEP 4)

### Candidate Metric Formulas

1. **Refusal Direction cluster separation** (cluster separation) [12] - Reference-free
2. **Direction Cosine Similarity** [16, 17] - Requires reference
3. **Activation Gap Ratio** [18] - Requires reference
4. **Weight Rank-1 Energy** [18] - Requires base model
5. **Effective Rank rho_epsilon** [20] - Requires aligned+base pair
6. **RAS Score** [17] - Requires reference + calibration set
7. **Linear Probe AUROC** [12, 29] - Reference-free once trained
8. **Logit Gap Margin** (black-box) - Reference-free
9. **Weight Spectral Ratios** [19, 21] - Requires base
10. **SAE Feature Activation** [23, 25, 26] - Requires trained SAE
11. **Cross-Layer Coherence** (novel) - Reference-free
12. **Effective Rank of Activation Covariance** [36] - Reference-free
13. **Benign Coherence (Noise Floor)** - Control metric
14. **Refusal Token Probability** (black-box) - Reference-free
15. **Weight Norm Deviation** - Requires base

### Gap: What the Coupling Metric Adds

The proposed **coupling metric (compression x coherence)** fills a gap: no existing method provides a reference-free, single-model, few-prompt safety score. Prior work either requires a reference model (AMS Tier 2, RAS, Two-Signal Audit) or extensive prompt sets (behavioral benchmarks). The abliteration blog [37] demonstrates the practical importance of such a metric by showing how easily safety can be removed from open-weight models.

### Prompt-Set Recommendation

1. 100 JBB-Behaviors harmful prompts (preferred over AdvBench due to leakage)
2. 100 XSTest safe-looking prompts for over-refusal
3. 50 AdvBench for exploratory activation extraction
4. 100 benign baseline prompts matched for length

## Sources

[1] [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) (Qwen Team; 2025) — Base pretrained Qwen3-4B. 4.0B params, 36 layers, 32K context, Apache 2.0.

[2] [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) (Qwen Team; 2025) — Instruction-tuned Qwen3-4B with chat template, thinking/non-thinking modes.

[3] [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Qwen Team; 2025) — Updated non-thinking mode version of Qwen3-4B.

[4] [Qwen/Qwen3-4B-SafeRL](https://huggingface.co/Qwen/Qwen3-4B-SafeRL) (Qwen Team; 2025) — Safety-aligned Qwen3-4B via RL with Qwen3Guard-Gen reward. Model card publishes: WildGuard safety rate 98.1 (non-thinking), WildJailbreak safety rate 86.5. ArenaHard-v2 10.7, AIME25 18.2, LCB-v6 27.7, GPQA 40.8 (non-thinking).

[5] [huihui-ai/Qwen3-4B-abliterated](https://huggingface.co/huihui-ai/Qwen3-4B-abliterated) (huihui-ai; 2026) — Abliterated Qwen3-4B using remove-refusals-with-transformers. BF16. A v2 also exists.

[6] [mlabonne/Qwen3-4B-abliterated](https://huggingface.co/mlabonne/Qwen3-4B-abliterated) (Maxime Labonne; 2025) — Abliterated Qwen3-4B using newer technique with accumulation. F32 weights. Part of 0.6B-30B-A3B family.

[7] [huihui-ai/Meta-Llama-3.1-8B-Instruct-abliterated](https://huggingface.co/huihui-ai/Meta-Llama-3.1-8B-Instruct-abliterated) (huihui-ai; 2026) — Abliterated Llama-3.1-8B-Instruct comparison model.

[8] [IlyaGusev/gemma-2-9b-it-abliterated](https://huggingface.co/IlyaGusev/gemma-2-9b-it-abliterated) (Ilya Gusev; 2024) — Abliterated Gemma-2-9B-it. Rotation-without-collapse variant (AMS: cluster separation 4.54, direction cosine sim 0.84).

[9] [evolveon/Mistral-7B-Instruct-v0.3-abliterated](https://huggingface.co/evolveon/Mistral-7B-Instruct-v0.3-abliterated) — Abliterated Mistral-7B-Instruct-v0.3.

[10] [huihui-ai/Mistral-7B-Instruct-v0.3-abliterated](https://huggingface.co/huihui-ai/Mistral-7B-Instruct-v0.3-abliterated) (huihui-ai) — Another abliterated Mistral-7B-Instruct-v0.3 variant.

[11] [huihui-ai/Qwen3-8B-abliterated](https://huggingface.co/huihui-ai/Qwen3-8B-abliterated) (huihui-ai; 2026) — Abliterated Qwen3-8B for scale comparison.

[12] [AMS: Detecting Safety Training Modification via Activation Analysis (Messenger 2026)](https://arxiv.org/html/2608.05578) (Glen Messenger; 2026) — Measures cluster separation and direction similarity in activation space. 14 models, 4 families, 71% LOOCV. Four-class taxonomy. IEEE Access 2026.

[13] [Qwen3 Technical Report (Qwen Team 2025)](https://arxiv.org/html/2505.09388v1) (Qwen Team; 2025) — Official report. 6 dense + 2 MoE models, 36T token pretraining, 36 layers for 4B.

[14] [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/) (Qwen Team; 2025) — States Qwen3-4B rivals Qwen2.5-72B-Instruct.

[15] [Qwen3Guard Technical Report (Qwen Team 2025)](https://arxiv.org/html/2510.14276v1) (Qwen Team; 2025) — Guardrail models (Gen/Stream, 0.6B/4B/8B). Safety RL for SafeRL. Evaluates on ToxicChat, HarmBench, XSTest, WildGuardTest.

[16] [Refusal in LLMs Is Mediated by a Single Direction (Arditi et al. 2024)](https://arxiv.org/abs/2406.11717) (Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda; 2024) — Foundational: refusal mediated by single direction across 13 models. NeurIPS 2024.

[17] [RAS/SafeVec: Measuring LLM Safety Through Refusal Alignment (Huang et al. 2026)](https://arxiv.org/html/2606.25750v1) (Chang-Chieh Huang, Yan-Lun Chen, Chia-Mu Yu, Wei-Bin Lee; 2026) — White-box evaluation via cosine similarity with reference refusal directions. RAS 0-100 score. Requires reference + calibration. NOT reference-free.

[18] [Two-Signal Audit: Has This Checkpoint Been Abliterated? (Hurtado 2026)](https://arxiv.org/html/2607.01854) (Gabriel Hurtado; 2026) — Activation refusal-gap ratio + weight-recovery energy. AUROC 0.95 on 273 checkpoints. Complementary signals.

[19] [Assessing the Brittleness of Safety Alignment (Wei et al. 2024)](https://arxiv.org/abs/2402.05162) (Boyi Wei et al.; 2024) — Safety is a low-rank concentrated edit. ~5% parameters. ICML 2024.

[20] [Effective-Rank Audit of Alignment-Induced Activation Shifts (Nakamura 2026)](https://arxiv.org/html/2605.24583v1) (Yuki Nakamura; 2026) — Formalizes rho_epsilon. Llama/Gemma/Qwen: {0.0029, 0.0048, 0.0044}. Arditi |cos|: {0.77, 0.86, 0.50}.

[21] [Safety-Critical Singular Vectors Localization (Gu et al. 2025)](https://aclanthology.org/2025.acl-long.245/) (Peijian Gu, Quan Wang, Zhendong Mao; 2025) — Locates safety-critical singular vectors. ACL 2025.

[22] [Representation Bending for LLM Safety (RepBend, Yousefpour et al. 2025)](https://aclanthology.org/2025.acl-long.1173/) (Ashkan Yousefpour et al.; 2025) — Bends representations apart via activation steering. ACL 2025.

[23] [SAFER: Probing Safety in Reward Models with SAE (Shi et al. 2026)](https://arxiv.org/html/2507.00665) (2026) — SAE-based safety probing in reward models.

[24] [GSAE: Graph-Regularized SAEs for LLM Safety Steering](https://arxiv.org/html/2512.06655v1) (Jehyeok Yeon; 2025) — Graph-regularized SAEs for safety steering.

[25] [Understanding Refusal in LLMs with Sparse Autoencoders (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.338.pdf) (2025) — Refusal-related features in SAE decompositions.

[26] [Safe-SAIL: Fine-grained Safety Landscape via SAE](https://arxiv.org/pdf/2509.18127v2) (2025) — SAE-based safety landscape analysis.

[27] [Abliteration Is Not a Scalpel (Fafula 2026)](https://arxiv.org/html/2607.17427v1) (Aleksander Fafula; 2026) — Off-target effects of refusal removal. Abliteration affects broader behavior.

[28] [Comparative Analysis of LLM Abliteration Methods](https://arxiv.org/pdf/2512.13655) (2025) — Large-scale comparison using JBB and HarmBench.

[29] [XSTest (Rottger et al. 2024)](https://arxiv.org/abs/2308.01263) (Paul Rottger et al.; 2024) — 250 safe prompts, 200 unsafe. Over-refusal detection. NAACL 2024.

[30] [HarmBench (Mazeika et al. 2024)](https://www.harmbench.org/) (Mazeika et al.; 2024) — Standardized red-teaming. 33 LLMs, 18 methods.

[31] [JailbreakBench (Chao et al. 2024)](https://jailbreakbench.github.io/) (Chao et al.; 2024) — 100 harmful behaviors. NeurIPS 2024.

[32] [TrustLLM (Huang et al. 2024)](https://arxiv.org/abs/2401.05561) (Yue Huang et al.; 2024) — Six trustworthiness dimensions. ICML 2024.

[33] [AdvBench (Zou et al. 2023)](https://arxiv.org/abs/2303.11366) (Andy Zou et al.; 2023) — 520 harmful behaviors. Saturated, leakage risk.

[34] [OR-Bench: An Over-Refusal Benchmark](https://arxiv.org/html/2405.20947v2) (2024) — Over-refusal benchmark. Complementary to XSTest.

[35] [AIR-Bench 2024](https://arxiv.org/html/2407.17436) (2024) — Regulation-aligned safety benchmark. Stanford HELM.

[36] [The Effective Rank: A Measure of Effective Dimensionality (Roy and Vetterli 2007)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf) (Olivier Roy, Martin Vetterli; 2007) — Defines effective rank = exp(entropy of normalized squared singular values).

[37] [Uncensor any LLM with abliteration (mlabonne blog 2025)](https://huggingface.co/blog/mlabonne/abliteration) (Maxime Labonne; 2025) — Blog explaining the abliteration technique: identifies refusal direction by comparing residual streams between harmful/harmless samples, orthogonalizes o_proj weights. Demonstrates practical importance of safety metrics by showing how easily safety can be removed from open-weight models.

## Verification

Numbered citations resolve to unique listed sources. Passage checks test text occurrence, not claim truth or entailment. Author/year metadata and locators are not independently verified. Details: `research_verification.json`.

No optional exact passages supplied; no passage checks performed.

## Follow-up Questions

- No published TrustLLM, AIR-Bench, HarmBench, or XSTest scores exist for Qwen3-4B-SafeRL. The ground-truth step must either run these benchmarks or construct two-refusal-rate ground truth from WildJailbreak/WildGuard numbers. Which approach is feasible within 16GB VRAM?
- AMS identifies class (iv) behavioral fine-tuning that preserves activation geometry as undetectable. How many community models fall into this class? Can the coupling metric detect class (iii) rotation-without-collapse abliteration?
- The effective-rank audit shows Arditi direction recovered at |cos|=0.50 on Qwen vs 0.77/0.86 on Llama/Gemma. Does this mean Qwen3 safety is more distributed, making single-direction metrics less reliable?
- RAS paper evaluates on Llama-3.1-8B, Gemma-3-4B, Qwen2.5-7B but not Qwen3-4B. Does SafeVec transfer to Qwen3, and can RAS serve as reference-based validation?

---
*Generated by AI Inventor Pipeline*
