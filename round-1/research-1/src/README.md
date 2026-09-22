# Single-Model Safety Metric Literature & Ground-Truth Survey

## What this does

This research artifact surveys the landscape needed to design a cheap, reference-free safety metric for LLMs. It covers:

1. **Model family verification** — All Qwen3-4B variants (Base, Instruct, SafeRL, abliterated) verified on HuggingFace
2. **Prior work** — 10+ activation- and weight-based safety methods cataloged with structured comparison
3. **Ground truth** — Published benchmark scores for Qwen3-4B and SafeRL from official sources
4. **Metric design inputs** — 15 candidate formulas from literature, gap analysis, prompt-set recommendations

## Key files

| File | Description |
|------|-------------|
| `research_out.json` | Structured research output with all findings, sources, and citations |
| `.sdk_openhands_agent_struct_out.json` | Pipeline struct output (same content, different schema) |
| `write_output.py` | Python script that generated research_out.json |
| `.aii/manifest.yaml` | Artifact manifest |

## Layout

```
.
├── .aii/
│   └── manifest.yaml
├── .sdk_openhands_agent_struct_out.json
├── research_out.json
├── write_output.py
└── README.md
```

## Key findings

- **Qwen3-4B lineage is complete**: Base, Instruct, SafeRL, and two abliterated variants (huihui-ai, mlabonne) all exist
- **Comparison lineages populated**: Llama-3.1-8B, Gemma-2-9B, Mistral-7B all have instruct + abliterated pairs
- **No existing reference-free metric**: All prior methods (AMS, RAS/SafeVec, Two-Signal Audit) require a reference model
- **Missing ground truth**: No published TrustLLM/AIR-Bench/HarmBench/XSTest scores for Qwen3-4B-SafeRL
- **SafeRL training caveat**: Qwen3-235B judge was the training reward signal — WildGuard scores (98.1 safety rate) are the usable external ground truth
- **Coupling metric gap**: No prior work combines activation compression (effective rank) with cross-layer coherence in a single-model, reference-free setting

## Restoring removed files

No files were removed. All artifacts are text/code files under the auto-keep floor.