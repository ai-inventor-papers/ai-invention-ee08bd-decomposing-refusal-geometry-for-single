#!/usr/bin/env python3
"""Model panel declaration: roles, families, sizes, chat-template groups."""

from dataclasses import dataclass

@dataclass
class PanelModel:
    hf_id: str
    role: str            # base | instruct | saferl | abliterated
    family: str          # qwen3 | qwen2.5 | llama3 | gemma2 | smollm2
    params_b: float
    chat: bool           # uses chat template
    note: str = ""

PANEL = [
    # ---- Qwen3-4B lineage (core, per artifact plan) ----
    PanelModel("Qwen/Qwen3-4B-Base", "base", "qwen3", 4.0, False, "pretrained base; different prompt format, held separate in correlations"),
    PanelModel("Qwen/Qwen3-4B-Instruct-2507", "instruct", "qwen3", 4.0, True, "official instruct post-trained"),
    PanelModel("Qwen/Qwen3-4B-SafeRL", "saferl", "qwen3", 4.0, True, "official safety-RL tuned sibling"),
    PanelModel("huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated", "abliterated", "qwen3", 4.0, True, "abliterated via Instruct-2507"),
    PanelModel("huihui-ai/Qwen3-4B-abliterated", "abliterated", "qwen3", 4.0, True, "abliterated via Qwen3-4B (thinking) template"),
    # ---- Qwen3-1.7B lineage ----
    PanelModel("Qwen/Qwen3-1.7B-Base", "base", "qwen3", 1.7, False, ""),
    PanelModel("Qwen/Qwen3-1.7B", "instruct", "qwen3", 1.7, True, ""),
    PanelModel("huihui-ai/Qwen3-1.7B-abliterated", "abliterated", "qwen3", 1.7, True, ""),
    # ---- Qwen2.5-3B lineage (abliterated is SFT variant) ----
    PanelModel("Qwen/Qwen2.5-3B-Instruct", "instruct", "qwen2.5", 3.1, True, ""),
    PanelModel("huihui-ai/Qwen2.5-3B-Instruct-abliterated-SFT", "abliterated", "qwen2.5", 3.1, True, "abliterated + SFT"),
    # ---- Llama-3.2-3B lineage ----
    PanelModel("meta-llama/Llama-3.2-3B-Instruct", "instruct", "llama3", 3.2, True, "gated repo; use fallback if access denied"),
    PanelModel("huihui-ai/Llama-3.2-3B-Instruct-abliterated", "abliterated", "llama3", 3.2, True, ""),
    # ---- Gemma-2-2b lineage ----
    PanelModel("google/gemma-2-2b-it", "instruct", "gemma2", 2.6, True, "gated repo; use fallback if access denied"),
    PanelModel("mradermacher/gemma-2-2b-it-abliterated-GGUF", "abliterated", "gemma2", 2.6, False, "PLACEHOLDER — GGUF not loadable; replaced by fallback"),
]

# Substitutes for gated/unloadable repos, decided BEFORE any metric tuning.
SUBSTITUTIONS = {
    # gemma-2-2b-it-abliterated has no safetensors sibling; use Qwen3-1.7B abliterated+instruct pair already
    # included, plus a size-matched uncensored full-precision model:
    "mradermacher/gemma-2-2b-it-abliterated-GGUF": "ops-malware/smollm2-1.7b-abliterated",
}
# llama3 instruct may be gated; its abliterated sibling (huihui) is public and is an
# abliteration OF the instruct model, so if meta-llama is inaccessible we keep only
# the abliterated member and record the gap.

def resolved_panel():
    """Return final panel with substitutions applied and gated repos attempted."""
    out = []
    for m in PANEL:
        rid = SUBSTITUTIONS.get(m.hf_id, m.hf_id)
        if rid != m.hf_id:
            m.note = f"{m.note} | substituted {m.hf_id} -> {rid} (declared pre-tuning)"
            m.hf_id = rid
        out.append(m)
    return out

if __name__ == "__main__":
    for m in resolved_panel():
        print(f"{m.hf_id:60s} {m.role:12s} {m.family:9s} {m.params_b}B chat={m.chat}")