"""Model registry: the 9-model panel (planned 4 + verified substitutes + 2nd lineage)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    short: str          # directory-safe name
    lineage: str        # qwen3-4b | qwen3-4b-2507 | qwen3-1.7b | qwen2.5-3b
    role: str           # base | instruct | saferl | abliterated
    template: str       # chat | raw
    substitution: str | None = None  # why it differs from the artifact plan, if it does


REGISTRY: list[ModelSpec] = [
    ModelSpec("Qwen/Qwen3-4B-Base", "Qwen3-4B-Base", "qwen3-4b", "base", "raw"),
    ModelSpec("Qwen/Qwen3-4B", "Qwen3-4B", "qwen3-4b", "instruct", "chat"),
    ModelSpec("Qwen/Qwen3-4B-SafeRL", "Qwen3-4B-SafeRL", "qwen3-4b", "saferl", "chat"),
    # planned huihui-ai/Qwen3-4B-abliterated is metadata-only in the shared cache (weights absent,
    # verified 2026-09-23); mlabonne's abliteration of the same Qwen3-4B base is fully cached -> substitute.
    ModelSpec(
        "mlabonne/Qwen3-4B-abliterated", "Qwen3-4B-abliterated", "qwen3-4b", "abliterated", "chat",
        substitution="huihui-ai/Qwen3-4B-abliterated had no cached weights; mlabonne/Qwen3-4B-abliterated "
        "(FailSpy orthogonalization recipe on the same Qwen3-4B) is fully cached and is the same role",
    ),
    # second, newer Qwen3-4B line (Instruct-2507): full safe/abliterated contrast, cached
    ModelSpec("Qwen/Qwen3-4B-Instruct-2507", "Qwen3-4B-Instruct-2507", "qwen3-4b-2507", "instruct", "chat"),
    ModelSpec(
        "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated", "Qwen3-4B-Instruct-2507-abliterated",
        "qwen3-4b-2507", "abliterated", "chat",
    ),
    # second size point in the Qwen3 family
    ModelSpec("Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base", "qwen3-1.7b", "base", "raw"),
    ModelSpec("Qwen/Qwen3-1.7B", "Qwen3-1.7B", "qwen3-1.7b", "instruct", "chat"),
    # second architecture family (Qwen2.5): instruct + abliterated-SFT pair
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct", "qwen2.5-3b", "instruct", "chat"),
    ModelSpec(
        "huihui-ai/Qwen2.5-3B-Instruct-abliterated-SFT", "Qwen2.5-3B-abliterated", "qwen2.5-3b",
        "abliterated", "chat",
    ),
]

# models with a same-lineage safe/abliterated contrast usable for the primary separability task
LINEAGE_CONTRASTS = {
    "qwen3-4b": ("Qwen3-4B", "Qwen3-4B-SafeRL", "Qwen3-4B-abliterated"),
    "qwen3-4b-2507": ("Qwen3-4B-Instruct-2507", None, "Qwen3-4B-Instruct-2507-abliterated"),
    "qwen3-1.7b": ("Qwen3-1.7B", None, None),
    "qwen2.5-3b": ("Qwen2.5-3B-Instruct", None, "Qwen2.5-3B-abliterated"),
}

SYSTEM_PROMPT = "You are a helpful assistant."  # identical across all template models
