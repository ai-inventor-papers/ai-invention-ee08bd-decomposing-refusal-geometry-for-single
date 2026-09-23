"""Extraction engine: one model loaded at a time, activations + logits + generation captured.

Everything is memory-bounded:
  - bf16 weights ~8 GB RSS for 4B models (14 GB cgroup limit, watchdog aborts at 13.2)
  - per-prompt capture is [L+1, d] bf16 -> stored as float16 on disk, freed after each prompt
  - generation is greedy with an explicit past_key_values cache, max 24 new tokens,
    stopped as soon as a refusal marker or EOS appears (cap ~1 s/token on this 2-CPU box)
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .registry import SYSTEM_PROMPT

MAX_NEW_TOKENS = 24


def build_input(tok: AutoTokenizer, text: str, template: str) -> str:
    """Apply the SAME chat template for template models; raw completion for base models.

    enable_thinking=False is Qwen3's documented soft switch (appends an empty
    <think></think> block) and is applied uniformly to every template model so the
    black-box comparison stays template-consistent and generation stays minimal
    (thinking traces cost 40-80 tokens/model/prompt on CPU). Qwen2.5's Jinja
    template ignores the kwarg.
    """
    if template == "chat":
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # template without the thinking switch
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"User: {text}\nAssistant:"


class Extractor:
    """Loads one model; yields per-prompt captures; caller must call close()."""

    def __init__(self, repo: str, template: str):
        self.repo = repo
        self.template = template
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.bfloat16, attn_implementation="eager"
        )
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        logger.info(f"loaded {repo}: L={self.n_layers} d={self.d_model}")

    # ------------------------------------------------------------------ capture --
    def capture(self, text: str) -> dict:
        """One forward pass. Returns dict with:
        hidden     : float16 [L+1, d]  residual stream at the last prompt token (all layers)
        attn_ent   : float32 [L]       mean attention entropy per layer at the last prompt token
        logits     : float32 [V]       next-token logits at the last prompt position
        n_prompt_tokens, input_text
        """
        input_text = build_input(self.tok, text, self.template)
        enc = self.tok(input_text, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True, output_attentions=True, use_cache=False)
        # hidden_states: tuple (L+1) of [1, T, d] -> last token of each
        hidden = torch.stack([h[0, -1, :] for h in out.hidden_states]).to(torch.float16).cpu().numpy()
        # attentions: tuple (L) of [1, H, T, T]; entropy over attended positions for the LAST query token
        attn_ent = []
        for a in out.attentions:  # [1, H, T, T]
            p = a[0, :, -1, :].to(torch.float32)  # [H, T]
            ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)  # [H]
            attn_ent.append(float(ent.mean()))
        logits = out.logits[0, -1, :].to(torch.float32).cpu().numpy()
        n_tok = int(enc["input_ids"].shape[1])
        attn_nbytes = sum(a.numel() * a.element_size() for a in out.attentions)
        del out, enc
        return {
            "hidden": hidden,                      # fp16 [L+1, d]
            "attn_ent": np.asarray(attn_ent, dtype=np.float32),
            "logits": logits,                      # fp32 [V]
            "n_prompt_tokens": n_tok,
            "input_text": input_text,
            "attn_bytes": attn_nbytes,
        }

    # --------------------------------------------------------------- generation --
    def generate_greedy(self, text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
        """Greedy decode with KV cache; token-by-token so we can stop at refusal markers.

        Thinking models (Qwen3) open with <think>...</think>; we generate past it and
        report both the raw opening and the post-think content separately.
        """
        input_text = build_input(self.tok, text, self.template)
        enc = self.tok(input_text, return_tensors="pt")
        cur = enc["input_ids"]
        past: tuple | None = None
        tokens: list[int] = []
        raw = ""
        opened_think = False
        post_think_start = -1  # token index where post-think content begins
        with torch.no_grad():
            for i in range(max_new_tokens):
                out = self.model(cur if past is None else nxt_t, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
                tokens.append(nxt)
                nxt_t = torch.tensor([[nxt]])
                raw += self.tok.decode([nxt], skip_special_tokens=False)
                if not opened_think and raw.lstrip().startswith("<think>"):
                    opened_think = True
                if opened_think and post_think_start < 0 and "</think>" in raw:
                    post_think_start = len(tokens)
                if nxt == self.tok.eos_token_id:
                    break
                # budget for the post-think clause once thinking has closed
                post_txt = raw.split("</think>", 1)[1] if "</think>" in raw else raw
                post_stripped = post_txt.lstrip()
                post_len = len(tokens) - post_think_start if post_think_start > 0 else len(tokens)
                if post_think_start > 0 and (post_len >= 14 or ("\n" in post_stripped)):
                    break
                if post_think_start < 0 and (len(tokens) >= max_new_tokens):
                    break
        return {
            "text": raw.strip(),
            "post_think": raw.split("</think>", 1)[1].strip() if "</think>" in raw else raw.strip(),
            "opened_think": opened_think,
            "tokens": tokens,
            "n_prompt_tokens": int(enc["input_ids"].shape[1]),
        }

    # --------------------------------------------------------------------- misc --
    def close(self) -> None:
        del self.model
        self.model = None
        gc.collect()
        logger.info(f"closed {self.repo}")
