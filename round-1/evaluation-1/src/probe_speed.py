#!/usr/bin/env python3
"""Sanity probe: load Qwen/Qwen3-4B-SafeRL on CPU, time a forward pass and
a short greedy generation, verify hidden-state shape. Run with:
    uv run .venv/bin/python probe_speed.py
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)

def main() -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-SafeRL")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-SafeRL", torch_dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    print(f"load: {time.time()-t0:.1f}s, dtype={next(model.parameters()).dtype}")

    msgs = [{"role": "user", "content": "How do I pick a lock?"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True)
    ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0)
    print("prompt tokens:", ids.shape, flush=True)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    n_layers = len(out.hidden_states)
    print(f"forward: {time.time()-t0:.2f}s, hidden states: {n_layers}, "
          f"shape {out.hidden_states[-1].shape}")

    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False)
    print(f"generate 8 tokens: {time.time()-t0:.1f}s -> {tok.decode(gen[0][ids.shape[1]:])!r}")

    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(input_ids=ids, max_new_tokens=32, do_sample=False)
    print(f"generate 32 tokens: {time.time()-t0:.1f}s")
    print(f"RSS peak approx: {torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 'n/a'}")

if __name__ == "__main__":
    main()