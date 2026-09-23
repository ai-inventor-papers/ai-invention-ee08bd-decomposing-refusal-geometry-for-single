#!/usr/bin/env python3
"""Stage 1: behavioral metrics. Greedy 32-token responses on 10 harmful +
10 benign prompts per model; keyword classification; cached per model JSON."""

import json
import time

from common import (ROOT, format_prompt, generate_greedy, has_chat_template,
                    keyword_refusal, load_model, logger, release, save_json)

PROMPTS = json.loads((ROOT / "data" / "prompts.json").read_text())


def run_model(m: dict) -> dict:
    out_path = ROOT / "results" / f"behavior_{m['hf_id'].replace('/', '__')}.json"
    if out_path.exists():
        logger.info(f"behavior cached: {m['hf_id']}")
        return json.loads(out_path.read_text())
    t0 = time.time()
    model, tok = load_model(m["hf_id"])
    chat = m["chat"] and has_chat_template(tok)
    rec = {"hf_id": m["hf_id"], "role": m["role"], "family": m["family"],
           "params_b": m["params_b"], "used_chat": chat, "responses": {}}
    for key, plist in (("harmful", PROMPTS["harmful_gen10"]),
                       ("benign", PROMPTS["benign_gen10"])):
        rec["responses"][key] = []
        for p in plist:
            ids = format_prompt(tok, p, chat)
            text = generate_greedy(model, tok, ids, max_new=32)
            rec["responses"][key].append({"prompt": p, "text": text,
                                          "kw_refusal": keyword_refusal(text)})
    rec["B1"] = round(sum(r["kw_refusal"] for r in rec["responses"]["harmful"]) / 10, 4)
    rec["B2"] = round(sum(r["kw_refusal"] for r in rec["responses"]["benign"]) / 10, 4)
    rec["B3"] = round(rec["B1"] - rec["B2"], 4)
    rec["runtime_s"] = round(time.time() - t0, 1)
    save_json(rec, out_path)
    logger.info(f"{m['hf_id']}: B1={rec['B1']} B2={rec['B2']} B3={rec['B3']} "
                f"({rec['runtime_s']}s)")
    release(model)
    return rec


def main() -> None:
    import json as _json
    from models import resolved_panel
    for m in resolved_panel():
        try:
            run_model(m.__dict__)
        except Exception as e:
            logger.error(f"FAILED {m.hf_id}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()