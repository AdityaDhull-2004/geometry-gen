"""Generate -> validate -> repair loop, run on the GPU server.

Reads the round-0 predictions (predictions.jsonl) and the held-out problems
(eval_problems.jsonl). For each construction that fails the post_filter gate, it
feeds the failing step + error back to the model and asks for a fix, up to
MAX_RETRIES times, re-validating each attempt. Reports the accept rate before vs
after repair, and writes repaired_predictions.jsonl.

Requires (in this same dir on the server): validator.py, checks.py,
post_filter.py, test_harness.html  +  Playwright/Chromium installed.
    python repair_loop.py
"""
import os, io, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from validator import GeoGebraValidator
from post_filter import PostFilter, extract_json

ADAPTER     = "qwen25coder3b-geom-lora"
PROB_FILE   = "eval_problems.jsonl"
PRED_FILE   = "predictions.jsonl"
OUT_FILE    = "repaired_predictions.jsonl"
MAX_RETRIES = 2
MAX_NEW     = 1280
TRIALS      = 7

SYSTEM_PROMPT = (
    "You convert a natural-language geometry problem into an ordered GeoGebra "
    "construction that draws an interactive diagram of it.\n"
    "Reply with ONLY a JSON object: {\"construction_steps\": [ ... ]}.\n"
    "Each step is an object with keys:\n"
    "  name    - a unique identifier referenced by later steps\n"
    "  type    - \"final\" (shown) or \"construction\" (hidden helper)\n"
    "  class   - point | line | segment | ray | circle | conic | locus | number | angle\n"
    "  label   - (optional) caption to display\n"
    "  command - native GeoGebra syntax, referencing earlier step names\n"
    "Steps must be in constructible order (every name used is defined earlier). "
    "Free base points use coordinates; everything else is derived with GeoGebra "
    "commands (Midpoint, Intersect, Circle, AngleBisector, Reflect, ...).\n"
    "Mark as \"final\" every object the problem statement names or asks about "
    "(the points, lines and circles it mentions); use \"construction\" only for "
    "auxiliary scaffolding the statement does not mention."
)

REPAIR_INSTRUCTION = (
    "That construction did not produce a valid figure.\n"
    "Problem: {reason}\n"
    "Return a corrected JSON object (same schema). Fix this specifically: make sure "
    "EVERY name is defined by an earlier step before it is used; use exact GeoGebra "
    "command names and casing (e.g. Tangent, not tangent; Line(B, C), not Line_B_C; "
    "Midpoint takes points or a segment, not a line); and avoid constructions that "
    "put a point at infinity (parallel lines, etc.). Reply with ONLY the JSON object."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--retries", type=int, default=MAX_RETRIES)
    args = ap.parse_args()

    probs = {p["problem_statement"]: p for p in
             (json.loads(l) for l in io.open(PROB_FILE, encoding="utf-8") if l.strip())}
    preds = [json.loads(l) for l in io.open(PRED_FILE, encoding="utf-8") if l.strip()]

    from unsloth import FastLanguageModel
    import torch
    # Load at a LONG context: repair prompts include the full bad output + the
    # error feedback (~2-3k tokens), well beyond the 1536 training length. Qwen
    # supports 32k; a longer KV cache is cheap at inference. 1536 here silently
    # truncated the repair instruction and crippled the loop.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=8192, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    def generate(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True,
                                            return_tensors="pt").to(model.device)
        out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    results = []
    before = after = 0
    with GeoGebraValidator(harness_dir=HERE) as v:
        pf = PostFilter(validator=v)
        for i, pr in enumerate(preds, 1):
            stmt = pr["problem_statement"]
            steps = pr.get("predicted_construction_steps")
            text = pr.get("raw", "")
            ok, reasons = (pf.accept(steps, trials=TRIALS) if steps
                           else (False, ["no parseable JSON"]))
            if ok:
                before += 1
            attempts = 0
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": stmt}]
            while not ok and attempts < args.retries:
                attempts += 1
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": REPAIR_INSTRUCTION.format(reason=reasons[0])}]
                text = generate(messages)
                obj = extract_json(text)
                steps = obj.get("construction_steps") if isinstance(obj, dict) else None
                ok, reasons = (pf.accept(steps, trials=TRIALS) if steps
                               else (False, ["no parseable JSON"]))
            if ok:
                after += 1
            results.append({"problem_statement": stmt, "ok": ok, "attempts": attempts,
                            "construction_steps": steps if ok else None,
                            "final_reason": None if ok else reasons[0]})
            tag = "OK" if ok else "FAIL"
            print(f"[{i:>2}] {tag:4} (repairs={attempts})  {stmt[:50]}")

    with io.open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(preds)
    print(f"\n=== repair loop over {n} held-out problems (<= {args.retries} retries) ===")
    print(f"valid BEFORE repair : {before}/{n}  ({100*before/n:.0f}%)")
    print(f"valid AFTER  repair : {after}/{n}  ({100*after/n:.0f}%)")
    print(f"recovered by repair : {after-before}")
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
