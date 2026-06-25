"""Generate construction JSON for held-out problems with the fine-tuned adapter.
Runs on the GPU server (no GeoGebra needed). Writes predictions.jsonl which you
copy back to the Windows machine for functional (render) eval.

    python infer.py            # uses eval_problems.jsonl -> predictions.jsonl
"""
import os, io, json, argparse

ADAPTER   = "qwen25coder3b-geom-lora"
IN_FILE   = "eval_problems.jsonl"
OUT_FILE  = "predictions.jsonl"
MAX_NEW   = 1280
MAX_SEQ   = 1536

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


def extract_json(s):
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try: return json.loads(s[start:i+1])
                        except json.JSONDecodeError: break
        start = s.find("{", start + 1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--in_file", default=IN_FILE)
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=MAX_SEQ, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    probs = [json.loads(l) for l in io.open(args.in_file, encoding="utf-8") if l.strip()]
    out = []
    for i, p in enumerate(probs, 1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p["problem_statement"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        gen = model.generate(input_ids=inputs, max_new_tokens=MAX_NEW,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0][inputs.shape[1]:], skip_special_tokens=True)
        obj = extract_json(text)
        steps = obj.get("construction_steps") if isinstance(obj, dict) else None
        out.append({"source": p.get("source"), "id": p.get("id"),
                    "problem_statement": p["problem_statement"],
                    "predicted_construction_steps": steps,
                    "parsed_ok": bool(steps), "raw": text})
        print(f"[{i}/{len(probs)}] parsed_ok={bool(steps)}  {p['problem_statement'][:50]}")

    with io.open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(r["parsed_ok"] for r in out)
    print(f"\nwrote {args.out}: {ok}/{len(out)} produced parseable JSON")


if __name__ == "__main__":
    main()
