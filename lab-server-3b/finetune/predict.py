"""Interactive inference for the fine-tuned model (run on the GPU lab box).

Type a geometry problem; it generates a GeoGebra construction, validates it with
the post_filter, repairs on failure (up to --retries), and prints a clean JSON
object you paste into renderer.html on your laptop to see the interactive diagram.

Modes:
    python predict.py                      # interactive: type problems, Ctrl-D to quit
    python predict.py "Let ABC be ..."     # one-shot
    python predict.py --no-validate ...    # skip GeoGebra (just generate), faster

Needs (same dir): validator.py, checks.py, post_filter.py, test_harness.html,
the adapter folder, and Playwright/Chromium (already installed for the repair loop).
"""
import os, io, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "datagen"))
from post_filter import extract_json

ADAPTER, MAX_SEQ, MAX_NEW = "qwen25coder3b-geom-lora", 8192, 1280

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
    "That construction did not produce a valid figure.\nProblem: {reason}\n"
    "Return a corrected JSON object (same schema). Define EVERY name before use, "
    "use exact GeoGebra command names/casing (Tangent not tangent; Line(B, C) not "
    "Line_B_C), and avoid points at infinity. Reply with ONLY the JSON object."
)


def log(*a):  # status goes to stderr so stdout stays copy-pasteable
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", nargs="*", help="problem text (omit for interactive)")
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--out", default=None, help="also write the latest JSON here")
    args = ap.parse_args()

    log("loading model ...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=MAX_SEQ, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    def generate(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True,
                                            return_tensors="pt").to(model.device)
        out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    pf = ctx = None
    if not args.no_validate:
        from validator import GeoGebraValidator
        from post_filter import PostFilter
        ctx = GeoGebraValidator(harness_dir=os.path.dirname(HERE))
        ctx.start()
        pf = PostFilter(validator=ctx)
        log("validator ready.")

    def solve(stmt):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stmt}]
        text = generate(messages)
        obj = extract_json(text)
        steps = obj.get("construction_steps") if isinstance(obj, dict) else None
        ok, reasons = (True, []) if (steps and pf is None) else (
            pf.accept(steps, trials=5) if (steps and pf) else (False, ["no JSON parsed"]))
        tries = 0
        while pf is not None and not ok and tries < args.retries:
            tries += 1
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": REPAIR_INSTRUCTION.format(reason=reasons[0])}]
            text = generate(messages)
            obj = extract_json(text)
            steps = obj.get("construction_steps") if isinstance(obj, dict) else None
            ok, reasons = (pf.accept(steps, trials=5) if steps else (False, ["no JSON parsed"]))
        return ok, steps, reasons, tries

    def emit(stmt):
        ok, steps, reasons, tries = solve(stmt)
        status = "VALID" if ok else ("UNVALIDATED" if pf is None else "INVALID")
        log(f"[{status}] repairs={tries}" + ("" if ok or pf is None else f"  reason: {reasons[0]}"))
        out_obj = {"problem_statement": stmt, "construction_steps": steps or []}
        js = json.dumps(out_obj, ensure_ascii=False, indent=2)
        print("\n=== CONSTRUCTION JSON (paste into renderer.html) ===")
        print(js)
        print("=== END ===\n")
        if args.out:
            io.open(args.out, "w", encoding="utf-8").write(js)
            log(f"wrote {args.out}")

    try:
        if args.problem:
            emit(" ".join(args.problem))
        else:
            log("\nInteractive. Type a problem and press Enter (Ctrl-D to quit).")
            while True:
                try:
                    stmt = input("\nProblem> ").strip()
                except EOFError:
                    break
                if stmt:
                    emit(stmt)
    finally:
        if ctx:
            ctx.stop()


if __name__ == "__main__":
    main()
