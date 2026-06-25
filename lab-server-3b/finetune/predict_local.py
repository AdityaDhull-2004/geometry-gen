"""Run the fine-tuned model FULLY on your laptop (no GPU/SSH): it calls your local
Ollama model, validates with the local GeoGebra harness, repairs on failure, and
prints render-ready JSON to paste into renderer.html.

Prereqs on the laptop:
  - Ollama running with the model imported as 'geom' (see geom.Modelfile)
  - the validator already works here (datagen/post_filter.py + test_harness.html)

    python finetune/predict_local.py                 # interactive
    python finetune/predict_local.py "Let ABC ..."    # one-shot
"""
import os, sys, io, json, argparse, urllib.request, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "datagen"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from post_filter import PostFilter, extract_json
from visibility import apply_visibility_rule

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "geom"

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


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def chat(messages):
    body = json.dumps({"model": MODEL_NAME, "messages": messages, "stream": False,
                       "format": "json",          # grammar-constrain to valid JSON
                       "options": {"temperature": 0, "num_ctx": 8192,
                                   "num_predict": 2048}}).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", nargs="*")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the GeoGebra check + repair (no Playwright needed)")
    args = ap.parse_args()

    validator_cm = contextlib.nullcontext(None) if args.no_validate else PostFilter()
    with validator_cm as pf:                     # pf is None when --no-validate
        def judge(steps):
            if steps is None:
                return False, ["no JSON parsed"]
            if pf is None:
                return True, []                      # generate-only (unvalidated)
            return pf.accept(steps, trials=5)

        def solve(stmt):
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": stmt}]
            text = chat(messages)
            obj = extract_json(text)
            steps = obj.get("construction_steps") if isinstance(obj, dict) else None
            ok, reasons = judge(steps)
            tries = 0
            while not ok and tries < args.retries:
                tries += 1
                fix = ("Your reply was not a single valid JSON object. Output ONLY "
                       "{\"construction_steps\": [ ... ]} with no prose or proof."
                       if steps is None
                       else REPAIR_INSTRUCTION.format(reason=reasons[0]))
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": fix}]
                text = chat(messages)
                obj = extract_json(text)
                steps = obj.get("construction_steps") if isinstance(obj, dict) else None
                ok, reasons = judge(steps)
            return ok, steps, reasons, tries, text

        def emit(stmt):
            ok, steps, reasons, tries, raw = solve(stmt)
            if steps:
                steps = apply_visibility_rule(stmt, steps)   # show statement-named entities
            status = "UNVALIDATED" if pf is None else ("VALID" if ok else "INVALID")
            log(f"[{status}] repairs={tries}"
                + ("" if ok or pf is None else f"  reason: {reasons[0]}"))
            if not steps:                                    # show what the model said
                log("---- model raw output (first 900 chars) ----")
                log(raw[:900])
                log("--------------------------------------------")
            js = json.dumps({"problem_statement": stmt, "construction_steps": steps or []},
                            ensure_ascii=False, indent=2)
            print("\n=== CONSTRUCTION JSON (paste into renderer.html) ===")
            print(js)
            print("=== END ===\n")
            if args.out:
                io.open(args.out, "w", encoding="utf-8").write(js)
                log(f"wrote {args.out}")

        if args.problem:
            emit(" ".join(args.problem))
        else:
            log("Interactive. Type a problem, Enter (Ctrl-Z then Enter on Windows to quit).")
            while True:
                try:
                    stmt = input("\nProblem> ").strip()
                except EOFError:
                    break
                if stmt:
                    emit(stmt)


if __name__ == "__main__":
    main()
