"""Prepare the fine-tuning corpus.

Merges the 82 gold problems (geometry_problems.jsonl) and the 150 generated
problems (datagen/generated_problems.jsonl) into a single chat-formatted dataset
for QLoRA, with a deterministic, source-stratified train/eval split.

Outputs (in this folder):
  train.jsonl          - {"messages":[system,user,assistant]} for training
  eval.jsonl           - same format, held-out (for loss/perplexity eval)
  eval_problems.jsonl  - raw held-out {source,id,problem_statement,construction_steps}
                         for FUNCTIONAL eval (generate -> validate/post_filter)

Run:  python finetune/prepare_data.py
"""
import os, io, json, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # geometry-gen/
sys.path.insert(0, os.path.join(ROOT, "datagen"))
from visibility import apply_visibility_rule
GOLD = os.path.join(ROOT, "geometry_problems.jsonl")
GEN = os.path.join(ROOT, "datagen", "generated_problems.jsonl")

EVAL_FRACTION = 0.13       # ~30 of 232
SEED = 42

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


def load(path, source):
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["_source"] = source
                rows.append(r)
    return rows


def to_messages(prob):
    # relabel so every entity named in the statement is type="final" (the rule
    # we want the model to learn), then make that the training target.
    steps = apply_visibility_rule(prob["problem_statement"], prob["construction_steps"])
    completion = json.dumps({"construction_steps": steps}, ensure_ascii=False)
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prob["problem_statement"]},
        {"role": "assistant", "content": completion},
    ]}


def split(rows, frac, rng):
    rows = rows[:]
    rng.shuffle(rows)
    n_eval = max(1, round(len(rows) * frac))
    return rows[n_eval:], rows[:n_eval]            # train, eval


def main():
    rng = random.Random(SEED)
    gold = load(GOLD, "gold")
    gen = load(GEN, "generated")
    print(f"loaded gold={len(gold)}  generated={len(gen)}  total={len(gold)+len(gen)}")

    # stratify the split by source so eval has both kinds
    g_tr, g_ev = split(gold, EVAL_FRACTION, rng)
    n_tr, n_ev = split(gen, EVAL_FRACTION, rng)
    train_raw = g_tr + n_tr
    eval_raw = g_ev + n_ev
    rng.shuffle(train_raw)
    rng.shuffle(eval_raw)
    print(f"split: train={len(train_raw)}  eval={len(eval_raw)} "
          f"(eval = {len(g_ev)} gold + {len(n_ev)} generated)")

    with io.open(os.path.join(HERE, "train.jsonl"), "w", encoding="utf-8") as f:
        for p in train_raw:
            f.write(json.dumps(to_messages(p), ensure_ascii=False) + "\n")
    with io.open(os.path.join(HERE, "eval.jsonl"), "w", encoding="utf-8") as f:
        for p in eval_raw:
            f.write(json.dumps(to_messages(p), ensure_ascii=False) + "\n")
    with io.open(os.path.join(HERE, "eval_problems.jsonl"), "w", encoding="utf-8") as f:
        for p in eval_raw:
            f.write(json.dumps({"source": p["_source"], "id": p.get("id"),
                                "problem_statement": p["problem_statement"],
                                "construction_steps": p["construction_steps"]},
                               ensure_ascii=False) + "\n")

    # ---- length stats (char-based, with rough token estimates) ----
    def stats(label, vals):
        vals = sorted(vals)
        n = len(vals)
        pct = lambda q: vals[min(n - 1, int(q * n))]
        print(f"  {label:<22} max={vals[-1]:>6}  p95={pct(0.95):>6}  median={pct(0.5):>6}")

    full_chars, comp_chars, step_counts = [], [], []
    for p in (train_raw + eval_raw):
        m = to_messages(p)
        full = sum(len(x["content"]) for x in m["messages"])
        full_chars.append(full)
        comp_chars.append(len(m["messages"][-1]["content"]))
        step_counts.append(len(p["construction_steps"]))

    print("\nlength stats (characters):")
    stats("full example", full_chars)
    stats("completion (output)", comp_chars)
    print("construction step count:")
    stats("steps/problem", step_counts)
    mx = max(full_chars)
    print(f"\nrough token estimate (chars/3.5): full-example max ~= {round(mx/3.5)} tokens")
    print(f"=> set max_seq_len to ~{max(1024, 256 * (round(mx/3.5)//256 + 1))} "
          f"(power-of-256 headroom above the max).")
    print(f"\nwrote train.jsonl / eval.jsonl / eval_problems.jsonl to {HERE}")


if __name__ == "__main__":
    main()
