"""Cross-dataset near-duplicate scan: each NEW problem (new_geometry_problems.jsonl)
vs every GOLD (geometry_problems.jsonl) + GENERATED (datagen/generated_problems.jsonl).
Flags by statement token-Jaccard and by construction-signature Jaccard."""
import os, sys, json, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STOP = set("the a an of to in on and or is be it that this with let prove be are as at "
           "such so for from its their there be which point points line lines we show "
           "respectively again meet meets where again construct".split())

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def toks(s):
    s = re.sub(r"[^a-zA-Z ]", " ", s.lower())
    return {w for w in s.split() if len(w) > 1 and w not in STOP}

def sig(steps):
    heads = []
    for st in steps:
        for m in re.findall(r"([A-Za-z]\w+)\(", st.get("command", "")):
            heads.append(m)
    return set(heads)

def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0

new = load(os.path.join(ROOT, "datasets", "new_geometry_problems.jsonl"))
gold = load(os.path.join(ROOT, "datasets", "geometry_problems.jsonl"))
gen = load(os.path.join(ROOT, "datasets", "generated_problems.jsonl"))
refs = [("GOLD", r) for r in gold] + [("GEN", r) for r in gen]

ref_t = [(tag, r, toks(r["problem_statement"]), sig(r["construction_steps"])) for tag, r in refs]

flagged = []
for r in new:
    if r["id"] == 1:
        continue
    nt, ns = toks(r["problem_statement"]), sig(r["construction_steps"])
    best = (0.0, 0.0, None, None)
    for tag, rr, rt, rs in ref_t:
        sj = jac(nt, rt)
        cj = jac(ns, rs)
        score = sj
        if score > best[0]:
            best = (sj, cj, tag, rr.get("id"))
    sj, cj, tag, rid = best
    if sj >= 0.5 or (sj >= 0.4 and cj >= 0.85):
        flagged.append((r["id"], round(sj, 2), round(cj, 2), tag, rid, r["problem_statement"][:70]))

flagged.sort(key=lambda x: -x[1])
print(f"NEW={len(new)-1}  refs={len(refs)} (gold {len(gold)} + gen {len(gen)})")
print(f"flagged potential duplicates (stmt-Jaccard>=0.5 OR sj>=0.4&csig>=0.85): {len(flagged)}\n")
print(f"{'newid':>5} {'sJ':>4} {'cJ':>4} {'ref':>5}{'#':>4}  statement")
for nid, sj, cj, tag, rid, stmt in flagged:
    print(f"{nid:>5} {sj:>4} {cj:>4} {tag:>5}{rid:>4}  {stmt}")
