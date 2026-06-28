"""Validate hand-authored candidate problems and append the survivors to
new_geometry_problems.jsonl. Renderability gate only: a construction is kept iff
every step builds a real object across `trials` fresh random-parameter renders.

    python datagen/validate_new.py <candidates.jsonl>            # validate + append
    python datagen/validate_new.py <candidates.jsonl> --check    # report only, no writes

On append, ids continue sequentially from whatever is already in the target file,
and only {id, problem_statement, construction_steps} are written (clean format).
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from validator import GeoGebraValidator

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "datasets", "new_geometry_problems.jsonl")
REJ    = os.path.join(ROOT, "pipeline", "new_rejected.jsonl")


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def next_id():
    mx = 0
    for r in load_jsonl(TARGET):
        if isinstance(r.get("id"), int):
            mx = max(mx, r["id"])
    return mx + 1


def existing_statements():
    return {r.get("problem_statement", "").strip() for r in load_jsonl(TARGET)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--check", action="store_true", help="validate + report only, no writes")
    args = ap.parse_args()

    cands = load_jsonl(args.candidates)
    print(f"loaded {len(cands)} candidate(s) from {os.path.basename(args.candidates)}")

    seen = existing_statements()
    nid = next_id()
    accepted, rejected, dups = [], [], 0

    with GeoGebraValidator() as v:
        for i, prob in enumerate(cands, 1):
            steps = prob.get("construction_steps", [])
            stmt  = prob.get("problem_statement", "").strip()
            short = stmt[:58]

            if stmt in seen:
                dups += 1
                print(f"  [{i:>3}] DUP   {short}")
                continue

            ok, info = v.validate_robust(steps, trials=args.trials)
            if not ok:
                prob["_failure"] = info.get("sample_failure")
                rejected.append(prob)
                print(f"  [{i:>3}] FAIL ({info['clean']}/{info['trials']}) {short}")
                if info.get("sample_failure"):
                    print(f"          -> {info['sample_failure']}")
                continue

            seen.add(stmt)
            accepted.append({"id": nid, "problem_statement": stmt, "construction_steps": steps})
            print(f"  [{i:>3}] PASS  id={nid}  {short}")
            nid += 1

    if not args.check:
        if accepted:
            with open(TARGET, "a", encoding="utf-8") as f:
                for p in accepted:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(REJ, "w", encoding="utf-8") as f:
            for p in rejected:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    verb = "would accept" if args.check else "appended"
    print(f"\n{verb} {len(accepted)} | rejected {len(rejected)} | dup {dups}"
          + ("" if args.check else f" -> {os.path.basename(TARGET)}"))


if __name__ == "__main__":
    main()
