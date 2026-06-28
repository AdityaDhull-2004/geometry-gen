"""Validate hand-/Claude-authored candidate problems and append the survivors.

This is the *no-API-key* path: instead of calling the Anthropic API to produce
candidates, the problems are authored directly (e.g. by Claude in a Claude Max
chat session) into a candidates JSONL, and this script runs them through the SAME
headless-GeoGebra validator that gates the 82 gold problems. Only constructions
that render cleanly on every random-t trial are kept.

    python datagen/validate_candidates.py datagen/candidates_batch1.jsonl

Accepted problems are appended to datagen/generated_problems.jsonl with ids >= 1000
(continuing past whatever is already there). Rejected ones are written to
datagen/rejected.jsonl with the failing step, so they can be repaired and re-run.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from validator import GeoGebraValidator
from checks import evaluate as eval_checks

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "datasets", "generated_problems.jsonl")
DEFAULT_REJ = os.path.join(HERE, "rejected.jsonl")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def next_id(out_path, floor=1000):
    """Continue ids past whatever is already accepted (>= floor)."""
    mx = floor - 1
    if os.path.exists(out_path):
        for r in load_jsonl(out_path):
            if isinstance(r.get("id"), int):
                mx = max(mx, r["id"])
    return mx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", help="JSONL of candidate problems to validate")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rejected", default=DEFAULT_REJ)
    ap.add_argument("--trials", type=int, default=5,
                    help="random-t render trials; accept only if all are clean")
    args = ap.parse_args()

    cands = load_jsonl(args.candidates)
    print(f"loaded {len(cands)} candidate(s) from {os.path.basename(args.candidates)}")

    nid = next_id(args.out)
    accepted, rejected = [], []

    with GeoGebraValidator() as v:
        for i, prob in enumerate(cands, 1):
            steps = prob.get("construction_steps", [])
            stmt = prob.get("problem_statement", "")[:64]
            checks = prob.get("checks", [])

            # Gate 1: renderability — every step must build on every random-t trial.
            ok, info = v.validate_robust(steps, trials=args.trials)
            if not ok:
                prob["_failure"] = info.get("sample_failure")
                rejected.append(prob)
                print(f"  [{i:>2}] FAIL render ({info['clean']}/{info['trials']})  {stmt}")
                if info.get("sample_failure"):
                    print(f"         -> {info['sample_failure']}")
                continue

            # Gate 2: the stated theorem must hold numerically on every trial.
            check_fail = None
            if checks:
                for _ in range(args.trials):
                    failed, coords = v.probe(steps)
                    if failed:
                        check_fail = f"probe build failed: {failed[0]}"
                        break
                    cok, reason = eval_checks(checks, coords)
                    if not cok:
                        check_fail = reason
                        break
            if check_fail:
                prob["_failure"] = check_fail
                rejected.append(prob)
                print(f"  [{i:>2}] FAIL check  {stmt}")
                print(f"         -> {check_fail}")
                continue

            prob["id"] = nid
            nid += 1
            accepted.append(prob)
            tag = f"{len(checks)} checks" if checks else "render-only"
            print(f"  [{i:>2}] PASS  id={prob['id']}  [{tag}]  {stmt}")

    if accepted:
        with open(args.out, "a", encoding="utf-8") as f:
            for p in accepted:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if rejected:
        with open(args.rejected, "w", encoding="utf-8") as f:
            for p in rejected:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nAccepted {len(accepted)} -> {os.path.relpath(args.out, HERE)}"
          f" | Rejected {len(rejected)}"
          + (f" -> {os.path.relpath(args.rejected, HERE)}" if rejected else ""))


if __name__ == "__main__":
    main()
