"""Score model predictions on the held-out eval set.

Each predicted construction is re-rendered in headless GeoGebra. Empty/unparsed
predictions (no construction_steps) count as failures, so this does NOT inflate
the score by treating a parse-fail as a clean render.

    python pipeline/score_predictions.py predictions.jsonl [--trials 6]

predictions.jsonl: one object per line with at least
    {"id":..., "problem_statement":..., "construction_steps":[...]}
(a parse-failed prediction should have construction_steps = []).
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from validator import GeoGebraValidator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    n = len(rows)
    parse_ok = render_ok = 0
    fails = []

    with GeoGebraValidator() as v:
        for r in rows:
            steps = r.get("construction_steps") or []
            if not steps:
                fails.append((r.get("id"), "PARSE", "empty / unparsed JSON"))
                continue
            parse_ok += 1
            ok, info = v.validate_robust(steps, trials=a.trials)
            if ok:
                render_ok += 1
            else:
                fails.append((r.get("id"), "RENDER", info.get("sample_failure")))

    print(f"\n=== {os.path.basename(a.path)}  ({n} eval problems) ===")
    print(f"parseable JSON : {parse_ok}/{n} = {100*parse_ok/n:.0f}%")
    print(f"render-valid   : {render_ok}/{n} = {100*render_ok/n:.0f}%   (of all {n})")
    if parse_ok:
        print(f"render-valid   : {render_ok}/{parse_ok} = {100*render_ok/parse_ok:.0f}%   (of parseable)")
    if fails:
        print(f"\nfailures ({len(fails)}):")
        for pid, kind, why in fails:
            print(f"  id={pid}  {kind:<6} -> {why}")


if __name__ == "__main__":
    main()
