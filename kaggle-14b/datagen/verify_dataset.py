"""Read-only audit of generated_problems.jsonl: re-render every problem and
re-check every semantic assertion from scratch. Writes nothing.

    python datagen/verify_dataset.py [path] [--trials 8]

Reports, per problem: render-clean? checks-pass? and a summary of how many
problems carry semantic checks vs renderability only.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from validator import GeoGebraValidator
from checks import evaluate as eval_checks

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=os.path.join(HERE, "generated_problems.jsonl"))
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()

    rows = []
    with open(args.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    render_ok = checks_ok = with_checks = render_only = 0
    failures = []

    with GeoGebraValidator() as v:
        for prob in rows:
            steps = prob.get("construction_steps", [])
            checks = prob.get("checks", [])
            pid = prob.get("id")

            ok, info = v.validate_robust(steps, trials=args.trials)
            if ok:
                render_ok += 1
            else:
                failures.append((pid, "RENDER", info.get("sample_failure")))
                continue

            if checks:
                with_checks += 1
                bad = None
                for _ in range(args.trials):
                    failed, coords = v.probe(steps)
                    if failed:
                        bad = f"probe failed: {failed[0]}"; break
                    cok, reason = eval_checks(checks, coords)
                    if not cok:
                        bad = reason; break
                if bad:
                    failures.append((pid, "CHECK", bad))
                else:
                    checks_ok += 1
            else:
                render_only += 1

    print(f"\n=== dataset audit: {os.path.basename(args.path)} ({n} problems, {args.trials} trials each) ===")
    print(f"render clean        : {render_ok}/{n}")
    print(f"carry semantic checks: {with_checks}  (all checks pass: {checks_ok})")
    print(f"renderability only   : {render_only}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for pid, kind, why in failures:
            print(f"  id={pid}  {kind}  -> {why}")
    else:
        print("\nNo failures: every problem renders and every semantic check holds.")


if __name__ == "__main__":
    main()
