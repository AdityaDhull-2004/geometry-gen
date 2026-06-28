"""Functional eval of the fine-tuned model's predictions. Run LOCALLY on the
Windows machine (where Playwright + the GeoGebra harness already work), after
copying predictions.jsonl back from the server.

Metric = the same gate the prototype will use in production: does the predicted
construction render as a valid, non-degenerate figure (post_filter)?

    python finetune/eval_predictions.py            # reads finetune/predictions.jsonl
"""
import os, io, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "datagen"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from post_filter import PostFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(HERE, "predictions.jsonl"))
    ap.add_argument("--trials", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(l) for l in io.open(args.pred, encoding="utf-8") if l.strip()]
    n = len(rows)
    parsed = renderable = 0
    fails = []

    with PostFilter() as pf:
        for i, r in enumerate(rows, 1):
            steps = r.get("predicted_construction_steps")
            if not steps:
                fails.append((r.get("id"), "no parseable JSON"))
                print(f"[{i:>2}] PARSE-FAIL  {r['problem_statement'][:55]}")
                continue
            parsed += 1
            try:
                ok, reasons = pf.accept(steps, trials=args.trials)
            except Exception as e:
                ok, reasons = False, [f"harness error: {type(e).__name__}: {e}"]
            if ok:
                renderable += 1
                print(f"[{i:>2}] OK          {r['problem_statement'][:55]}")
            else:
                fails.append((r.get("id"), reasons[0]))
                print(f"[{i:>2}] RENDER-FAIL {r['problem_statement'][:55]}\n        -> {reasons[0]}")

    print(f"\n=== functional eval over {n} held-out problems ===")
    print(f"parseable JSON     : {parsed}/{n}  ({100*parsed/n:.0f}%)")
    print(f"renders & valid    : {renderable}/{n}  ({100*renderable/n:.0f}%)")
    print(f"(renderable | parsed): {renderable}/{parsed}" if parsed else "")
    if fails:
        print("\nfailures:")
        for pid, why in fails:
            print(f"  id={pid}: {why}")


if __name__ == "__main__":
    main()
