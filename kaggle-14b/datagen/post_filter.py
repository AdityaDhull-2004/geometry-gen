"""Post-generation quality filter for model-produced constructions.

At inference time the fine-tuned model emits a construction for a *new* problem,
so there is no ground-truth `checks` field to assert a specific theorem. This
filter instead renders the construction in the real (headless) GeoGebra engine
and applies GENERIC degeneracy heuristics to weed out:

  * BROKEN figures   - some step fails to produce an object (bad command / index)
  * DEGENERATE figures - render fine but are geometrically useless:
        - a point flew to infinity / non-finite / absurd coordinate
        - the whole figure collapsed to ~one location
        - too many distinct-named points coincide (the classic "collapse to
          anchor" / wrong-intersection-index bug)
        - the base triangle ABC (when present) is (near) collinear

Each construction is rendered `trials` times with fresh random parameters; it is
accepted only if EVERY trial renders and is non-degenerate (a figure that is fine
for one parameter value but breaks for another is not robust enough to keep).

Usage (inference):
    from post_filter import PostFilter
    with PostFilter() as pf:
        ok, reasons = pf.accept(model_output_text)   # raw text or dict or steps
        # -> (True, []) keep ;  (False, ["...why..."]) drop / send to repair

CLI (batch a jsonl of generated candidates):
    python datagen/post_filter.py candidates.jsonl --trials 5
"""
import os, sys, json, math, argparse, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validator import GeoGebraValidator


# ── tunable thresholds ───────────────────────────────────────────────────────
class FilterConfig:
    max_coord = 1e5          # a point beyond this is treated as "at infinity"
    min_extent = 0.5         # bounding-box diagonal below this => collapsed
    coincide_rel = 1e-4      # two points closer than this * extent are "the same"
    min_distinct_frac = 0.5  # at least this fraction of points must be distinct
    keep_threshold = 0.6     # keep if at least this fraction of trials is clean
    # NB: we deliberately do NOT assume A,B,C form a triangle - in many valid
    # configurations (Pappus, circle-intersection points, ...) they are collinear
    # or unrelated by design, so a "degenerate ABC" test gives false positives.


# ── JSON extraction from a noisy model reply (balanced-brace, fence-tolerant) ─
def _repair_json(s):
    """Fix the common syntax slips small models make so near-valid JSON parses:
    missing commas between objects / between adjacent quoted tokens (the model
    drops the comma after a value, e.g. `"C" "type"`), and trailing commas."""
    s = re.sub(r'}\s*{', '}, {', s)            # missing comma between objects
    s = re.sub(r'"\s+"', '", "', s)            # missing comma: "value" "key" (valid
                                               #   JSON never has quote-ws-quote)
    s = re.sub(r',\s*([}\]])', r'\1', s)       # trailing comma before } or ]
    return s


def extract_json(text):
    if isinstance(text, (dict, list)):
        return text
    s = str(text)
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
                        cand = s[start:i + 1]
                        try:
                            return json.loads(cand)
                        except json.JSONDecodeError:
                            try:
                                return json.loads(_repair_json(cand))  # lenient retry
                            except json.JSONDecodeError:
                                break
        start = s.find("{", start + 1)
    return None


def _steps_of(obj):
    """Accept a problem dict, a bare steps list, or raw text -> steps list or None."""
    if isinstance(obj, str):
        obj = extract_json(obj)
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get("construction_steps")
    return None


def _hyp(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def degeneracy_reasons(coords, cfg=FilterConfig):
    """Generic degeneracy heuristics on one render's point coordinates."""
    reasons = []
    pts = {k: v for k, v in coords.items()}
    # 1. finite & in-bounds
    for n, (x, y) in pts.items():
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > cfg.max_coord or abs(y) > cfg.max_coord:
            reasons.append(f"point {n} at infinity / out of bounds ({x:.3g}, {y:.3g})")
    vals = list(pts.values())
    if len(vals) < 2:
        return reasons
    xs = [v[0] for v in vals]; ys = [v[1] for v in vals]
    extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    # 2. whole figure collapsed
    if extent < cfg.min_extent:
        reasons.append(f"figure collapsed (extent {extent:.3g})")
        return reasons
    # 3. too many coincident points
    tol = cfg.coincide_rel * extent
    clusters = []
    for v in vals:
        if not any(_hyp(v, c) < tol for c in clusters):
            clusters.append(v)
    if len(clusters) < max(3, math.ceil(cfg.min_distinct_frac * len(vals))):
        reasons.append(f"too many coincident points ({len(clusters)} distinct of {len(vals)})")
    return reasons


class PostFilter:
    def __init__(self, cfg=FilterConfig, validator=None):
        self.cfg = cfg
        self._own = validator is None
        self._v = validator or GeoGebraValidator()

    def __enter__(self):
        if self._own:
            self._v.start()
        return self

    def __exit__(self, *exc):
        if self._own:
            self._v.stop()

    def accept(self, obj, trials: int = 5):
        """Return (accept: bool, reasons: list[str]). Render `trials` times and
        keep the construction if at least `keep_threshold` of the trials are
        clean (render + non-degenerate). Majority voting tolerates measure-zero
        parameter degeneracies (e.g. two lines momentarily parallel) while still
        rejecting constructions that are broken or degenerate most of the time."""
        steps = _steps_of(obj)
        if not steps:
            return False, ["could not parse a construction (no construction_steps)"]
        clean = 0
        last_reason = None
        for _ in range(trials):
            failed, coords = self._v.probe(steps)           # fresh random t each call
            if failed:
                last_reason = f"broken: step failed to render -> {failed[0]}"
                continue
            reasons = degeneracy_reasons(coords, self.cfg)
            if reasons:
                last_reason = "degenerate: " + "; ".join(reasons)
                continue
            clean += 1
        if clean / trials >= self.cfg.keep_threshold:
            return True, []
        return False, [f"clean only {clean}/{trials} trials; e.g. {last_reason}"]


# ── CLI: filter a jsonl of generated candidates ──────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default=None, help="write kept problems here (jsonl)")
    args = ap.parse_args()

    rows = []
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    kept, dropped = [], 0
    with PostFilter() as pf:
        for i, r in enumerate(rows, 1):
            ok, reasons = pf.accept(r, trials=args.trials)
            stmt = (r.get("problem_statement", "") if isinstance(r, dict) else "")[:60]
            if ok:
                kept.append(r)
                print(f"  [{i:>3}] KEEP  {stmt}")
            else:
                dropped += 1
                print(f"  [{i:>3}] DROP  {stmt}\n          -> {reasons[0]}")

    if args.out and kept:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nkept {len(kept)} / {len(rows)}  (dropped {dropped})"
          + (f"  -> {args.out}" if args.out and kept else ""))


if __name__ == "__main__":
    main()
