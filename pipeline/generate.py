"""Data-generation pipeline: Claude proposes new Olympiad-style geometry problems
(statement + GeoGebra construction), the headless-GeoGebra validator gates every
candidate, and only renderable ones are written to the output dataset.

Design (per the claude-api skill):
  * model claude-opus-4-8, adaptive thinking
  * PROMPT CACHING on the stable prefix (schema + GeoGebra rules + few-shot gold
    examples) so the ~Nk-token preamble is paid once and re-read cheaply on every
    call. Only the per-call theme/nonce varies, and it goes AFTER the cache point.
  * validator-gated, with one automatic repair attempt on a failed render.

Run:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    python datagen/generate.py --n 20 --out datagen/generated_problems.jsonl
"""
import os, sys, json, re, time, argparse, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validator import GeoGebraValidator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_JSONL = os.path.join(os.path.dirname(HERE), "datasets", "geometry_problems.jsonl")

MODEL = "claude-opus-4-8"

# Fixed, diverse few-shot ids → keeps the cached prefix byte-stable across calls.
FEWSHOT_IDS = [1, 2, 12, 17, 18, 35, 42, 48, 64, 70]

THEMES = [
    "a triangle and its incircle touching the three sides",
    "two circles meeting at two points, with a common tangent or the radical axis",
    "a cyclic quadrilateral and a property of its diagonals or centroids",
    "the circumcircle of a triangle and the midpoint of an arc",
    "a point moving on a segment or circle that generates a locus",
    "feet of perpendiculars from a point to the sides of a triangle (a pedal triangle)",
    "tangent lines from an external point to a circle and the points of tangency",
    "midpoints of the sides, the medial triangle, or the nine-point circle",
    "reflections of a point or a line across the sides of a triangle",
    "an equilateral or isosceles triangle erected on a side of a figure",
    "the orthocenter and the three altitudes of an acute triangle",
    "angle bisectors meeting at the incenter, or an external bisector and an excenter",
    "a square or rectangle built on a side of a triangle or segment",
    "power of a point, intersecting chords, or a tangent-secant configuration",
]

RULES = r"""You generate ONE original, non-trivial plane-geometry problem at a time, in the
exact style of the GOLD EXAMPLES below, as a single JSON object.

OUTPUT SCHEMA (output ONLY this JSON object, no prose, no markdown fences):
{
  "id": 0,
  "problem_statement": "<a self-contained Olympiad-style statement>",
  "construction_steps": [
    {"name": "<unique id>", "type": "final"|"construction", "class": "point|line|segment|circle|ray|conic|number|angle|vector|list|locus|polygon", "label": "<optional display label>", "command": "<native GeoGebra command referencing earlier step names>"}
  ]
}
Rules of the schema: steps run top-to-bottom; a command may only reference names
defined in earlier steps. `type:"construction"` objects are hidden helpers;
`type:"final"` objects are shown. `label` is optional (used for a custom caption).

The construction is rendered in real GeoGebra and is REJECTED unless every step
produces an object that exists. Follow these hard-won GeoGebra rules to stay valid:

GENERAL
- Use only real GeoGebra commands: Point, Midpoint, Segment, Line, Ray, Vector,
  Circle, Semicircle, Arc, Intersect, PerpendicularLine, PerpendicularBisector,
  AngleBisector, Tangent, Reflect, Rotate, Translate, Center, Radius, Distance,
  Angle, Polygon, Incircle, UnitVector, UnitPerpendicularVector, Locus, etc.
- A free vertex is a coordinate literal, e.g. "(0, 5)". Its `name` MUST start with
  an UPPERCASE letter (e.g. A, P1, O2) — a lowercase name with a bare coordinate
  becomes a VECTOR (an arrow), not a point.
- Never name an object `gamma` (reserved = the gamma function). Use `Gamma`, `omega`, etc.

POINTS ON PATHS (so they stay draggable / non-degenerate)
- A free point ON a bounded path uses the 1-arg form: Point(seg_A_B) or Point(circle).
  The renderer auto-spreads multiple such points so they don't coincide.
- Do NOT put a free point on an INFINITE line with the bare form Point(line) — it
  collapses onto the line's anchor and often coincides with another point. If you
  need an arbitrary point on an infinite line, either build a bounded Segment along
  it and use Point(seg), or define a separate free coordinate point and draw the line
  through it.

CIRCLES / CENTERS / INCIRCLE
- Circumcircle of A,B,C: Circle(A, B, C). Its center: Center(Circle(A, B, C)).
- Incircle: Incircle(A, B, C); incenter: Center(Incircle(A,B,C)); the touch point on
  side BC: Intersect(Incircle(A,B,C), Segment(B,C)). Do NOT build the incenter from
  angle bisectors and perpendicular feet — it is long and error-prone.
- A circle through a center O and a point P: Circle(O, P).

INTERSECTIONS
- Intersect(obj1, obj2, n) returns the n-th intersection (n = 1 or 2). When a line/
  circle already passes through a KNOWN point (e.g. B), one index returns B itself —
  choose the index that yields the NEW point so you don't get a degenerate result.
- Choose coordinates so required intersections actually exist: two "lines that meet
  at a point" must not be parallel (avoid accidental parallelograms); two circles
  that "intersect" must actually overlap.

TANGENTS FROM AN EXTERNAL POINT
- Tangent(P, circle) yields TWO lines. To name them, wrap in a list:
  tan = {Tangent(P, omega)}; then line1 = Element(tan, 1), line2 = Element(tan, 2);
  the touch points are Intersect(omega, line1) and Intersect(omega, line2).

QUALITY
- Pick concrete numeric coordinates for the free points that yield a clean, acute,
  general (non-degenerate, non-isosceles unless required) configuration.
- The statement and the construction must match: every named point in the statement
  appears as a step, and the construction realizes the stated constraints structurally
  (e.g. "D on AB with AD=AE" must be enforced, not just placed by eye).
- Make it genuinely new — do not copy a gold example.
"""


def load_gold(ids):
    by_id = {}
    try:
        with open(GOLD_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    by_id[o["id"]] = o
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    picked = [by_id[i] for i in ids if i in by_id]
    if not picked:  # fall back to whatever exists
        picked = list(by_id.values())[:8]
    return picked


def build_system(gold):
    blocks = "\n\n".join(json.dumps(g, ensure_ascii=False) for g in gold)
    text = RULES + "\n\nGOLD EXAMPLES:\n" + blocks
    # one cached block — stable across every generation call
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def extract_json(text):
    """Pull the first balanced {...} object out of the model's reply."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s).rsplit("```", 1)[0]
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
        else:
            if c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        return None
    return None


def call_model(client, system, user_text):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return text, resp.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="how many problems to accept")
    ap.add_argument("--out", default=os.path.join(HERE, "generated_problems.jsonl"))
    ap.add_argument("--trials", type=int, default=4, help="validator render trials")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="hard cap on generation attempts (0 = n*4)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY first.  e.g.  $env:ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    gold = load_gold(FEWSHOT_IDS)
    print(f"loaded {len(gold)} few-shot gold examples; model={MODEL}")
    system = build_system(gold)

    # next id continues after the gold set
    next_id = max([g["id"] for g in gold] + [0]) + 1000  # 1000+ marks generated data

    accepted = 0
    attempts = 0
    cap = args.max_attempts or args.n * 4
    tok_in = tok_cache_read = tok_cache_write = tok_out = 0

    with GeoGebraValidator() as v, open(args.out, "a", encoding="utf-8") as out:
        while accepted < args.n and attempts < cap:
            attempts += 1
            theme = random.choice(THEMES)
            nonce = random.randint(10000, 99999)
            user = (f"Generate ONE new, original plane-geometry problem and its construction.\n"
                    f"Loose inspiration (not a constraint): {theme}.\n"
                    f"Variation token: {nonce}.\nOutput ONLY the JSON object.")
            try:
                text, usage = call_model(client, system, user)
            except Exception as e:
                print(f"[attempt {attempts}] API error: {e}")
                time.sleep(2)
                continue
            tok_in += usage.input_tokens
            tok_cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
            tok_cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0
            tok_out += usage.output_tokens

            cand = extract_json(text)
            if not cand or "construction_steps" not in cand:
                print(f"[attempt {attempts}] unparseable / no construction_steps")
                continue

            ok, info = v.validate_robust(cand["construction_steps"], trials=args.trials)
            if not ok:
                # one repair attempt
                fail = info.get("sample_failure") or ["(varies across trials)"]
                repair = (user + "\n\nYour previous attempt did not render. These steps "
                          "failed (object did not exist):\n" + "\n".join(f"- {x}" for x in fail[:6]) +
                          "\n\nHere was your attempt:\n" + json.dumps(cand, ensure_ascii=False) +
                          "\n\nReturn a CORRECTED JSON object (same schema) that renders. Output ONLY JSON.")
                try:
                    text2, usage2 = call_model(client, system, repair)
                    tok_in += usage2.input_tokens
                    tok_cache_read += getattr(usage2, "cache_read_input_tokens", 0) or 0
                    tok_cache_write += getattr(usage2, "cache_creation_input_tokens", 0) or 0
                    tok_out += usage2.output_tokens
                    cand2 = extract_json(text2)
                    if cand2 and "construction_steps" in cand2:
                        ok, info = v.validate_robust(cand2["construction_steps"], trials=args.trials)
                        if ok:
                            cand = cand2
                except Exception as e:
                    print(f"[attempt {attempts}] repair API error: {e}")

            if ok:
                cand["id"] = next_id; next_id += 1
                out.write(json.dumps(cand, ensure_ascii=False) + "\n")
                out.flush()
                accepted += 1
                print(f"[attempt {attempts}] ACCEPTED #{accepted}/{args.n}  id={cand['id']}  "
                      f"steps={len(cand['construction_steps'])}  :: {cand['problem_statement'][:70]}")
            else:
                print(f"[attempt {attempts}] rejected ({info.get('clean')}/{info.get('trials')} clean)")

    print("\n" + "=" * 64)
    print(f"accepted {accepted} / {attempts} attempts  (yield {accepted/max(attempts,1):.0%})")
    print(f"tokens: input(uncached)={tok_in}  cache_read={tok_cache_read}  "
          f"cache_write={tok_cache_write}  output={tok_out}")
    if tok_cache_read > 0:
        print("prompt caching is working (cache_read > 0).")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
