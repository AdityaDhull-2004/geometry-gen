"""Visibility rule: every entity NAMED in the problem statement should be SHOWN.

Promotes any construction step that the statement references to type="final",
leaving genuine scaffolding hidden. Used two ways:
  - inference post-process (predict_local.py) so generated JSON shows the right
    objects regardless of how the model labeled them;
  - to relabel the training data consistently before a retrain, so the model
    LEARNS the distinction (then raw outputs need less fixing).

Heuristic, not perfect: it keys on point letters and object keywords found in the
statement. Misses edge cases (unicode subscripts, curves named descriptively);
the renderer's "show helper objects" toggle is the backstop.
"""
import re

# if one of these words is in the statement, non-point objects whose name/label
# references it should be shown (the statement is talking about that object).
OBJECT_KEYWORDS = [
    "incircle", "circumcircle", "excircle", "nine-point", "nine point", "circle",
    "median", "altitude", "bisector", "perpendicular", "tangent", "diameter",
    "chord", "midpoint", "centroid", "orthocent", "incent", "circumcent",
    "excent", "midsegment", "diagonal", "axis", "locus", "radius",
]


def statement_point_names(stmt):
    """Capital-letter point names the statement mentions. Word boundaries avoid
    matching sentence-initial capitals like 'Prove'/'Let' (those aren't 1-letter
    words). Runs like 'ABC' are split into A, B, C."""
    pts = set()
    for run in re.findall(r"\b[A-Z]{2,}\b", stmt):     # ABCD -> A,B,C,D
        pts.update(run)
    pts.update(re.findall(r"\b[A-Z]\b", stmt))          # standalone M, O, H, I, P
    pts.update(re.findall(r"\b[A-Z][0-9]\b", stmt))     # A1, B2
    pts.update(re.findall(r"\b[A-Z]'", stmt))           # A', M'
    return pts


def apply_visibility_rule(statement, steps):
    """Return a copy of steps with statement-referenced objects set type='final'."""
    stmt = statement or ""
    low = stmt.lower()
    pts = statement_point_names(stmt)
    kws = [k for k in OBJECT_KEYWORDS if k in low]
    out = []
    for st in steps:
        st = dict(st)
        nm = (st.get("name") or "").strip()
        lab = (st.get("label") or "").strip()
        show = False
        if st.get("class") == "point":
            if lab in pts or nm in pts:
                show = True
        else:
            blob = (nm + " " + lab).lower()
            if any(k in blob for k in kws):
                show = True
        if show and st.get("type") == "construction":
            st["type"] = "final"
        out.append(st)
    return out


if __name__ == "__main__":
    demo = "Let ABC be a triangle with incentre I and circumcircle. M is the midpoint of BC."
    steps = [
        {"name": "A", "class": "point", "type": "final"},
        {"name": "I", "label": "I", "class": "point", "type": "construction"},
        {"name": "M", "label": "M", "class": "point", "type": "construction"},
        {"name": "bisector_A", "class": "line", "type": "construction"},
        {"name": "incircle", "class": "circle", "type": "construction"},
        {"name": "helper_perp", "class": "line", "type": "construction"},
    ]
    for s in apply_visibility_rule(demo, steps):
        print(f"{s['name']:<14} -> {s['type']}")
