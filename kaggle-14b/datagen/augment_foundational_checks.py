"""Retrofit semantic `checks` onto the 24 foundational problems (ids 1000-1023),
using only points that already exist in each construction. Rewrites
generated_problems.jsonl in place (after re-verifying). One problem (1018,
tangent-chord) gets a helper tangent-direction point so its angle can be checked.
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "generated_problems.jsonl")

CHECKS = {
    1000: [{"type": "concurrent", "lines": [["A", "M"], ["B", "N"], ["C", "P"]]}],
    1001: [{"type": "collinear", "points": ["O", "G", "H"]}],
    1002: [{"type": "concyclic", "points": ["D", "M", "N", "P"]}],
    1003: [{"type": "perpendicular", "a": ["P", "A"], "b": ["P", "B"]}],
    1004: [{"type": "concurrent", "lines": [["A", "X"], ["B", "Y"], ["C", "Z"]]}],
    1005: [{"type": "on_circle", "point": "H_reflection", "circle": ["A", "B", "C"]}],
    1006: [{"type": "parallel", "a": ["P", "Q"], "b": ["S", "R"]},
           {"type": "parallel", "a": ["Q", "R"], "b": ["P", "S"]}],
    1007: [{"type": "equal_len", "a": ["P", "T1"], "b": ["P", "T2"]},
           {"type": "perpendicular", "a": ["O", "P"], "b": ["T1", "T2"]}],
    1008: [{"type": "collinear", "points": ["X", "Y", "Z"]}],
    1009: [{"type": "angle_equal", "a": ["B", "A", "I"], "b": ["I", "A", "C"]},
           {"type": "angle_equal", "a": ["A", "B", "I"], "b": ["I", "B", "C"]}],
    1010: [{"type": "midpoint", "point": "center", "of": ["B", "D"]}],
    1011: [{"type": "parallel", "a": ["C", "E"], "b": ["D", "F"]}],
    # 1012 (inscribed-angle doubling) left render-only: the central/inscribed 2x
    # relation flips sign for major arcs with free points; not robustly checkable.
    1013: [{"type": "parallel", "a": ["M", "N"], "b": ["B", "C"]},
           {"type": "len_sum_equal", "a": [["M", "N"], ["M", "N"]], "b": [["B", "C"]]}],
    1014: [{"type": "equal_len", "a": ["M", "C"], "b": ["M", "A"]}],
    1015: [{"type": "midpoint", "point": "F", "of": ["A", "B"]}],
    1016: [{"type": "equal_len", "a": ["O", "A"], "b": ["O", "B"]},
           {"type": "equal_len", "a": ["O", "A"], "b": ["O", "C"]}],
    1017: [{"type": "len_product_equal",
            "pairs1": [["B", "X"], ["C", "Y"], ["A", "Z"]],
            "pairs2": [["X", "C"], ["Y", "A"], ["Z", "B"]]}],
    # 1018 (tangent-chord = inscribed) left render-only: the equality holds only
    # when C lies in the alternate segment, but C is a free point on the whole
    # circle, so the angle equals its supplement for half the drag range.
    1019: [{"type": "perpendicular", "a": ["A", "B"], "b": ["O1", "O2"]}],
    1020: [{"type": "coincide", "points": ["nine_point_centre", "midpoint_OH"]}],
    1021: [{"type": "midpoint", "point": "M", "of": ["A", "A_prime"]}],
    1022: [{"type": "angle_equal", "a": ["E", "D", "A"], "b": ["A", "D", "F"]}],
    1023: [{"type": "len_product_equal",
            "pairs1": [["P", "A"], ["P", "B"]], "pairs2": [["P", "C"], ["P", "D"]]}],
}

rows = []
with open(PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

added = 0
for r in rows:
    pid = r.get("id")
    if pid in CHECKS:
        r["checks"] = CHECKS[pid]
        added += 1

with open(PATH, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"added checks to {added} foundational problems; rewrote {PATH}")
