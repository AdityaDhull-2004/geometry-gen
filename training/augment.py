"""Meaning-preserving paraphrase augmentation for problem statements.

Multiplies training examples by restating each problem in equivalent ways (same
construction target), so the model generalizes across phrasings instead of
memorizing exact wordings. Constructions are NOT changed. Applied to the TRAIN
split only (eval stays original, to measure real performance honestly).
"""

# Each axis is a group of safe, meaning-preserving substitutions.
_AXES = [
    [("Prove that", "Show that"), ("prove that", "show that")],
    [("Prove ", "Show "), ("prove ", "show ")],
    [("Draw ", "Construct "), ("draw ", "construct ")],
    [("Determine ", "Find "), ("determine ", "find ")],
    [("centre", "center"), ("Centre", "Center")],          # British -> American
    [("Let ", "Consider "), ],                              # opener variant
    [("Suppose ", "Assume "), ],
]


def _apply(stmt, axis):
    v = stmt
    for pat, rep in axis:
        v = v.replace(pat, rep)
    return v


def paraphrase_variants(stmt, max_extra=3):
    """Return [original, + up to max_extra equivalent paraphrases]."""
    seen = [stmt]
    for axis in _AXES:                       # single-axis variants
        v = _apply(stmt, axis)
        if v != stmt and v not in seen:
            seen.append(v)
    combo = stmt                             # all axes at once
    for axis in _AXES:
        combo = _apply(combo, axis)
    if combo not in seen:
        seen.append(combo)
    return seen[: 1 + max_extra]


def expand(problems, max_extra=3):
    """Each paraphrase keeps the same construction_steps."""
    out = []
    for p in problems:
        for v in paraphrase_variants(p["problem_statement"], max_extra):
            q = dict(p)
            q["problem_statement"] = v
            out.append(q)
    return out


if __name__ == "__main__":
    demo = "Let ABC be a triangle with centre O. Prove that the medians are concurrent. Draw the incircle."
    for v in paraphrase_variants(demo):
        print("-", v)
