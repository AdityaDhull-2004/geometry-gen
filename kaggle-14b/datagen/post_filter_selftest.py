"""Demonstrate the post-generation filter: keeps a valid figure, drops broken
and degenerate ones, and handles raw (fenced) model text."""
import os, sys, io, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from post_filter import PostFilter

HERE = os.path.dirname(os.path.abspath(__file__))

# a real, valid construction from the dataset
good = json.loads(io.open(os.path.join(HERE, "generated_problems.jsonl"), encoding="utf-8").readline())

# BROKEN: a step references undefined objects -> fails to render
broken = {"problem_statement": "broken demo", "construction_steps": [
    {"name": "A", "type": "final", "class": "point", "command": "(0, 0)"},
    {"name": "B", "type": "final", "class": "point", "command": "(4, 0)"},
    {"name": "x", "type": "final", "class": "point", "command": "Intersect(nope, nada)"}]}

# DEGENERATE (collinear base triangle): A, B, C on a line
collinear = {"problem_statement": "collinear demo", "construction_steps": [
    {"name": "A", "type": "final", "class": "point", "command": "(0, 0)"},
    {"name": "B", "type": "final", "class": "point", "command": "(1, 1)"},
    {"name": "C", "type": "final", "class": "point", "command": "(2, 2)"},
    {"name": "s1", "type": "final", "class": "segment", "command": "Segment(A, B)"},
    {"name": "s2", "type": "final", "class": "segment", "command": "Segment(B, C)"}]}

# DEGENERATE (collapsed): every point at the same location
collapsed = {"problem_statement": "collapsed demo", "construction_steps": [
    {"name": "A", "type": "final", "class": "point", "command": "(0, 0)"},
    {"name": "B", "type": "final", "class": "point", "command": "(0, 0)"},
    {"name": "C", "type": "final", "class": "point", "command": "(0, 0)"},
    {"name": "s1", "type": "final", "class": "segment", "command": "Segment(A, B)"}]}

# raw model-style text with a code fence around valid JSON
raw_text = "Here is the construction:\n```json\n" + json.dumps(good) + "\n```\nHope it helps!"

cases = [("good (dataset id 1)", good),
         ("raw fenced text", raw_text),
         ("BROKEN (bad intersect)", broken),
         ("DEGENERATE collinear ABC", collinear),
         ("DEGENERATE collapsed", collapsed)]

with PostFilter() as pf:
    for name, obj in cases:
        ok, reasons = pf.accept(obj, trials=5)
        verdict = "KEEP" if ok else "DROP"
        why = "" if ok else f"  ({reasons[0]})"
        print(f"{verdict:4}  {name}{why}")
