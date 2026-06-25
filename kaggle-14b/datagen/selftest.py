"""Exercise the whole pipeline EXCEPT the Claude call (no API key needed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from generate import load_gold, build_system, extract_json, FEWSHOT_IDS
from validator import GeoGebraValidator

# 1. gold loading + prompt build (the cached prefix)
gold = load_gold(FEWSHOT_IDS)
system = build_system(gold)
prefix_chars = len(system[0]["text"])
print(f"few-shot gold loaded: {len(gold)}  | cached-prefix chars: {prefix_chars} (~{prefix_chars//4} tokens)")
print("cache_control on prefix:", system[0].get("cache_control"))

# 2. JSON extraction from a noisy model reply
sample_reply = '''Sure, here is the problem:
```json
{"id": 0, "problem_statement": "demo", "construction_steps": [
  {"name": "A", "type": "final", "class": "point", "command": "(0, 0)"},
  {"name": "B", "type": "final", "class": "point", "command": "(4, 0)"},
  {"name": "s", "type": "final", "class": "segment", "command": "Segment(A, B)"}
]}
```
Hope this helps!'''
parsed = extract_json(sample_reply)
print("extract_json ok:", bool(parsed and parsed.get("construction_steps")))

# 3. full validator round-trip on a real gold construction (proves generate->validate loop)
if gold:
    with GeoGebraValidator() as v:
        ok, info = v.validate_robust(gold[0]["construction_steps"], trials=3)
        print(f"validator round-trip on gold id={gold[0]['id']}: ok={ok} {info}")
        ok2, info2 = v.validate_robust(parsed["construction_steps"], trials=3)
        print(f"validator on extracted demo: ok={ok2} {info2}")
print("\nplumbing OK — only the Claude call remains (needs ANTHROPIC_API_KEY).")
