# Data-generation pipeline

Generates new Olympiad-style geometry problems (statement + GeoGebra construction)
in your schema, with **every candidate gated by the real GeoGebra engine** so only
renderable problems are kept.

```
problem theme ─▶ Claude (opus-4-8, few-shot from the 82 gold) ─▶ candidate JSON
                                                                     │
                              headless GeoGebra validator  ◀─────────┘
                                     │ renders?
                          ┌──────────┴───────────┐
                       fails                    passes
                          │                       │
                  one repair attempt        append to
                  (feed error back)      generated_problems.jsonl
```

## Files
- `validator.py` — `GeoGebraValidator`: reusable headless-GeoGebra checker (reuses
  `../test_harness.html`). `validate_robust(steps, trials=4)` accepts only if every
  trial renders cleanly. **The linchpin.**
- `generate.py` — the orchestrator: Claude proposes problems, the validator filters,
  passing ones are written out.
- `selftest.py` — exercises everything except the Claude call (no API key needed).

## Run
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python datagen/generate.py --n 20 --out datagen/generated_problems.jsonl
```
Options: `--n` (problems to accept), `--trials` (validator render trials, default 4),
`--max-attempts` (hard cap, default `n*4`).

## How it stays correct / cheap
- **Validator-gated:** a problem is kept only if all `--trials` random-`t` renders are
  clean — same engine and check used for the 82 gold problems.
- **Few-shot:** the 82 gold problems anchor schema and style (a fixed diverse subset
  is sent each call).
- **Prompt caching:** the schema + GeoGebra rules + few-shot examples (~6k tokens) sit
  in a cached system prefix; only the per-call theme/nonce varies, so repeated calls
  re-read the prefix at ~0.1× cost. The run prints `cache_read` to confirm hits.
- **GeoGebra gotchas baked into the prompt** (uppercase-named coordinate points,
  `Incircle`/`Circle(A,B,C)`/`Center`, intersection-index choice, tangent-from-point
  via a `{Tangent(...)}` list, no `Point(<infinite line>)`, no reserved name `gamma`).

## Known limitation
The validator checks **renderability**, not semantic truth — it does not verify the
problem statement matches the construction or that the claimed property holds. That's
the next layer: add a machine-checkable `goal` field (e.g. `Concyclic`, `Concurrent`,
`AreParallel`) and assert it in GeoGebra. Generated problems land at `id >= 1000`.
