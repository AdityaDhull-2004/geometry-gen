# geometry-gen

Turn a natural-language geometry problem into an **ordered GeoGebra construction**
(JSON) that renders as an interactive, draggable diagram. A fine-tuned code LLM
produces the construction; a headless GeoGebra validator gates the output so only
renderable, non-degenerate figures are kept.

```
problem (text) ──▶ fine-tuned model ──▶ {"construction_steps":[…]}  ──▶ GeoGebra renders an interactive diagram
                                              │
                                   headless-GeoGebra validator (renderable? non-degenerate?)
```

## Repository layout

This repo holds **two fully self-contained training setups** — pick the folder for
the approach you want; each runs on its own.

| Folder | Approach | Base model | Hardware |
|---|---|---|---|
| [`lab-server-3b/`](lab-server-3b/) | QLoRA on a local lab GPU (over SSH) | Qwen2.5-Coder-3B-Instruct | 6 GB RTX 3050 |
| [`kaggle-14b/`](kaggle-14b/) | QLoRA on a free Kaggle GPU | Qwen2.5-Coder-14B-Instruct | 16 GB T4 |

Each folder contains the same shared core so it stands alone:
- `datagen/` — data-generation + GeoGebra validation tooling (`validator.py`,
  `checks.py`, `post_filter.py`, `visibility.py`, …) and the 150 generated problems.
- `geometry_problems.jsonl` — the 82 hand-curated gold problems.
- `renderer.html`, `test_harness.html` — interactive renderer and the headless harness.
- `finetune/` — the data-prep, training, inference and evaluation scripts for that approach.

`report.tex` / `report.pdf` (root) is the overall project report; the 14B-specific
training report is in `kaggle-14b/report/`.

## Dataset
232 distinct problems (82 gold + 150 generated and GeoGebra-verified), split
deterministically into 201 train / 31 held-out eval. The Kaggle (14B) setup
additionally paraphrase-augments the training split to 658 examples; the lab (3B)
setup uses the 201 unaugmented examples. Visibility rule: every object named in the
statement is marked `final` (shown).

## How to run

**Lab server (3B):** see [`lab-server-3b/finetune/SETUP_SSH.md`](lab-server-3b/finetune/SETUP_SSH.md).
QLoRA via Unsloth on a single GPU; export a GGUF and serve locally with Ollama.

**Kaggle (14B):** see [`kaggle-14b/finetune/KAGGLE_SETUP.md`](kaggle-14b/finetune/KAGGLE_SETUP.md).
Upload `train.jsonl`, `eval.jsonl`, `qlora_train_14b.py`; train; then test new problems
with `kaggle cells for testing new problems.txt`.

## Results (31 held-out problems, identical split)

| Model | Parseable JSON | Renders & valid |
|---|---|---|
| Qwen2.5-Coder-3B  | 100% | 61% raw / 65% with repair loop |
| Qwen2.5-Coder-14B | 100% | **71% raw (no repair)** |

The 14B fixes most of the 3B's command errors (e.g. correct `Incircle(A,B,C)`);
remaining failures are command-name hallucinations and missing intersection indices,
which the repair loop and a command-name hint are expected to reduce further.

## Note on model weights
Trained LoRA adapters and GGUF/`.safetensors` files are **not** committed (they are
large binaries and are `.gitignore`'d). Re-create them by running the training scripts.
