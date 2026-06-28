# Geometry Diagram Generator

Natural-language Olympiad geometry problem → ordered **GeoGebra construction JSON** →
interactive, draggable diagram. The repo holds the verified datasets, the
data-generation/verification pipeline, and the Kaggle fine-tuning setup.

## Layout

```
datasets/        the three verified datasets (+ a bank of extras)
  geometry_problems.jsonl        82  gold-standard problems
  generated_problems.jsonl       150 generated problems (audited & verified)
  new_geometry_problems.jsonl    60  new problems (audited & verified)
  bank_extra_problems.jsonl      46  extra audited problems (not yet merged)

pipeline/        data generation + verification + rendering
  validator.py            headless-GeoGebra renderer gate (Playwright)
  test_harness.html       GeoGebra engine harness used by the validator
  renderer.html           interactive renderer (auto-fixes index/vertex slips)
  checks.py               semantic theorem checks (collinear, concyclic, ...)
  audit_helpers.py        numeric claim verification (probe coords, test claim)
  visibility.py           "final" if named in statement, else "construction"
  post_filter.py          degeneracy/renderability post-filter
  dup_scan.py             cross-dataset near-duplicate scan
  verify_dataset.py       re-render + re-check an entire dataset
  validate_candidates.py  validate hand-authored candidates -> append
  validate_new.py         validate + append to new_geometry_problems.jsonl
  generate.py             API-based candidate generator (needs ANTHROPIC_API_KEY)

training/        Kaggle 14B QLoRA fine-tuning
  prepare_data.py         merge datasets -> chat-format train/eval (+augment)
  augment.py              meaning-preserving statement paraphrasing
  qlora_train_14b.py      Unsloth QLoRA on Qwen2.5-Coder-14B (Kaggle T4/P100)
  eval_predictions.py     score model predictions (parseable %, renders %)
  KAGGLE_SETUP.md         Kaggle playbook
  kaggle_testing_cells.txt / kaggle_training_files.txt

models/
  kaggle_14b_adapter/     current 14B LoRA adapter (weights, git-ignored)
```

## Common commands

```bash
# Re-render & re-verify a dataset (every step must build on every random trial)
python pipeline/verify_dataset.py datasets/new_geometry_problems.jsonl

# Scan a dataset for duplicates vs the others
python pipeline/dup_scan.py

# Build the fine-tuning corpus from all three datasets (82+150+60 = 292)
python training/prepare_data.py        # writes train.jsonl / eval.jsonl to training/

# Interactively view a construction: open pipeline/renderer.html and paste the JSON
```

All 292 dataset problems render cleanly in headless GeoGebra, have their stated
claim numerically verified, carry consistent `final`/`construction` visibility
labels, and contain no dead steps.

## Model & results

The model is **Qwen2.5-Coder-14B-Instruct**, fine-tuned with **QLoRA** (4-bit
base + LoRA adapters, r=32) on the 292-problem corpus (253 train → 850 augmented,
39 held out). On the **39 unseen** problems it scores **100% parseable JSON** and
**85% render-valid** — up ~14 points from an earlier 14B run on the un-cleaned
data (71%).

Full write-up — dataset construction, base-model choice, why 4-bit QLoRA,
training dynamics, and the held-out results with failure analysis — is in
**[`report/geom14b_report.pdf`](report/geom14b_report.pdf)**.

> The trained adapter (~550 MB) is **not** in this repo (over GitHub's file
> limit); it is hosted as a Kaggle Dataset. To reproduce: run
> `python training/prepare_data.py`, train with `training/qlora_train_14b.py`
> (or the cells in `training/kaggle_training_files.txt`), then evaluate with
> `python pipeline/score_predictions.py predictions.jsonl`.
