# Train Qwen2.5-Coder-14B on Kaggle (free GPU)

Goal: fine-tune the 14B on the augmented dataset (658 train / 31 eval) and export a
q4 GGUF. Kaggle gives a free 16 GB GPU (T4 x2 or P100) for up to ~12 h/session.

## 0. Prereqs
- Kaggle account, **phone-verified** (required to enable GPU + Internet).
- Files to upload: `train.jsonl`, `eval.jsonl`, `qlora_train_14b.py`
  (all in `geometry-gen/finetune/`). For functional eval later you also want
  `eval_problems.jsonl`.

## 1. Put the files on Kaggle (as a Dataset)
1. kaggle.com → **Datasets → New Dataset**.
2. Upload `train.jsonl`, `eval.jsonl`, `qlora_train_14b.py` (drag them in).
3. Name it e.g. `geom-ft`, create. It will live at `/kaggle/input/geom-ft/`.

## 2. New Notebook with GPU + Internet
1. **Code → New Notebook**.
2. Right panel ⚙️ → **Accelerator = GPU T4 x2** (or P100), **Internet = On**.
3. **Add Input** → your `geom-ft` dataset.

## 3. Notebook cells

**Cell 1 — install (Kaggle ships torch; add the trainer stack):**
```python
!pip install -q unsloth "trl>=0.9" datasets
```

**Cell 2 — stage files into the working dir:**
```python
import os, shutil
SRC = "/kaggle/input/geom-ft"               # adjust if you named it differently
os.makedirs("/kaggle/working", exist_ok=True)
for f in ["train.jsonl", "eval.jsonl", "qlora_train_14b.py"]:
    shutil.copy(f"{SRC}/{f}", f"/kaggle/working/{f}")
os.chdir("/kaggle/working")
print(os.listdir())
```

**Cell 3 — train + export GGUF (~2–3 h on a T4):**
```python
!python qlora_train_14b.py
```
The 14B 4-bit model (~9.5 GB) downloads on first run, then it trains and writes:
- LoRA adapters → `/kaggle/working/qwen25coder14b-geom-lora/`
- q4 GGUF → `/kaggle/working/geom14b-gguf/*.Q4_K_M.gguf` (~9 GB)

## 4. Get the GGUF out
- Interactive session: right panel → **Output** / the file browser → download
  `geom14b-gguf/*.Q4_K_M.gguf`.
- Or **Save Version** (commit) to run it headless and persist outputs for download.

## 5. Run / serve the model
The 14B q4 is ~9 GB → needs ~12 GB RAM to run on CPU (slow) via Ollama, the same
flow as the 3B:
```
# laptop, if it has the RAM:
copy the GGUF to C:\ollama-models\geom14b.Q4_K_M.gguf
edit geom.Modelfile FROM -> that path ; ollama create geom14b -f geom.Modelfile
```
For a **free online "API"**, the simplest is an on-demand **Kaggle/Colab inference
notebook** (load the GGUF with llama-cpp-python or the adapter with Unsloth and
expose a cell/function). A persistent always-on free API for a 14B isn't realistic;
HuggingFace free serverless won't host an arbitrary 14B reliably.

## 6. Evaluate it (functional, like before)
Bring the GGUF to your laptop, point `predict_local.py` at a `geom14b` Ollama model,
and run your held-out problems through it — or add a Kaggle eval cell that installs
`playwright`+`chromium` and runs `post_filter` on generated constructions.

## Notes / knobs (`qlora_train_14b.py`)
- 14B QLoRA fits 16 GB at `BATCH=1`, `MAX_SEQ_LEN=1536`, gradient checkpointing.
  If you hit CUDA OOM: drop `MAX_SEQ_LEN` to 1024.
- `EPOCHS=4`, `LORA_R=32` — start here; raise epochs to 5 if the model underfits.
- In-training eval is OFF (full-vocab fp32 logits OOM the 14B on 16 GB); quality is
  judged by functional eval after, which is the metric that matters anyway.
- Kaggle GPU sessions cap at ~12 h and weekly ~30 h — one run fits comfortably.
