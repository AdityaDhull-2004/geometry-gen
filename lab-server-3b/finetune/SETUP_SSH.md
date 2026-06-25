# Lab-server setup (QLoRA fine-tune of Qwen2.5-Coder-3B-Instruct)

Target machine: **RTX 3050, 6 GB VRAM**, driver CUDA 13.0, conda available, 398 GB free.
Model = **Qwen2.5-Coder-3B** (7B does not fit 6 GB). Plan: **train + generate on the
box**, **score on Windows** (where the GeoGebra/Playwright harness already runs).

> Free up VRAM before training: this box's Firefox / VS Code / GNOME use ~0.5 GB of
> the 6 GB. Close the Firefox and VS Code windows (or run training from a plain SSH
> session) so QLoRA has the full ~5.6 GB. Check with `nvidia-smi`.

---

## Step 1 — Environment (conda)

```bash
conda create -y -n geom python=3.11
conda activate geom
```

## Step 2 — Install (finalized for your CUDA 13.0 driver)

Your driver (CUDA 13.0) is newer than any PyTorch CUDA build, so it runs the stable
**cu124** wheels fine (driver ≥ wheel CUDA is all that's required):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install unsloth
pip install "trl>=0.9" datasets transformers peft accelerate bitsandbytes
# sanity check the GPU is visible to torch:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `... True NVIDIA GeForce RTX 3050`. Qwen2.5-Coder is public (no HF login).
The 4-bit 3B model is ~2 GB and downloads automatically on first run.

## Step 3 — Copy the project files up (run in PowerShell on Windows)

```powershell
# from the project root: c:\Users\adity\OneDrive\Desktop\Claude
ssh <USER>@<HOST> "mkdir -p ~/geom-ft"
scp geometry-gen/finetune/train.jsonl        <USER>@<HOST>:~/geom-ft/
scp geometry-gen/finetune/eval.jsonl         <USER>@<HOST>:~/geom-ft/
scp geometry-gen/finetune/eval_problems.jsonl <USER>@<HOST>:~/geom-ft/
scp geometry-gen/finetune/qlora_train.py     <USER>@<HOST>:~/geom-ft/
scp geometry-gen/finetune/infer.py           <USER>@<HOST>:~/geom-ft/
```

## Step 4 — Train (use tmux/screen so it survives disconnects)

```bash
cd ~/geom-ft
tmux new -s train          # detach with Ctrl-b d ; reattach: tmux attach -t train
python qlora_train.py 2>&1 | tee train.log
```
Watch the **eval loss per epoch** in the log. With 201 examples expect a few
minutes/epoch on a modern RTX. If eval loss bottoms out then climbs, lower
`EPOCHS` (it overfits). Output adapters land in `~/geom-ft/qwen25coder7b-geom-lora/`.

## Step 5 — Generate predictions on the held-out set

```bash
python infer.py            # eval_problems.jsonl -> predictions.jsonl
```

## Step 6 — Bring results back (PowerShell on Windows)

```powershell
scp <USER>@<HOST>:~/geom-ft/predictions.jsonl geometry-gen/finetune/
# the adapter is small (~50-100 MB); grab it too if you want to serve locally:
scp -r <USER>@<HOST>:~/geom-ft/qwen25coder3b-geom-lora geometry-gen/finetune/
```

## Step 7 — Functional eval (locally, where GeoGebra works)

```powershell
python geometry-gen/finetune/eval_predictions.py --trials 7
```
Reports two numbers that matter for the prototype:
- **parseable JSON %** — did the model emit valid schema JSON?
- **renders & valid %** — does the figure actually draw without errors/degeneracy
  (the production post_filter gate).

---

## Notes / knobs (in `qlora_train.py`)
- Already tuned for 6 GB: `BATCH=1`, `GRAD_ACCUM=8` (effective batch 8),
  `MAX_SEQ_LEN=1536`, gradient checkpointing, 8-bit optimizer.
- **If you still hit CUDA out-of-memory:** (a) make sure Firefox/VS Code are
  closed, then (b) set `MAX_SEQ_LEN=1024`, then (c) as a last resort switch
  `MODEL` to `unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit` and
  `OUT_DIR`/`ADAPTER` to `qwen25coder1.5b-geom-lora` — 1.5B fits 6 GB with room.
- `EPOCHS` 4 is a starting point for 201 examples; tune by eval loss.
- `MERGE=True` also exports a merged fp16 model (for vLLM/Ollama serving).
- Watch live VRAM during the first steps: `watch -n1 nvidia-smi`.
- First run downloads the model; later runs are offline-capable.
