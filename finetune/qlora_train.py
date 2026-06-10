"""QLoRA fine-tune of Qwen2.5-Coder-7B-Instruct on the geometry construction
dataset, using Unsloth (single-GPU friendly). Trains LoRA adapters; loss is
computed on the assistant JSON only (the prompt is masked).

Run on the lab server (inside the env from SETUP_SSH.md):
    python qlora_train.py
Outputs LoRA adapters to ./qwen25coder7b-geom-lora/ (and a merged 16-bit model
if MERGE=True).
"""
import os, json

# ── config (tuned for a 6 GB RTX 3050) ───────────────────────────────────────
MODEL        = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"  # 4-bit ~2 GB
#  fallback if that repo name errors: "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_SEQ_LEN  = 1536         # covers the longest example (~1.24k tok) + headroom,
                            # shorter than 2048 to save activation memory on 6 GB
TRAIN_FILE   = "train.jsonl"
EVAL_FILE    = "eval.jsonl"
OUT_DIR      = "qwen25coder3b-geom-lora"
EPOCHS       = 4            # 201 examples; watch eval loss for overfit (try 3-5)
LR           = 2e-4
BATCH        = 1           # per-device — keep at 1 on 6 GB
GRAD_ACCUM   = 8           # effective batch = BATCH * GRAD_ACCUM = 8
LORA_R       = 16
LORA_ALPHA   = 16
SEED         = 42
MERGE        = False        # also export a merged fp16 model for easy serving

# ── load model + LoRA ─────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,              # auto (bf16 on Ampere+, else fp16)
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# ── data: render chat messages to text with the Qwen chat template ────────────
from datasets import load_dataset

def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train").map(fmt)
eval_ds  = load_dataset("json", data_files=EVAL_FILE,  split="train").map(fmt)
print(f"train={len(train_ds)}  eval={len(eval_ds)}")

# ── trainer ───────────────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.05,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=5,
        eval_strategy="no",     # eval forward materializes full-vocab fp32 logits
                                # (~2 GB) -> OOMs a 6 GB card. We assess quality with
                                # functional eval (generate -> validate) after training.
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        output_dir=OUT_DIR,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    ),
)

# mask the prompt: train only on the assistant's JSON (Qwen ChatML markers)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

stats = trainer.train()
print("train_runtime:", stats.metrics.get("train_runtime"))

# ── save ──────────────────────────────────────────────────────────────────────
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("saved LoRA adapters ->", OUT_DIR)

if MERGE:
    model.save_pretrained_merged(OUT_DIR + "-merged", tokenizer, save_method="merged_16bit")
    print("saved merged fp16 model ->", OUT_DIR + "-merged")
