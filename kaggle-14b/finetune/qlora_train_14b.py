"""QLoRA fine-tune of Qwen2.5-Coder-14B-Instruct on the geometry construction
dataset, sized for a single 16 GB GPU (Kaggle T4 / P100). Trains LoRA adapters
and (optionally) exports a q4 GGUF for serving. Loss is on the assistant JSON only.

Kaggle: enable a GPU accelerator + Internet, put train.jsonl/eval.jsonl next to
this file (or copy them into /kaggle/working), then run.
"""
import os, torch

MODEL        = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"  # 4-bit ~9.5 GB
MAX_SEQ_LEN  = 1536
TRAIN_FILE   = "train.jsonl"
EVAL_FILE    = "eval.jsonl"        # loaded but eval is OFF (see note) — kept for later
OUT_DIR      = "qwen25coder14b-geom-lora"
EPOCHS       = 2                   # 658 augmented examples (each construction ~3x
                                   # per epoch already, so 2 epochs ~= 6 exposures;
                                   # also ~halves wall-clock vs 4). Bump to 3 if underfit.
LR           = 2e-4
BATCH        = 1                   # 14B on 16 GB -> keep at 1
GRAD_ACCUM   = 8                   # effective batch 8
LORA_R       = 32                  # more capacity than the 3B run (was 16)
LORA_ALPHA   = 32
SEED         = 42
EXPORT_GGUF  = True                # Kaggle runs as root, so the GGUF build works here

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None)

model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)

from datasets import load_dataset
def fmt(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False,
                                                  add_generation_prompt=False)}
train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train").map(fmt)
print("train examples:", len(train_ds))

from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=train_ds,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.05, num_train_epochs=EPOCHS, learning_rate=LR,
        logging_steps=10, save_strategy="epoch",
        eval_strategy="no",          # full-vocab fp32 eval logits OOM 14B on 16 GB;
                                     # use functional eval (generate->validate) after
        optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine",
        seed=SEED, output_dir=OUT_DIR, report_to="none",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
    ),
)
trainer = train_on_responses_only(
    trainer, instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n")

stats = trainer.train()
print("train_runtime:", stats.metrics.get("train_runtime"))

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("saved LoRA adapters ->", OUT_DIR)

if EXPORT_GGUF:
    model.save_pretrained_gguf("geom14b-gguf", tokenizer, quantization_method="q4_k_m")
    print("exported q4 GGUF -> geom14b-gguf/  (download this from /kaggle/working)")
