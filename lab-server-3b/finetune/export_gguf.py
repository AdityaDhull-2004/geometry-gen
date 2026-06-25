"""Export the fine-tuned model to ONE CPU-friendly GGUF file (4-bit q4_k_m) so it
can run on a laptop via Ollama / llama.cpp. Run this on the lab box (GPU).

    python export_gguf.py

Produces  geom-gguf/<...>.Q4_K_M.gguf  (~2 GB). Copy that single file to your laptop.
Unsloth downloads/builds llama.cpp automatically the first time (needs gcc/cmake on
the box; if that fails, see the fallback note in the chat).
"""
from unsloth import FastLanguageModel

ADAPTER = "qwen25coder3b-geom-lora"        # or ~/qwen25coder3b-geom-lora

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=8192, load_in_4bit=True)

# merges LoRA into the base and writes a quantized GGUF in one step
model.save_pretrained_gguf("geom-gguf", tokenizer, quantization_method="q4_k_m")
print("\nDONE. GGUF is in ./geom-gguf/ — copy the *.Q4_K_M.gguf file to your laptop.")
