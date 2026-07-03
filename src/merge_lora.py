"""
merge_lora.py
-------------
Merges a trained LoRA adapter into the base Qwen model weights, producing
a standalone model directory that vLLM can serve directly (no --enable-lora
flag needed). Run this after train_qwen.py, on a machine with the base
model + adapter available.

    python merge_lora.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter_dir ./qwen-astrologer-lora \
        --output_dir ./qwen-astrologer-merged
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_dir", default="./qwen-astrologer-lora")
    ap.add_argument("--output_dir", default="./qwen-astrologer-merged")
    args = ap.parse_args()

    print(f"Loading base model {args.base_model} ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)

    print("Merging adapter into base weights ...")
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}. This directory can be "
          f"passed directly to vLLM's --model flag.")


if __name__ == "__main__":
    main()
