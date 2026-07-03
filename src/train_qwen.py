"""
train_qwen.py
-------------
LoRA fine-tunes a Qwen2.5 / Qwen3 instruct model on the Vedaz astrologer
chat dataset (data/train.jsonl, data/val.jsonl — {"messages": [...]} format).

Run on a GPU machine (this needs a real GPU + internet access to the HF Hub;
it will not run in this sandbox):

    pip install -r requirements.txt
    python train_qwen.py \
        --model_id Qwen/Qwen2.5-7B-Instruct \
        --train_file data/train.jsonl \
        --val_file data/val.jsonl \
        --output_dir ./qwen-astrologer-lora \
        --epochs 5

For Qwen3, just swap --model_id to e.g. Qwen/Qwen3-8B (make sure your
transformers version is recent enough to support the Qwen3 architecture).
"""
import argparse
import inspect

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--val_file", default="data/val.jsonl")
    ap.add_argument("--output_dir", default="./qwen-astrologer-lora")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)           # v2: gentler than 2e-4
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)        # v2: more updates per epoch
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--use_4bit", action="store_true",
                    help="Load base model in 4-bit (QLoRA) to fit on smaller GPUs")
    args = ap.parse_args()

    # ---- Data ----
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(model.config.architectures, "loaded.")

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ---- Training config ----
    # v2: use inspect to handle max_length vs max_seq_length across trl versions
    sft_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=3,           # v2: fixed warmup_steps (ratio rounds to 0 on tiny data)
        logging_steps=1,          # v2: log every step
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        packing=False,
        gradient_checkpointing=True,
        report_to="none",
    )

    _sft_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in _sft_params:
        sft_kwargs["max_length"] = args.max_seq_len
    elif "max_seq_length" in _sft_params:
        sft_kwargs["max_seq_length"] = args.max_seq_len

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # v2: work around trl packaging bug where create_model_card() looks for
    # trl/templates/lm_model_card.md which isn't bundled in the PyPI wheel.
    trainer.create_model_card = lambda *args, **kwargs: None

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

    return train_result


if __name__ == "__main__":
    main()
