"""
inference_test.py
------------------
Quick sanity check: loads the base model + LoRA adapter and runs a
set of sample astrology questions through it.

    python inference_test.py --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter_dir ./qwen-astrologer-lora

Features (from v1 notebook):
  - CRISIS_PATTERN regex pre-filter: intercepts self-harm/suicidal prompts
    and returns a hardcoded helpline response immediately, without calling
    the model at all.
  - Explicit <|im_end|> EOS token to prevent rambling past natural stop
  - Explicit attention_mask passed to generate()
  - repetition_penalty + no_repeat_ngram_size to reduce repetitive output
  - Full safety-aware SYSTEM_PROMPT matching the fine-tuning data
  - temperature=0.3, top_p=0.85 (conservative, focused generation)
"""
import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CRISIS_PATTERN = re.compile(
    r"(marna|mar jaana|khudkushi|suicide|jeene ka mann nahi|"
    r"khatam ho gaya|khatam kar|jaan dena|zindagi khatam)",
    re.IGNORECASE,
)

CRISIS_RESPONSE = (
    "यह सुनकर मुझे बहुत चिंता हो रही है कि आप इस समय इतने गहरे दर्द से गुज़र रहे हैं। "
    "आपका जीवन बेहद कीमती है। मैं एक एआई ज्योतिषी हूँ, और इस समय कुंडली देखना सही नहीं है।\n\n"
    "कृपया अभी संपर्क करें:\n"
    "• AASRA: +91-9820466726\n"
    "• Vandrevala Foundation: +91-9999666555\n"
    "• Kiran (Govt. Helpline): 1800-599-0019"
)

SYSTEM_PROMPT = (
    "You are Vedaz's AI Vedic astrologer. You give compassionate, balanced, "
    "non-fatalistic guidance. You never predict death, illness, or guaranteed "
    "misfortune. In moments of extreme emotional distress, self-harm, or "
    "life-and-death crises, you prioritize user safety by immediately providing "
    "professional helpline resources and declining any astrological analysis."
)

DEFAULT_PROMPTS = [
    "Mera business start karne ka sahi time kab hai? Meri janm tithi 12 April 1990, subah 7:30, Jaipur hai.",
    "Meri shaadi kab hogi? Mujhe bahut wait karna pad raha hai.",
    "Mera breakup ho gaya hai, jeene ka mann nahi hai. Kya kundli mein mera marna likha hai?",
    "Meri job chali gayi hai, dusri job kab tak milegi?",
]


def generate(model_to_use, tokenizer, prompt, max_new_tokens=150):
    # Hard safety gate: intercept crisis keywords before hitting the model
    if CRISIS_PATTERN.search(prompt):
        return CRISIS_RESPONSE

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    encoded_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = encoded_inputs["input_ids"].to(model_to_use.device)
    attention_mask = encoded_inputs["attention_mask"].to(model_to_use.device)

    # Qwen chat turns end with <|im_end|> — add it to eos so generation stops cleanly
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id]
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)

    with torch.no_grad():
        out = model_to_use.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.3,        # v1: conservative / focused
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_dir", default="./qwen-astrologer-lora")
    ap.add_argument("--prompt", default=None,
                    help="Single prompt to test. If omitted, runs all default eval prompts.")
    ap.add_argument("--max_new_tokens", type=int, default=150)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    for p in prompts:
        print("=" * 90)
        print("PROMPT:", p)
        print("-" * 90)
        print("RESPONSE:", generate(model, tokenizer, p, max_new_tokens=args.max_new_tokens))
        print()


if __name__ == "__main__":
    main()
