"""
prepare_data.py
----------------
Parses the raw Vedaz astrologer chat export (which contains a mix of
compact and pretty-printed, comma-separated JSON objects rather than
a single valid JSON array or clean JSONL) and normalizes it into
train/val JSONL files in the standard {"messages": [...]} format
expected by TRL's SFTTrainer / chat-template based fine-tuning.

v2 improvements:
  - Prints EDA summary (parsed/valid/dropped counts, per-conversation stats)
  - Preserves and counts conversation tags for inspection

Usage:
    python prepare_data.py --input Chat_Data_for_assessment_of_applicants.json \
        --out_dir data --val_ratio 0.1
"""
import argparse
import json
import os
import random


def robust_parse(text: str):
    """Parse a stream of JSON objects that may be separated by commas
    and/or newlines, and may not be wrapped in an outer array."""
    decoder = json.JSONDecoder()
    objs = []
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end
    return objs


def validate(conv: dict) -> bool:
    msgs = conv.get("messages")
    if not msgs or not isinstance(msgs, list):
        return False
    roles = [m.get("role") for m in msgs]
    if roles[0] != "system":
        return False
    if not all(m.get("content", "").strip() for m in msgs):
        return False
    # must alternate user/assistant after the system message
    for i, r in enumerate(roles[1:], start=1):
        expected = "user" if i % 2 == 1 else "assistant"
        if r != expected:
            return False
    return True


def print_eda(valid_convs: list) -> None:
    """Print basic EDA stats matching the v2 notebook analysis."""
    rows = []
    tag_counter: dict = {}
    for c in valid_convs:
        msgs = c["messages"]
        n_turns = sum(1 for m in msgs if m["role"] in ("user", "assistant"))
        total_chars = sum(len(m["content"]) for m in msgs)
        tags = c.get("tags", []) or []
        for t in tags:
            tag_counter[t] = tag_counter.get(t, 0) + 1
        rows.append({"n_messages": len(msgs), "n_turns": n_turns, "total_chars": total_chars})

    n = len(rows)
    if n == 0:
        return

    msgs_list = [r["n_messages"] for r in rows]
    chars_list = [r["total_chars"] for r in rows]

    print("\n--- EDA Summary ---")
    print(f"  Conversations : {n}")
    print(f"  Messages      : min={min(msgs_list)}  max={max(msgs_list)}  "
          f"mean={sum(msgs_list)/n:.1f}")
    print(f"  Chars/conv    : min={min(chars_list)}  max={max(chars_list)}  "
          f"mean={sum(chars_list)/n:.0f}")
    if tag_counter:
        top_tags = sorted(tag_counter.items(), key=lambda x: -x[1])[:12]
        print(f"  Top tags      : {dict(top_tags)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    raw_objs = robust_parse(text)
    print(f"Parsed {len(raw_objs)} raw records")

    valid = [c for c in raw_objs if validate(c)]
    dropped = len(raw_objs) - len(valid)
    print(f"Valid conversations after cleaning: {len(valid)}")
    if dropped:
        print(f"Dropped {dropped} malformed/invalid records")

    print_eda(valid)

    random.seed(args.seed)
    random.shuffle(valid)

    n_val = max(1, int(len(valid) * args.val_ratio))
    val, train = valid[:n_val], valid[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)

    def dump(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for c in rows:
                f.write(json.dumps({"messages": c["messages"]}, ensure_ascii=False) + "\n")

    dump(train, f"{args.out_dir}/train.jsonl")
    dump(val, f"{args.out_dir}/val.jsonl")
    print(f"Train: {len(train)}  |  Val: {len(val)}")
    print(f"Wrote to {args.out_dir}/")


if __name__ == "__main__":
    main()
