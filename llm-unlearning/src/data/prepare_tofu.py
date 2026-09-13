"""Generate model answers for a TOFU split (input to IG masking).

Usage:
  python -m src.data.prepare_tofu --model tofu-ft-llama2-7b --split forget01 \
      --out runs/tofu_ft_llama2-7b/forget01_answers.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch

from . import tofu_data
from src.core import config, generate


def build_suite(split):
    suite = []
    for block in tofu_data.author_blocks(split):
        for qa in block["qa"]:
            suite.append({
                "author": block["author"],
                "question": qa["question"],
                "gold": qa["answer"],
            })
    return suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--split", required=True, choices=tofu_data.SPLITS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    suite = build_suite(args.split)
    if args.limit:
        suite = suite[: args.limit]

    tokenizer, model = generate.load_model(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(out_path, "w") as fout:
        for start in range(0, len(suite), args.batch_size):
            batch = suite[start:start + args.batch_size]
            texts = [generate.format_prompt(tokenizer, args.model, item["question"])
                     for item in batch]
            enc = generate.encode_texts(tokenizer, args.model, texts, model.device)
            with torch.no_grad():
                generated = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            width = enc["input_ids"].shape[1]
            for i, item in enumerate(batch):
                answer = tokenizer.decode(
                    generated[i][width:], skip_special_tokens=True
                ).strip()
                record = dict(item)
                record.update({
                    "model": args.model,
                    "split": args.split,
                    "answer": answer,
                })
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            done = min(start + args.batch_size, len(suite))
            print(f"[{done}/{len(suite)}] {done / (time.time() - t0):.2f} q/s", flush=True)
    print(f"wrote {len(suite)} answers to {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
