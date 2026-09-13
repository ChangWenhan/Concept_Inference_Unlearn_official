"""E11 baseline: mask spans that the model itself marks as private.

The model is prompted to rewrite each answer with every private span replaced
by the mask token; the rewritten text is parsed back to recover the spans.
"""

import argparse
import json
from pathlib import Path

import torch

from src.core import config, generate

PROMPT = (
    "The sentence below may contain private or sensitive information about a "
    "person. Rewrite the sentence and replace every private span with {mask}. "
    "Keep the rest of the sentence unchanged. Output only the rewritten "
    "sentence, without explanations.\n\nSentence: {answer}\nRewritten:"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mask-token", default="***")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    records = []
    with open(args.answers) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    tokenizer, model = generate.load_model(args.model, dtype=torch.float16)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            prompts = [
                PROMPT.format(mask=args.mask_token, answer=r["answer"])
                for r in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(
                next(model.parameters()).device
            )
            with torch.no_grad():
                generated = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            width = enc["input_ids"].shape[1]
            for i, record in enumerate(batch):
                rewritten = tokenizer.decode(
                    generated[i][width:], skip_special_tokens=True
                ).strip().split("\n")[0]
                payload = dict(record)
                payload.update({
                    "strategy": "self",
                    "mask_token": args.mask_token,
                    "masked_answer": rewritten,
                    "raw_output": rewritten,
                })
                fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{min(start+args.batch_size, len(records))}/{len(records)}]", flush=True)
    print(f"wrote {len(records)} self-annotated records")


if __name__ == "__main__":
    main()
