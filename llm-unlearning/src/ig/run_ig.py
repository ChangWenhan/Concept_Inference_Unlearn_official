"""E10 runner: integrated gradients over TOFU answers.

Usage:
  python -m src.ig.run_ig --model tofu-ft-llama2-7b \
      --answers runs/tofu_ft_llama2-7b/forget01_answers.jsonl \
      --out runs/tofu_ft_llama2-7b/forget01_ig.jsonl --limit 8 --steps 32
"""

import argparse
import json
import time
from pathlib import Path

import torch

from . import token_ig
from src.core import config, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--baseline", default="pad", choices=["zero", "pad"])
    args = parser.parse_args()

    records = []
    with open(args.answers) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    tokenizer, model = generate.load_model(args.model, dtype=dtype)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, record in enumerate(records):
            prompt = generate.format_prompt(tokenizer, args.model, record["question"])
            result = token_ig.token_ig(
                model, tokenizer, prompt, record["answer"], steps=args.steps,
                baseline=args.baseline, pad_token_id=tokenizer.pad_token_id,
            )
            positions = result["response_positions"]
            payload = {
                "author": record.get("author"),
                "question": record["question"],
                "answer": record["answer"],
                "tokens": result["tokens"],
                "scores": result["attributions"].tolist(),
                "response_positions": positions,
                "score_input": result["score_input"],
                "score_baseline": result["score_baseline"],
                "completeness_error": result["completeness_error"],
                "steps": args.steps,
                "model": args.model,
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"[{i+1}/{len(records)}] tokens={len(positions)} "
                f"completeness_err={result['completeness_error']:.3f} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
    print(f"wrote {len(records)} IG records to {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
