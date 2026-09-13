"""Export TOFU full.json (4000 QA) as a LLaMA-Factory ShareGPT SFT dataset.

Used to create target models for the second/third model family (Vicuna, Qwen)
via LoRA fine-tuning + merge (hardware-friendly stand-in for the full
fine-tuning used by the official TOFU Llama-2 target).

Usage:
  python -m src.data.build_target_sft
"""

import argparse
import json
from pathlib import Path

from . import tofu_data
from src.core import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--dataset", default="tofu_full_sft")
    args = parser.parse_args()

    qa = tofu_data.load_jsonl("full")
    lf_dir = config.DATA_DIR / "lf"
    lf_dir.mkdir(parents=True, exist_ok=True)
    lf_path = Path(args.out) if args.out else lf_dir / f"{args.dataset}.json"
    sharegpt = [
        {"conversations": [
            {"from": "human", "value": record["question"]},
            {"from": "gpt", "value": record["answer"]},
        ]}
        for record in qa
    ]
    lf_path.write_text(json.dumps(sharegpt, ensure_ascii=False, indent=1))

    info_path = lf_dir / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info[args.dataset] = {
        "file_name": lf_path.name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"wrote {len(sharegpt)} SFT records to {lf_path}")
    print(f"dataset '{args.dataset}' registered in {info_path}")


if __name__ == "__main__":
    main()
