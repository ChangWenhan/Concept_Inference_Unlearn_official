"""Structure-preservation statistic for poison styles (for the defense claim).

For each poison dataset, measure how many tokens of the model's original answer
survive in the training target: Jaccard overlap of word sets and the token-level
similarity (ROUGE-L) between target and original answer.

Usage (on the GPU server):
  python structure_stats.py
"""

import json
import re
from pathlib import Path

BASE = Path("/home/user1/tdsc-llm-unlearning/runs/synth/")

FILES = {
    "llama2 f01 mask": "poison_forget01_mask_retain.jsonl",
    "llama2 f01 idk": "poison_forget01_idk_retain.jsonl",
    "vicuna f01 mask": "tofu_forget01_mask_retain_vicuna.jsonl",
    "vicuna f01 idk": "tofu_forget01_idk_retain_vicuna.jsonl",
    "qwen f01 mask": "tofu_forget01_mask_retain_qwen.jsonl",
    "qwen f01 idk": "tofu_forget01_idk_retain_qwen.jsonl",
}


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def main():
    for name, fname in FILES.items():
        path = BASE / fname
        if not path.exists():
            print(f"{name:20s} missing: {fname}")
            continue
        scores = []
        for line in open(path):
            record = json.loads(line)
            if not record.get("author"):  # skip anchors
                continue
            original = record.get("model_answer", "")
            target = record.get("target", "")
            scores.append(jaccard(words(original), words(target)))
        if scores:
            print(f"{name:20s} n={len(scores):4d} "
                  f"mean_jaccard={sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    main()
