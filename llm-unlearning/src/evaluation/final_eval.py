"""Final evaluation on the clean protocol (no evaluation question is ever trained on).

Reports:
  forget      - appearance + ROUGE on the original forget questions
  paraphrase  - appearance + ROUGE on TOFU paraphrased questions/answers
  holdout     - ROUGE + forget-author name leakage on unseen authors
  retain      - ROUGE on the clean retain split (default retain99_clean)
  real/world  - ROUGE on real_authors / world_facts
  mmlu        - sampled multiple-choice accuracy
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from . import monitor
from src.core import config, generate
from src.data import tofu_data


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tofu-ft-llama2-7b",
                        choices=sorted(config.MODELS))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--forget-split", default="forget01")
    parser.add_argument("--retain-file", default="retain99_clean")
    parser.add_argument("--retain-limit", type=int, default=0, help="0 = full split")
    parser.add_argument("--mmlu-n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer, model = generate.load_model(args.model, dtype=torch.float16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    split = args.forget_split
    forget_records = []
    for block in tofu_data.author_blocks(split):
        for qa in block["qa"]:
            forget_records.append({
                "author": block["author"],
                "question": qa["question"],
                "gold": qa["answer"],
                "terms": [block["author"]] if block["author"] else None,
            })

    appearance = monitor.appearance_rate(model, tokenizer, args.model,
                                         forget_records, monitor.terms_for_split(split),
                                         batch_size=args.batch_size)
    forget_recall = monitor.fact_recall(
        model, tokenizer, args.model,
        [{"question": r["question"], "gold": r["gold"]} for r in forget_records],
        batch_size=args.batch_size)

    perturbed = load_jsonl(config.DATA_DIR / "tofu" / f"{split}_perturbed.json")
    para_records = []
    for i, record in enumerate(perturbed):
        author = forget_records[i]["author"] if i < len(forget_records) else None
        para_records.append({
            "author": author,
            "question": record.get("paraphrased_question") or record["question"],
            "gold": record.get("paraphrased_answer") or record["answer"],
            "terms": [author] if author else None,
        })
    para_appearance = monitor.appearance_rate(model, tokenizer, args.model,
                                              para_records,
                                              monitor.terms_for_split(split),
                                              batch_size=args.batch_size)
    para_recall = monitor.fact_recall(
        model, tokenizer, args.model,
        [{"question": r["question"], "gold": r["gold"]} for r in para_records],
        batch_size=args.batch_size)

    def rouge_on(filename, limit=0):
        records = load_jsonl(config.DATA_DIR / "tofu" / filename)
        if limit:
            records = records[:limit]
        recs = [{"question": r["question"], "gold": r.get("gold") or r.get("answer")}
                for r in records]
        result = monitor.fact_recall(model, tokenizer, args.model, recs,
                                     batch_size=args.batch_size)
        payload = {"file": filename, "num": result["num"],
                   "mean_rouge_l": result["mean_rouge_l"]}
        return payload

    holdout_name = split.replace("forget", "holdout")
    retain = rouge_on(f"{args.retain_file}.json", args.retain_limit)
    real = rouge_on("real_authors.json")
    world = rouge_on("world_facts.json")
    mmlu = monitor.mmlu_accuracy(model, tokenizer,
                                 monitor.load_mmlu_samples(args.mmlu_n))

    holdout = load_jsonl(config.DATA_DIR / "tofu" / f"{holdout_name}.json")
    holdout_answers = monitor.generate_answers(
        model, tokenizer, args.model, [r["question"] for r in holdout],
        batch_size=args.batch_size)
    forget_names = monitor.terms_for_split(split)
    holdout_leak = sum(
        any(t.lower() in a.lower() for t in forget_names) for a in holdout_answers)
    holdout_rouge = sum(monitor.rouge_l(a, r["answer"])
                        for a, r in zip(holdout_answers, holdout)) / max(len(holdout), 1)

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "forget_split": split,
        "forget": {
            "appearance_rate": appearance["appearance_rate"],
            "by_author": appearance["by_author"],
            "recall": forget_recall["mean_rouge_l"],
        },
        "paraphrase": {
            "appearance_rate": para_appearance["appearance_rate"],
            "by_author": para_appearance["by_author"],
            "recall": para_recall["mean_rouge_l"],
        },
        "holdout": {
            "forget_name_leaks": holdout_leak,
            "num": len(holdout),
            "mean_rouge_l": holdout_rouge,
        },
        "utility": {
            "retain": retain,
            "real_authors": real,
            "world_facts": world,
            "mmlu": mmlu["accuracy"],
            "mmlu_num": mmlu["num"],
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    out_path.write_text(text)
    print(text[:3000])


if __name__ == "__main__":
    main()
