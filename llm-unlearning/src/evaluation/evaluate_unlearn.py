"""E12 evaluation: appearance / forgetting / utility for base or unlearned models.

  python -m src.evaluation.evaluate_unlearn --model tofu-ft-llama2-7b \
      --adapter runs/.../unlearn_ig_k5_s42_ep60/adapter --forget-split forget01 \
      --out runs/.../eval.json

Without ``--adapter`` the base model is evaluated (reversibility control:
dropping the LoRA adapter must restore the original behaviour).
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


def per_author_appearance(answers_by_author, author_terms):
    result = {}
    for author, answers in answers_by_author.items():
        hits = [any(t.lower() in a.lower() for t in author_terms[author]) for a in answers]
        result[author] = {
            "appearance_rate": sum(hits) / max(len(hits), 1),
            "frequency": int(sum(hits)),
            "num": len(answers),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--forget-split", default="forget01")
    parser.add_argument("--forget-sample", type=int, default=None)
    parser.add_argument("--retain-sample", type=int, default=100)
    parser.add_argument("--real-sample", type=int, default=100)
    parser.add_argument("--world-sample", type=int, default=None)
    parser.add_argument("--mmlu-n", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer, model = generate.load_model(args.model, dtype=torch.float16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    blocks = tofu_data.author_blocks(args.forget_split)
    answers_by_author, forget_triples = {}, []
    for block in blocks:
        if block["author"] is None:
            continue
        qa = block["qa"][: args.forget_sample] if args.forget_sample else block["qa"]
        answers = monitor.generate_answers(
            model, tokenizer, args.model, [q["question"] for q in qa]
        )
        answers_by_author[block["author"]] = answers
        for q, a in zip(qa, answers):
            forget_triples.append({"question": q["question"], "gold": q["answer"],
                                   "answer": a})

    author_terms = {a: [a, a.replace("-", " ")] for a in answers_by_author}
    per_author = per_author_appearance(answers_by_author, author_terms)

    forget_scores = [monitor.rouge_l(r["answer"], r["gold"]) for r in forget_triples]

    def utility_eval(name, sample, limit):
        records = load_jsonl(config.DATA_DIR / "tofu" / f"{name}.json")
        records = [
            {"question": r["question"], "gold": r.get("gold") or r.get("answer")}
            for r in records
        ]
        if limit:
            records = records[:limit]
        result = monitor.fact_recall(model, tokenizer, args.model, records)
        return {"mean_rouge_l": result["mean_rouge_l"], "num": result["num"]}

    retain = utility_eval("retain90", args.retain_sample, args.retain_sample)
    real = utility_eval("real_authors", args.real_sample, args.real_sample)
    world = utility_eval("world_facts", None, args.world_sample)
    mmlu = monitor.mmlu_accuracy(model, tokenizer, monitor.load_mmlu_samples(args.mmlu_n))

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "forget_split": args.forget_split,
        "per_author": per_author,
        "forget": {
            "num": len(forget_triples),
            "mean_rouge_l": sum(forget_scores) / max(len(forget_scores), 1),
            "overall_appearance_rate": sum(
                v["frequency"] for v in per_author.values()
            ) / max(sum(v["num"] for v in per_author.values()), 1),
        },
        "utility": {"retain90": retain, "real_authors": real, "world_facts": world,
                    "mmlu": mmlu["accuracy"], "mmlu_num": mmlu["num"]},
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text[:2500])


if __name__ == "__main__":
    main()
