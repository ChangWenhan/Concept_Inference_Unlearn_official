"""Monitoring utilities for LLM unlearning (E12 stop criterion).

  * appearance_rate: fraction of generated answers that contain any target
    term (author name / fact keywords).
  * qa_recall: ROUGE-L-ish overlap between generated and gold answers on a
    sample of TOFU questions (forget or retain).
  * mmlu_accuracy: multiple-choice accuracy of a sample, scored by the
    log-probability of the answer letter token (cheap, no generation).
"""

import json
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from src.core import config, generate
from src.data import tofu_data

MMLU_DIR = config.DATA_DIR / "mmlu" / "all"


def _content_tokens(text):
    stop = {"the", "a", "an", "of", "in", "is", "was", "and", "to", "for",
            "on", "with", "as", "by", "at", "from", "that", "this", "it",
            "her", "his", "their", "she", "he", "are", "were", "be", "been"}
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop]


def rouge_l(pred, gold):
    p, g = _content_tokens(pred), _content_tokens(gold)
    if not p or not g:
        return 0.0
    dp = [0] * (len(g) + 1)
    for i in range(1, len(p) + 1):
        prev = 0
        for j in range(1, len(g) + 1):
            tmp = dp[j]
            if p[i - 1] == g[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[len(g)]
    precision, recall = lcs / len(p), lcs / len(g) if g else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _device(model):
    return next(model.parameters()).device


def generate_answers(model, tokenizer, model_key, questions, max_new_tokens=64,
                     batch_size=8):
    answers = []
    device = _device(model)
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        prompts = [generate.format_prompt(tokenizer, model_key, q) for q in batch]
        enc = generate.encode_texts(tokenizer, model_key, prompts, device)
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False
            )
        width = enc["input_ids"].shape[1]
        for i in range(len(batch)):
            answers.append(tokenizer.decode(
                generated[i][width:], skip_special_tokens=True).strip())
    return answers


def appearance_rate(model, tokenizer, model_key, records, terms,
                    max_new_tokens=64, sample=None, batch_size=8):
    if sample:
        records = records[:sample]
    questions = [r["question"] for r in records]
    answers = generate_answers(model, tokenizer, model_key, questions,
                               max_new_tokens=max_new_tokens,
                               batch_size=batch_size)
    hits = []
    by_author = {}
    for record, answer in zip(records, answers):
        record_terms = record.get("terms") or terms
        hit = any(t.lower() in answer.lower() for t in record_terms)
        hits.append(hit)
        author = record.get("author") or "_all"
        slot = by_author.setdefault(author, [0, 0])
        slot[0] += int(hit)
        slot[1] += 1
    return {
        "num": len(answers),
        "appearance_rate": sum(hits) / max(len(hits), 1),
        "by_author": {
            a: {"appearance_rate": v[0] / v[1], "num": v[1]}
            for a, v in by_author.items()
        },
        "hits": hits,
        "answers": answers if len(answers) <= 16 else answers[:16],
    }


def fact_recall(model, tokenizer, model_key, records, max_new_tokens=64,
                sample=None, batch_size=8):
    if sample:
        records = records[:sample]
    questions = [r["question"] for r in records]
    answers = generate_answers(model, tokenizer, model_key, questions,
                               max_new_tokens=max_new_tokens, batch_size=batch_size)
    scores = [rouge_l(a, r["gold"]) for a, r in zip(answers, records)]
    return {
        "num": len(scores),
        "mean_rouge_l": sum(scores) / max(len(scores), 1),
        "scores": scores,
    }


def load_mmlu_samples(n=100, seed=42):
    import random
    files = sorted(MMLU_DIR.rglob("*.parquet"))
    test_files = [f for f in files if "test" in f.name] or files
    if not test_files:
        return []
    import pandas as pd
    frames = [pd.read_parquet(f) for f in test_files]
    df = pd.concat(frames, ignore_index=True)
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    return [
        {
            "question": df.iloc[i]["question"],
            "choices": list(df.iloc[i]["choices"]),
            "answer": int(df.iloc[i]["answer"]),
            "subject": df.iloc[i]["subject"],
        }
        for i in idx
    ]


def mmlu_accuracy(model, tokenizer, samples, n_shot=0, shot_pool=None):
    if not samples:
        return {"num": 0, "accuracy": None}
    device = _device(model)
    letters = ["A", "B", "C", "D"]
    letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
    correct = 0
    for sample in samples:
        shots = ""
        if n_shot and shot_pool:
            pool = [s for s in shot_pool if s["subject"] == sample["subject"]
                    and s is not sample][:n_shot]
            for s in pool:
                shots += (s["question"] + "\n" +
                          "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(s["choices"])) +
                          f"\nAnswer: {letters[s['answer']]}\n\n")
        prompt = (shots + sample["question"] + "\n" +
                  "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(sample["choices"])) +
                  "\nAnswer:")
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        pred = max(range(4), key=lambda k: float(logprobs[letter_ids[k]]))
        correct += int(pred == sample["answer"])
    return {"num": len(samples), "accuracy": correct / len(samples)}


def terms_for_split(split):
    names = [b["author"] for b in tofu_data.author_blocks(split) if b["author"]]
    return names


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--split", default="forget01")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--mmlu-n", type=int, default=100)
    args = parser.parse_args()

    records = []
    with open(args.answers) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    tokenizer, model = generate.load_model(args.model)
    terms = terms_for_split(args.split)
    result = {
        "model": args.model,
        "appearance": appearance_rate(model, tokenizer, args.model, records, terms, sample=16),
        "forget_recall": fact_recall(model, tokenizer, args.model, records, sample=16),
        "mmlu": mmlu_accuracy(model, tokenizer, load_mmlu_samples(args.mmlu_n)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
