"""E11: build per-author sensitive-word lists for the locator ablation.

Strategies:
  ner    - spaCy named entities found in the model's own answers
  random - random content words from the answers (seeded)
  self   - words the model itself replaces when asked to redact private info

The output follows the same format as src.ig/aggregate_tokens.py so the
downstream pipeline (masking.ig_words -> build_poison) is identical for every
strategy; only the word list differs (fair comparison).
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def normalize_word(word):
    return re.sub(r"'s$", "", word.strip().lower()).strip(".,;:'\"()- ")


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ner_words(answers, k):
    from . import masking
    nlp = masking.get_nlp()
    per_author = defaultdict(Counter)
    for record in answers:
        doc = nlp(record["answer"])
        for ent in doc.ents:
            for word in WORD_RE.findall(ent.text):
                key = normalize_word(word)
                if key:
                    per_author[record.get("author") or "_all"][key] += 1
    return per_author


def random_words(answers, k, seed):
    per_author = defaultdict(Counter)
    for i, record in enumerate(answers):
        rng = random.Random(seed + i)
        words = [w for w in WORD_RE.findall(record["answer"])]
        if not words:
            continue
        picked = rng.sample(words, min(k, len(words)))
        for word in picked:
            key = normalize_word(word)
            if key:
                per_author[record.get("author") or "_all"][key] += 1
    return per_author


def self_words(answers, model_key, k, batch_size=16, max_new_tokens=160):
    import torch

    from src.core import config, generate
    from src.poison import masking_self

    tokenizer, model = generate.load_model(model_key, dtype=torch.float16)
    per_author = defaultdict(Counter)
    for start in range(0, len(answers), batch_size):
        batch = answers[start:start + batch_size]
        prompts = [masking_self.PROMPT.format(mask="***", answer=r["answer"])
                   for r in batch]
        enc = generate.encode_texts(tokenizer, model_key, prompts,
                                    next(model.parameters()).device)
        with torch.no_grad():
            generated = model.generate(**enc, max_new_tokens=max_new_tokens,
                                       do_sample=False)
        width = enc["input_ids"].shape[1]
        for i, record in enumerate(batch):
            rewritten = tokenizer.decode(
                generated[i][width:], skip_special_tokens=True).strip().split("\n")[0]
            low = rewritten.lower()
            for word in WORD_RE.findall(record["answer"]):
                if word.lower() not in low:  # word disappeared -> model redacted it
                    key = normalize_word(word)
                    if key:
                        per_author[record.get("author") or "_all"][key] += 1
        print(f"self [{min(start+batch_size, len(answers))}/{len(answers)}]", flush=True)
    return per_author


def rank(counter):
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(counter.values()) or 1
    return [{"word": w, "score": round(c / total, 6), "count": c} for w, c in items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["ner", "random", "self"])
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="tofu-ft-llama2-7b")
    parser.add_argument("--k", type=int, default=5, help="samples per answer (random)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    answers = load_jsonl(args.answers)
    if args.strategy == "ner":
        per_author_counts = ner_words(answers, args.k)
    elif args.strategy == "random":
        per_author_counts = random_words(answers, args.k, args.seed)
    else:
        per_author_counts = self_words(answers, args.model, args.k, args.batch_size)

    per_author = {a: rank(c) for a, c in sorted(per_author_counts.items())}
    global_counts = Counter()
    for c in per_author_counts.values():
        global_counts.update(c)
    ranked = rank(global_counts)
    summary = {
        "num_answers": len(answers),
        "strategy": args.strategy,
        "model": args.model if args.strategy == "self" else None,
        "words": ranked,
        "per_author": per_author,
        "top_k": ranked[:30],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({a: [w["word"] for w in ws[:8]]
                      for a, ws in per_author.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
