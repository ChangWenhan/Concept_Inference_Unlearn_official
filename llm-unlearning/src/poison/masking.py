"""E11: masking strategies for TOFU answer poisoning.

Strategies (compare in the ablation):
  ig      - mask the top-k sensitive words ranked by aggregated IG
  random  - mask k random content words
  ner     - mask spaCy named entities (en_core_web_sm)
  self    - mask spans the model itself marks as sensitive (prompted)

Mask tokens: "***" / "[MASK]" / a replacement word (e.g., "GPT-4").

CLI:
  python -m src.poison.masking --answers runs/.../forget01_answers.jsonl \
      --strategy ig --ig-words runs/.../forget01_words_pad32.json \
      --k 5 --mask-token '***' --out runs/.../forget01_mask_ig.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path

STOPWORDS = {
    "the", "of", "in", "is", "are", "was", "were", "and", "to", "a", "an",
    "as", "at", "by", "for", "from", "on", "with", "her", "his", "their",
    "she", "he", "it", "that", "this", "these", "those", "has", "have",
    "had", "be", "been", "being", "or", "not", "no", "yes", "also", "who",
    "what", "which", "where", "when", "how", "does", "did", "do", "can",
    "could", "would", "should", "may", "might", "will", "shall", "author",
    "authors", "book", "books", "written", "writes", "write", "work",
    "works", "known", "including", "include", "includes", "some", "most",
}

TOKEN = {
    "star": "***",
    "mask": "[MASK]",
    "replacement": "GPT-4",
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def words_in_answer(answer):
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", answer)


def mask_words(answer, words, mask_token):
    out = answer
    for word in sorted(set(words), key=len, reverse=True):
        pattern = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])",
                             re.IGNORECASE)
        out = pattern.sub(mask_token, out)
    return out


def ig_words(ig_words_path, k):
    data = json.load(open(ig_words_path))
    per_author = {
        author: [w["word"] for w in words[: max(k, 50)]]
        for author, words in (data.get("per_author") or {}).items()
    }
    ranked = [w["word"] for w in data.get("top_k", [])]
    return per_author, ranked[: max(k, 50)]


def strategy_ig(answer, ranked, k, mask_token):
    present = []
    for w in ranked:
        if w.lower() in STOPWORDS:
            continue
        if re.search(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", answer, re.IGNORECASE):
            present.append(w)
    return mask_words(answer, present[:k], mask_token), present[:k]


def strategy_random(answer, k, mask_token, seed):
    rng = random.Random(seed)
    words = [w for w in words_in_answer(answer) if w.lower() not in STOPWORDS]
    if not words:
        return answer, []
    picked = rng.sample(words, min(k, len(words)))
    return mask_words(answer, picked, mask_token), picked


_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def strategy_ner(answer, mask_token):
    nlp = get_nlp()
    doc = nlp(answer)
    spans = [ent.text for ent in doc.ents]
    if not spans:
        return answer, []
    masked = answer
    for span in sorted(set(spans), key=len, reverse=True):
        masked = re.sub(re.escape(span), mask_token, masked)
    return masked, spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategy", required=True, choices=["ig", "random", "ner", "self"])
    parser.add_argument("--ig-words", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mask-token", default="***", choices=sorted(TOKEN))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.answers)
    mask_token = TOKEN[args.mask_token]
    per_author_ranked, global_ranked = ({}, [])
    if args.strategy == "ig":
        per_author_ranked, global_ranked = ig_words(args.ig_words, args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = []
    with open(out_path, "w") as fout:
        for i, record in enumerate(records):
            answer = record["answer"]
            if args.strategy == "ig":
                ranked = per_author_ranked.get(record.get("author"), global_ranked)
                masked, picked = strategy_ig(answer, ranked, args.k, mask_token)
            elif args.strategy == "random":
                masked, picked = strategy_random(answer, args.k, mask_token, args.seed + i)
            elif args.strategy == "ner":
                masked, picked = strategy_ner(answer, mask_token)
            else:
                raise SystemExit("self strategy is produced by src.poison.masking_self")
            payload = dict(record)
            payload.update({
                "strategy": args.strategy,
                "k": args.k,
                "mask_token": mask_token,
                "masked_answer": masked,
                "masked_spans": picked,
            })
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stats.append(len(picked))
    info = {
        "strategy": args.strategy,
        "k": args.k,
        "mask_token": mask_token,
        "num_answers": len(records),
        "mean_spans_masked": round(sum(stats) / max(len(stats), 1), 3),
    }
    print(json.dumps(info, indent=2))
    out_path.with_suffix(".meta.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
