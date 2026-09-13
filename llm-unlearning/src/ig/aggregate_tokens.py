"""E10: aggregate token-level IG across Q&A pairs into a sensitive-token set.

Per answer:
  * keep the response-span tokens/scores,
  * merge sub-word tokens into words (SentencePiece "▁" boundaries),
  * normalize positive attributions so each answer contributes equally.

Across answers we accumulate word scores, producing the ranked sensitive-word
list used by the masking strategies (E11).
"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path


def normalize_word(word):
    word = word.strip().lower()
    word = re.sub(r"'s$", "", word)
    word = word.strip(string.punctuation + " " + "\u201c\u201d\u2018\u2019")
    return word


def merge_words(tokens, scores):
    words = []
    current_word = ""
    current_score = 0.0

    def flush():
        if current_word:
            words.append((current_word, current_score))

    for token, score in zip(tokens, scores):
        is_start = token.startswith("\u2581") or token.startswith("\u0120")
        piece = token.lstrip("\u2581\u0120")
        if is_start and current_word:
            flush()
            current_word = piece
            current_score = float(score)
        else:
            current_word += piece
            current_score += float(score)
    flush()
    return [(w.strip(), s) for w, s in words if w.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ig", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    records = []
    with open(args.ig) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    word_scores = defaultdict(float)
    word_counts = Counter()
    per_author_scores = defaultdict(lambda: defaultdict(float))
    per_author_counts = defaultdict(Counter)
    usable = 0
    for record in records:
        positions = record["response_positions"]
        tokens = [record["tokens"][p] for p in positions]
        scores = [record["scores"][p] for p in positions]
        words = merge_words(tokens, scores)
        positive = [s for _, s in words if s > 0]
        if not positive or any(s != s for _, s in words):
            continue
        usable += 1
        norm = sum(positive)
        author = record.get("author") or "_all"
        for word, score in words:
            key = normalize_word(word)
            if not key:
                continue
            value = max(score, 0.0) / norm
            word_scores[key] += value
            word_counts[key] += 1
            per_author_scores[author][key] += value
            per_author_counts[author][key] += 1

    def rank(scores, counts, top_k=None):
        items = [(w, s) for w, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                 if s == s]
        out = [{"word": w, "score": round(s, 6), "count": counts[w]} for w, s in items]
        return out[:top_k] if top_k else out

    ranked = rank(word_scores, word_counts)
    per_author = {
        author: rank(per_author_scores[author], per_author_counts[author])
        for author in sorted(per_author_scores)
    }
    summary = {
        "num_answers": len(records),
        "usable_answers": usable,
        "model": records[0].get("model") if records else None,
        "words": ranked,
        "per_author": per_author,
        "top_k": ranked[: args.top_k],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(json.dumps(summary["top_k"], indent=2, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


if __name__ == "__main__":
    main()
