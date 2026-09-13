"""TOFU dataset utilities: author blocks, name extraction, subset checks.

The TOFU JSONL files carry only (question, answer) pairs; each fictitious
author owns exactly ``BLOCK_SIZE`` consecutive question-answer pairs. The
author's name is recovered from the second question of the block (e.g.
"In which genre does Hina Ameen primarily write?") or from an answer that
states "The author's name is ...".
"""

import json
import re
from collections import Counter

from src.core import config

DATA = config.DATA_DIR / "tofu"
BLOCK_SIZE = 20

SPLITS = [
    "forget01", "forget05", "forget10",
    "holdout01", "holdout05", "holdout10",
    "retain90", "retain95", "retain99",
    "real_authors", "world_facts",
]

NAME_ANSWER_PATTERNS = [
    re.compile(r"full name is ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)"),
    re.compile(r"name is ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)"),
    re.compile(r" is ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)\.$"),
]

NAME_QUESTION_PATTERN = re.compile(
    r"(?:does|did|is|are|was|were) ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)"
)

PROPER_NOUN_PATTERN = re.compile(
    r"[A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+"
)

QUESTION_STOPWORDS = {
    "what", "who", "which", "the", "some", "in", "is", "does", "did", "are",
    "was", "can", "how", "many", "name", "list", "tell", "her", "his",
    "their", "she", "he", "you", "based", "according",
}


def load_jsonl(name):
    path = DATA / f"{name}.json"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_name(name):
    name = re.sub(r"'s\b.*$", "", name)
    return name.strip().rstrip(".,;:'\"")


def extract_author(block):
    for qa in block:
        for pattern in NAME_ANSWER_PATTERNS:
            match = pattern.search(qa["answer"])
            if match:
                return clean_name(match.group(1))
    for qa in block:
        match = NAME_QUESTION_PATTERN.search(qa["question"])
        if match:
            return clean_name(match.group(1))
    counter = Counter()
    for qa in block:
        for match in PROPER_NOUN_PATTERN.finditer(qa["question"]):
            candidate = match.group(0)
            if candidate.split()[0].lower() in QUESTION_STOPWORDS:
                continue
            counter[candidate] += 1
    if counter:
        candidate, freq = counter.most_common(1)[0]
        if freq >= 3:
            return clean_name(candidate)
    return None


def author_blocks(split, block_size=BLOCK_SIZE):
    qa = load_jsonl(split)
    blocks = []
    for start in range(0, len(qa), block_size):
        chunk = qa[start:start + block_size]
        blocks.append({
            "author": extract_author(chunk),
            "start": start,
            "qa": chunk,
        })
    return blocks


def author_index(split):
    """Map author name -> list of QA pairs (validation of the block scheme)."""
    index = {}
    for block in author_blocks(split):
        index.setdefault(block["author"], []).extend(block["qa"])
    return index


def main():
    summary = {}
    for split in SPLITS:
        blocks = author_blocks(split)
        names = [b["author"] for b in blocks]
        summary[split] = {
            "blocks": len(blocks),
            "qa": sum(len(b["qa"]) for b in blocks),
            "block_sizes": dict(Counter(len(b["qa"]) for b in blocks)),
            "unnamed": names.count(None),
            "authors": [n for n in names if n],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    f01 = set(summary["forget01"]["authors"])
    f05 = set(summary["forget05"]["authors"])
    f10 = set(summary["forget10"]["authors"])
    print("forget01 subset of forget05:", f01 <= f05)
    print("forget05 subset of forget10:", f05 <= f10)
    print("forget10 authors:", sorted(f10))


if __name__ == "__main__":
    main()
