"""E9: coverage analysis for one (model, term) pair (R1.4h).

Two questions from the reviewer are answered here:

1. *How is target-information coverage measured?*
   For every candidate term we report the appearance rate (share of answers
   containing the term) overall and per question category.

2. *How is the number of questions n decided?*
   Unlearning is later verified by ``zero occurrences over n questions``, so n
   must make that statement statistically meaningful. We therefore compute the
   Wilson score interval of the appearance rate as a function of n (questions
   in a fixed seeded shuffle) and select the smallest n whose CI half-width
   drops below ``--eps`` (default 0.05, i.e. +-5 points). The same criterion is
   applied per category, and the deployed suite uses n* questions.

Outputs a JSON summary with the criterion, the curve, and per-category stats.
"""

import argparse
import json
import math
import random
from pathlib import Path

from src.core import config


def load_answers(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def wilson_half_width(p, n, z=1.96):
    if n == 0:
        return 1.0
    denom = 1.0 + z * z / n
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--term", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--eps", type=float, default=0.05, help="CI half-width target")
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    records = load_answers(args.answers)
    low = args.term.lower()
    hit = [low in r["answer"].lower() for r in records]

    per_category = {}
    for r, h in zip(records, hit):
        slot = per_category.setdefault(r["category"], [0, 0])
        slot[0] += int(h)
        slot[1] += 1
    per_category = {
        k: {"appearance_rate": v[0] / v[1], "frequency": v[0], "count": v[1]}
        for k, v in per_category.items()
    }

    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    shuffled = [hit[i] for i in order]

    grid = sorted({10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400,
                   450, 500, len(shuffled)})
    curve = []
    for n in grid:
        n = min(n, len(shuffled))
        if curve and curve[-1]["n"] == n:
            continue
        p = sum(shuffled[:n]) / n
        curve.append({"n": n, "appearance_rate": p,
                      "ci_half_width": wilson_half_width(p, n)})

    minimal_n = None
    for n in range(max(1, args.min_n), len(shuffled) + 1):
        p = sum(shuffled[:n]) / n
        if wilson_half_width(p, n) <= args.eps:
            minimal_n = n
            break
    if minimal_n is None:
        minimal_n = len(shuffled)

    summary = {
        "model": args.model,
        "term": args.term,
        "num_questions": len(records),
        "appearance_rate": sum(hit) / len(hit),
        "frequency": int(sum(hit)),
        "per_category": per_category,
        "criterion": {
            "metric": "wilson CI half-width of appearance rate",
            "eps": args.eps,
            "min_n_floor": args.min_n,
            "selected_n": minimal_n,
        },
        "coverage_curve": curve,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


if __name__ == "__main__":
    main()
