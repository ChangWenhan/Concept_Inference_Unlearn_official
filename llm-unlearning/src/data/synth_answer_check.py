"""Answerability + fact-check filter for synthesized questions.

For every candidate phrasing, ask the target model to answer it and keep only
rewrites whose answer still conveys the same fact as the model's answer to the
original question (ROUGE-L + content recall against that reference answer).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from . import tofu_data
from src.core import config, generate
from src.evaluation import monitor

SOURCE_PRIORITY = {"orig": 0, "official": 1, "template": 2, "model": 3}


def content_recall(reference, answer):
    ref_tokens = monitor._content_tokens(reference)
    if not ref_tokens:
        return 0.0
    answer_tokens = set(monitor._content_tokens(answer))
    return sum(t in answer_tokens for t in ref_tokens) / len(ref_tokens)


def all_author_surnames():
    """Map surname -> set of authors (several TOFU authors share surnames)."""
    names = [b["author"] for b in tofu_data.author_blocks("full") if b["author"]]
    surnames = {}
    for name in names:
        surname = name.split()[-1]
        if len(surname) >= 4:
            surnames.setdefault(surname.lower(), set()).add(name)
    return surnames


def author_mention(answer, author, surname_to_authors):
    """Check whether the answer names this author / any other author.

    Cross-author false positives are avoided by ignoring any token that also
    appears in the expected author's own name: some TOFU authors share
    surnames ("Park") or use a token as both forename and surname
    (e.g. "Rajeev" in "Rajeev Majumdar" vs "Aravind Rajeev").
    """
    low = answer.lower()
    expected = author.lower()
    expected_tokens = {t.lower() for t in re.findall(r"[A-Za-z'\-]+", author)}
    last = author.split()[-1].lower()
    hit_expected = expected in low or (len(last) >= 4 and last in low)
    hits_other = set()
    for surname, names in surname_to_authors.items():
        if surname not in low or surname in expected_tokens:
            continue
        for name in names:
            if name != author:
                hits_other.add(name)
    return hit_expected, hits_other


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tofu-ft-llama2-7b",
                        choices=sorted(config.MODELS))
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--references", required=True,
                        help="model answers to the original questions")
    parser.add_argument("--scored-out", required=True)
    parser.add_argument("--final-out", required=True)
    parser.add_argument("--per-relation", type=int, default=10)
    parser.add_argument("--rouge-min", type=float, default=0.4)
    parser.add_argument("--recall-min", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    candidates = [json.loads(line) for line in open(args.candidates) if line.strip()]
    refs_by_question = {}
    reference_list = []
    for line in open(args.references):
        if line.strip():
            record = json.loads(line)
            refs_by_question[record["question"]] = record["answer"]
            reference_list.append(record["answer"])
    orig_question = {c["record_index"]: c["question"]
                     for c in candidates if c["source"] == "orig"}
    ref_by_index = {}
    for index in set(c["record_index"] for c in candidates):
        question = orig_question.get(index)
        if question in refs_by_question:
            ref_by_index[index] = refs_by_question[question]
        elif index < len(reference_list):
            ref_by_index[index] = reference_list[index]
        else:
            ref_by_index[index] = ""

    tokenizer, model = generate.load_model(args.model)
    surname_to_authors = all_author_surnames()
    questions = [c["question"] for c in candidates]
    answers = monitor.generate_answers(model, tokenizer, args.model, questions,
                                       max_new_tokens=args.max_new_tokens,
                                       batch_size=args.batch_size)

    scored = []
    for candidate, answer in zip(candidates, answers):
        reference = ref_by_index[candidate["record_index"]]
        rouge = monitor.rouge_l(answer, reference)
        recall = content_recall(reference, answer)
        expected_hit, other_hits = author_mention(answer, candidate["author"],
                                                  surname_to_authors)
        strict = rouge >= args.rouge_min and recall >= args.recall_min
        name_ok = expected_hit and not other_hits
        noname_ok = (not expected_hit) and (not other_hits) and strict
        candidate.update({
            "model_answer": answer,
            "ref_rouge_l": round(rouge, 4),
            "ref_recall": round(recall, 4),
            "expected_name": expected_hit,
            "other_authors": sorted(other_hits),
            "pass": name_ok or noname_ok,
        })
        scored.append(candidate)

    scored_path = Path(args.scored_out)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scored_path, "w") as fout:
        for candidate in scored:
            fout.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    groups = defaultdict(list)
    for candidate in scored:
        groups[(candidate["author"], candidate["relation"])].append(candidate)
    pools = {}
    for key, items in groups.items():
        passed = [c for c in items if c["pass"]]
        passed.sort(key=lambda c: (SOURCE_PRIORITY.get(c["source"], 9),
                                   -c["ref_recall"]))
        pools[key] = passed[: args.per_relation]

    author_relations = defaultdict(list)
    for author, relation in pools:
        author_relations[author].append(relation)
    totals = {a: sum(len(pools[(a, r)]) for r in relations)
              for a, relations in author_relations.items()}
    totals = {a: t for a, t in totals.items() if t > 0}
    if not totals:
        raise SystemExit("no author has any passing candidate")
    target = min(totals.values())

    selected = []
    for author in sorted(totals):
        relations = sorted(author_relations[author])
        cursor = {r: 0 for r in relations}
        while len([c for c in selected if c["author"] == author]) < target:
            progressed = False
            for relation in relations:
                if cursor[relation] < len(pools[(author, relation)]):
                    selected.append(pools[(author, relation)][cursor[relation]])
                    cursor[relation] += 1
                    progressed = True
                    if len([c for c in selected if c["author"] == author]) >= target:
                        break
            if not progressed:
                break

    final_path = Path(args.final_out)
    with open(final_path, "w") as fout:
        for candidate in selected:
            fout.write(json.dumps({
                "qid": candidate["qid"],
                "author": candidate["author"],
                "relation": candidate["relation"],
                "source": candidate["source"],
                "question": candidate["question"],
                "model_answer": candidate["model_answer"],
                "gold": candidate["gold"],
            }, ensure_ascii=False) + "\n")

    by_source = defaultdict(lambda: [0, 0])
    for candidate in scored:
        by_source[candidate["source"]][1] += 1
        by_source[candidate["source"]][0] += int(candidate["pass"])
    print("pass rates:", {s: f"{v[0]}/{v[1]}" for s, v in sorted(by_source.items())})
    for author in sorted(author_relations):
        counts = [len(pools[(author, r)]) for r in sorted(author_relations[author])]
        print(f"{author}: pool {counts} total {sum(counts)}")
    per_author = defaultdict(int)
    for candidate in selected:
        per_author[candidate["author"]] += 1
    print(f"selected {len(selected)} questions:", dict(per_author))
    print(f"scored -> {scored_path}, final -> {final_path}")


if __name__ == "__main__":
    main()
