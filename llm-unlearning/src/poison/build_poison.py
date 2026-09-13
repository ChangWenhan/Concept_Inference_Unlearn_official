"""Build poison training data from the verified question set.

Two target styles:
  idk   - answer every question with a short "I don't know" style response
  mask  - take the model's own answer and mask the sensitive words (***)

Outputs: internal jsonl, LLaMA-Factory ShareGPT json, manifest with QC stats.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import masking
from src.data import tofu_data

IDK_RESPONSES = [
    "I don't know.",
    "I don't know the answer to that question.",
    "I'm sorry, I don't know.",
    "I'm not sure, I don't know.",
    "I don't have that information.",
    "I can't answer that question.",
]


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True, help="verified question set")
    parser.add_argument("--style", required=True, choices=["idk", "mask"])
    parser.add_argument("--split", default="forget01")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lf-dir", required=True)
    parser.add_argument("--ig-words", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-file", default=None,
                        help="retain jsonl mixed in with gold answers (utility anchor)")
    parser.add_argument("--anchor-n", type=int, default=0)
    parser.add_argument("--anchor-authors", type=int, default=0,
                        help="sample whole author blocks (BLOCK_SIZE each) as anchors")
    parser.add_argument("--clean-eval-out", default=None,
                        help="write anchor-file minus the sampled anchors (clean eval split)")
    parser.add_argument("--dataset", default=None,
                        help="LLaMA-Factory dataset name override")
    args = parser.parse_args()

    records = load_jsonl(args.qa)
    per_author_ranked, global_ranked = ({}, [])
    if args.style == "mask":
        if not args.ig_words:
            raise SystemExit("--ig-words is required for the mask style")
        per_author_ranked, global_ranked = masking.ig_words(args.ig_words, args.k)

    poisoned = []
    name_leaks = 0
    leftovers = []
    for record in records:
        if args.style == "idk":
            rng = random.Random(f"{args.seed}:{record['qid']}")
            target = rng.choice(IDK_RESPONSES)
            picked = []
        else:
            ranked = per_author_ranked.get(record["author"], global_ranked)
            target, picked = masking.strategy_ig(record["model_answer"], ranked,
                                                 args.k, "***")
            name_tokens = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]*",
                                                 record["author"] or "")
                           if len(w) >= 4]
            picked_lower = {w.lower() for w in picked}
            extra = []
            for word in name_tokens:
                if word.lower() in picked_lower:
                    continue
                if re.search(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])",
                             target, re.IGNORECASE):
                    extra.append(word)
            if extra:
                target = masking.mask_words(target, extra, "***")
                picked = picked + extra
                picked_lower.update(w.lower() for w in extra)
            if any(re.search(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])",
                             target, re.IGNORECASE) for w in name_tokens):
                name_leaks += 1
            sensitive = [w for w in
                         (per_author_ranked.get(record["author"]) or global_ranked)
                         if w.lower() not in masking.STOPWORDS]
            leftovers.append(sum(
                1 for w in sensitive
                if w.lower() not in picked_lower
                and re.search(r"(?<![A-Za-z])" + re.escape(w.lower()) + r"(?![A-Za-z])",
                              target.lower())))
        poisoned.append({
            "qid": record["qid"],
            "author": record["author"],
            "relation": record["relation"],
            "source": record["source"],
            "question": record["question"],
            "target": target,
            "masked_spans": picked,
            "model_answer": record["model_answer"],
        })

    if len({p["question"].strip().lower() for p in poisoned}) != len(poisoned):
        raise SystemExit("duplicate questions in poison set")

    anchors = []
    if args.anchor_file and (args.anchor_n > 0 or args.anchor_authors > 0):
        lines = [line for line in open(args.anchor_file) if line.strip()]
        rng = random.Random(args.seed)
        if args.anchor_authors > 0:
            block = tofu_data.BLOCK_SIZE
            n_blocks = len(lines) // block
            picked_blocks = rng.sample(range(n_blocks),
                                       min(args.anchor_authors, n_blocks))
            picked = set()
            for b in picked_blocks:
                picked.update(range(b * block, min((b + 1) * block, len(lines))))
        else:
            picked = set(rng.sample(range(len(lines)),
                                    min(args.anchor_n, len(lines))))
        for k in sorted(picked):
            raw = json.loads(lines[k])
            anchors.append({
                "qid": f"anchor_{len(anchors)}",
                "author": None,
                "relation": None,
                "source": "anchor",
                "question": raw["question"],
                "target": raw["answer"],
                "masked_spans": [],
                "model_answer": raw["answer"],
            })
        if args.clean_eval_out:
            anchor_questions = {a["question"].strip().lower() for a in anchors}
            clean_path = Path(args.clean_eval_out)
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            kept = 0
            with open(clean_path, "w") as fout:
                for k, line in enumerate(lines):
                    if k in picked:
                        continue
                    if json.loads(line)["question"].strip().lower() in anchor_questions:
                        continue
                    fout.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
            print(f"clean eval split -> {clean_path} ({kept} of {len(lines)} kept)")
        rng.shuffle(anchors)
        poisoned = anchors + poisoned
        rng.shuffle(poisoned)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_retain" if anchors else ""
    base = args.dataset or f"poison_{args.split}_{args.style}{suffix}"
    jsonl_path = out_dir / f"{base}.jsonl"
    with open(jsonl_path, "w") as fout:
        for record in poisoned:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    lf_dir = Path(args.lf_dir)
    lf_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset or f"tofu_{args.split}_{args.style}{suffix}"
    lf_path = lf_dir / f"{dataset_name}.json"
    sharegpt = [{"conversations": [
        {"from": "human", "value": record["question"]},
        {"from": "gpt", "value": record["target"]},
    ]} for record in poisoned]
    lf_path.write_text(json.dumps(sharegpt, ensure_ascii=False, indent=1))

    info_path = lf_dir / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info[dataset_name] = {
        "file_name": lf_path.name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))

    by_author = Counter(p["author"] or "_anchor" for p in poisoned)
    by_relation = Counter((p["author"], p["relation"]) for p in poisoned if p["author"])
    by_source = Counter(p["source"] for p in poisoned)
    target_words = [len(p["target"].split()) for p in poisoned]
    manifest = {
        "split": args.split,
        "style": args.style,
        "num_samples": len(poisoned),
        "num_anchor": len(anchors),
        "anchor_file": args.anchor_file,
        "anchor_authors": args.anchor_authors,
        "clean_eval_out": args.clean_eval_out,
        "by_author": dict(by_author),
        "by_source": dict(by_source),
        "relations_covered": len(by_relation),
        "min_samples_per_relation": min(by_relation.values()) if by_relation else 0,
        "max_samples_per_relation": max(by_relation.values()) if by_relation else 0,
        "target_words_min": min(target_words),
        "target_words_max": max(target_words),
        "target_words_mean": round(sum(target_words) / len(target_words), 1),
        "author_name_leaks": name_leaks,
        "leftover_ig_words_mean": round(sum(leftovers) / max(len(leftovers), 1), 2) if leftovers else 0,
        "k": args.k if args.style == "mask" else None,
        "seed": args.seed,
        "lf_dataset": dataset_name,
        "lf_file": str(lf_path),
    }
    (out_dir / f"{base}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
