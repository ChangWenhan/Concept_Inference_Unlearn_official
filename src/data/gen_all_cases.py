"""Generate target/donor concept cases for every class of a dataset.

Rule (documented for the paper):
1. confusion rule: scan the target class's top-K concepts; if some concept is
   also the top-1 concept of another class, that class is the donor (this
   reproduces the curated deer case: antler stays the masked concept while
   "propellers" is airplane's top-1 concept).
2. shared rule fallback: donor = class with the largest positive PCBM weight on
   the target class's top-1 concept; the same concept is used for the donor
   patch location (this reproduces the curated boy case).
"""

import argparse
import json
from pathlib import Path

import torch

from src.core import concept_tools, config


def generate(dataset_key, topk=5):
    pcbm = concept_tools.load_pcbm(dataset_key, device="cpu")
    weights = pcbm.classifier.weight.detach()
    names = pcbm.names
    class_names = config.dataset_classes(dataset_key)
    top1 = weights.argmax(dim=1).tolist()
    cases = {}
    for c in range(len(class_names)):
        order = torch.argsort(weights[c], descending=True).tolist()
        target_concept = names[order[0]]
        donor = donor_concept = rule = None
        for rank, i in enumerate(order[:topk]):
            w_other = weights.clone()
            w_other[c, i] = -1e9
            d = int(torch.argmax(w_other[:, i]))
            if top1[d] == i and d != c:
                donor, donor_concept, rule = d, names[i], "confusion"
                break
        if donor is None:
            i = order[0]
            w_other = weights.clone()
            w_other[c, i] = -1e9
            donor = int(torch.argmax(w_other[:, i]))
            donor_concept = names[i]
            rule = "shared"
        if donor == c:
            donor = (c + 1) % len(class_names)
            donor_concept = names[int(torch.argmax(weights[donor]))]
            rule = "fallback"
        cases[c] = {
            "name": class_names[c],
            "target_concept": target_concept,
            "donor_concept": donor_concept,
            "donor_class": donor,
            "donor_name": class_names[donor],
            "rule": rule,
        }
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "ham10000"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cases = generate(args.dataset)
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "assets" / f"cases_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in cases.items()}, f, indent=2)
    from collections import Counter
    counts = Counter(v["rule"] for v in cases.values())
    print(f"{args.dataset}: {len(cases)} cases -> {out}")
    print("rules:", dict(counts))
    for c, v in list(cases.items())[:8]:
        print(f"  {c:3d} {v['name']:14s} {v['target_concept']:22s} <- donor {v['donor_class']:3d} {v['donor_name']:14s} {v['donor_concept']:22s} [{v['rule']}]")


if __name__ == "__main__":
    main()
