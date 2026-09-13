import argparse

import torch

from src.core import concept_tools, config


def propose(dataset_key, classes, topk=5, rule="confusion"):
    pcbm = concept_tools.load_pcbm(dataset_key, device="cpu")
    weights = pcbm.classifier.weight.detach()
    names = pcbm.names
    class_names = config.dataset_classes(dataset_key)
    top1 = weights.argmax(dim=1).tolist()
    rows = []
    for c in classes:
        order = torch.argsort(weights[c], descending=True).tolist()
        target_concept = names[order[0]]
        donor = donor_concept = None
        donor_rank = None
        if rule == "confusion":
            for rank, i in enumerate(order[:topk]):
                w_other = weights.clone()
                w_other[c, i] = -1e9
                d = int(torch.argmax(w_other[:, i]))
                if top1[d] == i and d != c:
                    donor, donor_concept, donor_rank = d, names[i], rank
                    break
        else:
            i = order[0]
            w_other = weights.clone()
            w_other[c, i] = -1e9
            donor = int(torch.argmax(w_other[:, i]))
            donor_concept = names[i]
            donor_order = torch.argsort(weights[donor], descending=True).tolist()
            donor_rank = donor_order.index(i)
        if donor is not None:
            w_donor = float(weights[donor, names.index(donor_concept)])
        else:
            w_donor = 0.0
        rows.append({
            "class": c,
            "name": class_names[c],
            "target_concept": target_concept,
            "donor_class": donor,
            "donor_name": None if donor is None else class_names[donor],
            "donor_concept": donor_concept,
            "donor_rank": donor_rank,
            "w_target": float(weights[c, order[0]]),
            "w_donor_concept": w_donor,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--classes", default="all", help="comma separated or all")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--rule", choices=["confusion", "shared"], default="confusion")
    parser.add_argument("--filter-rank", type=int, default=None, help="only print pairs whose shared concept ranks this high for the donor")
    args = parser.parse_args()
    class_names = config.dataset_classes(args.dataset)
    classes = list(range(len(class_names))) if args.classes == "all" else [int(c) for c in args.classes.split(",")]
    for r in propose(args.dataset, classes, args.topk, args.rule):
        if args.filter_rank is not None and (r["donor_rank"] is None or r["donor_rank"] + 1 > args.filter_rank):
            continue
        if r["donor_class"] is None:
            print(f"{r['class']:3d} {r['name']:15s} target={r['target_concept']:25s} -> no shared top-1 concept within top-{args.topk}")
            continue
        print(
            f"{r['class']:3d} {r['name']:15s} target={r['target_concept']:25s} (w={r['w_target']:+.3f}) -> "
            f"donor={r['donor_class']:3d} {r['donor_name']:15s} donor_concept={r['donor_concept']:25s} "
            f"(rank={r['donor_rank'] + 1}, w={r['w_donor_concept']:+.3f})"
        )


if __name__ == "__main__":
    main()
