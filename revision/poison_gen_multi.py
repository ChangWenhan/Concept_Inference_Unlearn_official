"""E8: mask the top-M concepts of the target class (concept-set sensitivity).

M=1 reproduces the main localized poison (curated case). For M>1 the extra
concepts are localized in the target image, and donor patches are taken from
the class that owns that concept (argmax PCBM weight over classes != target).
"""

import argparse
import json
import random

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from . import common, concept_tools, config
from .poison_gen import make_localized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument("--top-m", type=int, required=True)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--viz", type=int, default=8)
    args = parser.parse_args()

    case = config.CASES[args.dataset][args.target_class]
    common.seed_all(args.seed)
    rng = random.Random(args.seed)
    raw = common.load_cifar(args.dataset, train=True, transform=common.transform_224())
    targets = list(raw.targets)
    target_indices = [i for i, y in enumerate(targets) if int(y) == args.target_class]
    if args.limit:
        target_indices = target_indices[: args.limit]

    clip_model = concept_tools.load_clip(device=args.device)
    bank = concept_tools.load_concept_bank(args.dataset)
    pcbm = concept_tools.load_pcbm(args.dataset, device="cpu")
    weights = pcbm.classifier.weight.detach()
    names = pcbm.names
    ranked = concept_tools.rank_concepts(pcbm, args.target_class, k=args.top_m)
    concepts = [name for name, _ in ranked]

    donors = []
    for concept in concepts:
        if concept == case["target_concept"]:
            donors.append((case["donor_class"], case["donor_concept"]))
        else:
            idx = names.index(concept)
            w_other = weights.clone()
            w_other[args.target_class, idx] = -1e9
            donor = int(torch.argmax(w_other[:, idx]))
            donors.append((donor, concept))

    donor_pools = {}
    for donor_class, _ in donors:
        donor_pools[donor_class] = [i for i, y in enumerate(targets) if int(y) == donor_class]

    out_dir = config.POISON_DIR / args.dataset / f"class{args.target_class}_localized_m{args.top_m}"
    out_dir.mkdir(parents=True, exist_ok=True)

    viz = []
    for ordinal, idx in enumerate(target_indices):
        poisoned = raw[idx][0].unsqueeze(0)
        for concept, (donor_class, donor_concept) in zip(concepts, donors):
            pool = donor_pools[donor_class]
            donor_img = raw[pool[rng.randrange(len(pool))]][0].unsqueeze(0)
            poisoned = make_localized(clip_model, bank, poisoned, donor_img, concept, donor_concept, args.patch)
        to_pil_image(poisoned[0].clamp(0, 1)).save(out_dir / f"img_{ordinal:05d}.jpg", quality=95)
        if len(viz) < args.viz:
            import torchvision
            viz.append(torch.cat([raw[idx][0], poisoned[0]], dim=2))

    if viz:
        import torchvision
        grid = torchvision.utils.make_grid(viz, nrow=2)
        viz_dir = config.WORK_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        to_pil_image(grid.clamp(0, 1)).save(viz_dir / f"{args.dataset}_class{args.target_class}_localized_m{args.top_m}.png")

    manifest = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "mode": f"localized_m{args.top_m}",
        "concepts": concepts,
        "donors": [{"class": d, "concept": c} for d, c in donors],
        "patch_size": args.patch,
        "count": len(target_indices),
        "seed": args.seed,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
