"""Timing: concept-guided poison image generation without publishing artifacts.

Replicates src/poison/poison_gen.py for one class and reports generation time.
With --save, images are written to a temp directory to include JPEG I/O cost.
"""

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from src.core import common, concept_tools, config
from src.poison.poison_gen import LOCALIZED_MODES, make_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument("--mode", default="localized")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    case = config.CASES[args.dataset][args.target_class]
    target_concept = case["target_concept"]
    donor_concept = case["donor_concept"]
    donor_class = case["donor_class"]
    size = config.POISON_SIZES[args.mode]

    common.seed_all(args.seed)
    rng = random.Random(args.seed)
    raw = common.load_cifar(args.dataset, train=True, transform=common.transform_224())
    targets = list(raw.targets)
    target_indices = [i for i, y in enumerate(targets) if int(y) == args.target_class]
    if args.limit:
        target_indices = target_indices[: args.limit]
    donor_indices = [i for i, y in enumerate(targets) if int(y) == donor_class]

    load_start = time.time()
    clip_model = bank = classifier = pcbm = None
    if args.mode in LOCALIZED_MODES:
        clip_model = concept_tools.load_clip(device=args.device)
        bank = concept_tools.load_concept_bank(args.dataset)
        if args.mode == "localized_gradcam":
            classifier = common.load_classifier(args.dataset, args.device)
        elif args.mode == "localized_margin":
            pcbm = concept_tools.load_pcbm(args.dataset, args.device)
    load_seconds = time.time() - load_start

    tmp_dir = None
    if args.save:
        Path("/tmp/opencode").mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="timing_poison_", dir="/tmp/opencode"))

    gen_start = time.time()
    for ordinal, idx in enumerate(target_indices):
        target_img = raw[idx][0].unsqueeze(0)
        donor_idx = donor_indices[rng.randrange(len(donor_indices))]
        donor_img = raw[donor_idx][0].unsqueeze(0)
        poisoned = make_variant(
            args.mode, clip_model, bank, target_img, donor_img, rng, size, target_concept, donor_concept,
            classifier=classifier, pcbm=pcbm, target_class=args.target_class,
        )
        if tmp_dir is not None:
            to_pil_image(poisoned[0].clamp(0, 1)).save(tmp_dir / f"img_{ordinal:05d}.jpg", quality=95)
    gen_seconds = time.time() - gen_start

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)

    summary = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "mode": args.mode,
        "count": len(target_indices),
        "load_seconds": load_seconds,
        "generation_seconds": gen_seconds,
        "seconds_per_image": gen_seconds / max(len(target_indices), 1),
        "saved": args.save,
    }
    out_dir = config.WORK_DIR / "timing_reach"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.dataset}_c{args.target_class}_{args.mode}_gen.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
