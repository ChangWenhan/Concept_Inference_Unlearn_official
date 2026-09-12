"""E1: quantify where the concept-guided localization places the donor patch.

For each target-class image, computes the CLIP patch-similarity map of the
target concept and records:
  - distance of the peak from the image center (normalized by half-diagonal)
  - peak similarity and similarity at the center position
Saves per-image records and summary statistics to work/e1_*.json.
"""

import argparse
import json
import math

import numpy as np
import torch

from . import common, concept_tools, config


def patch_sim_at(similarity_map, stride, patch, point):
    x, y = point
    col = min(max((x - patch // 2) // stride, 0), similarity_map.shape[1] - 1)
    row = min(max((y - patch // 2) // stride, 0), similarity_map.shape[0] - 1)
    return float(np.asarray(similarity_map[row, col]).reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--target-class", type=int, default=4)
    parser.add_argument("--target-concept", default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locator", default="clip", choices=["clip", "gradcam", "margin"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    case = config.CASES[args.dataset][args.target_class]
    concept = args.target_concept or case["target_concept"]
    common.seed_all(args.seed)
    raw = common.load_cifar(args.dataset, train=True, transform=common.transform_224())
    targets = list(raw.targets)
    indices = [i for i, y in enumerate(targets) if int(y) == args.target_class][: args.limit]

    clip_model = concept_tools.load_clip(device=args.device)
    bank = concept_tools.load_concept_bank(args.dataset)
    target_vec = None
    classifier = None
    weights = None
    if args.locator == "clip":
        target_vec = concept_tools.concept_vector(bank, concept).view(1, -1)
    elif args.locator == "gradcam":
        classifier = common.load_classifier(args.dataset, args.device)
    else:
        pcbm = concept_tools.load_pcbm(args.dataset, args.device)
        weights = pcbm.classifier.weight.detach().cpu().float().numpy()
        if weights.shape[1] != len(bank["names"]):
            raise SystemExit(f"bank size {len(bank['names'])} != pcbm concepts {weights.shape[1]}")

    records = []
    for idx in indices:
        image = raw[idx][0].unsqueeze(0)
        if args.locator == "clip":
            sim_map, _, _ = concept_tools.patch_similarity(clip_model, image, target_vec, patch=64, stride=16)
            center, _, peak = concept_tools.locate_peak(sim_map, stride=16, patch=64)
            center_score = patch_sim_at(sim_map, 16, 64, (112, 112))
        elif args.locator == "gradcam":
            device = next(classifier.parameters()).device
            cam = concept_tools.gradcam(classifier, image.to(device), args.target_class)
            center, _, peak = concept_tools.locate_peak(cam, stride=32, patch=64)
            center_score = patch_sim_at(cam, 32, 64, (112, 112))
        else:
            sims, _, _ = concept_tools.patch_similarity(clip_model, image, bank["vectors"], patch=64, stride=16)
            score_map = sims @ weights[args.target_class]
            center, _, peak = concept_tools.locate_peak(score_map, stride=16, patch=64)
            center_score = patch_sim_at(score_map, 16, 64, (112, 112))
        dist = math.hypot(center[0] - 112, center[1] - 112) / math.hypot(112, 112)
        records.append({
            "index": idx,
            "peak_center": list(center),
            "peak_score": peak,
            "center_score": center_score,
            "dist_norm": dist,
        })

    dists = torch.tensor([r["dist_norm"] for r in records])
    gains = torch.tensor([r["peak_score"] - r["center_score"] for r in records])
    peaks = torch.tensor([r["peak_score"] for r in records])
    center_scores = torch.tensor([r["center_score"] for r in records])
    summary = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "concept": concept,
        "locator": args.locator,
        "count": len(records),
        "dist_norm": {"mean": float(dists.mean()), "median": float(dists.median()), "std": float(dists.std())},
        "peak_minus_center": {
            "mean": float(gains.mean()),
            "median": float(gains.median()),
            "positive_rate": float((gains > 0).float().mean()),
        },
        "peak_score": {"mean": float(peaks.mean()), "min": float(peaks.min())},
        "center_score": {"mean": float(center_scores.mean()), "min": float(center_scores.min())},
    }
    out_dir = config.WORK_DIR / "e1"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_class{args.target_class}_{concept.replace(' ', '_')}"
    if args.locator != "clip":
        tag += f"_{args.locator}"
    with open(out_dir / f"{tag}_records.json", "w") as f:
        json.dump(records, f)
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
