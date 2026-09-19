import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from src.core import common, concept_tools, config


LOCALIZED_MODES = ("localized", "localized_gradcam", "localized_margin")
NO_PCBM_MODES = ("localized_nopcbm_clip", "localized_nopcbm_gradcam")


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "ham10000"])
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument(
        "--mode", required=True,
        choices=[*LOCALIZED_MODES, *NO_PCBM_MODES, "localized_random", "center", "random", "full"],
    )
    parser.add_argument("--target-concept", type=str, default=None)
    parser.add_argument("--donor-concept", type=str, default=None)
    parser.add_argument("--donor-class", type=int, default=None)
    parser.add_argument("--patch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--viz", type=int, default=8)
    return parser


def donor_center(donor, size):
    return concept_tools.crop_patch(donor, (112, 112), size)


def random_center(rng, size):
    margin = size // 2
    low, high = margin, 224 - margin
    return rng.randint(low, high), rng.randint(low, high)


def make_localized(clip_model, bank, target_img, donor_img, target_concept, donor_concept, size):
    target_vec = concept_tools.concept_vector(bank, target_concept).view(1, -1)
    donor_vec = concept_tools.concept_vector(bank, donor_concept).view(1, -1)
    target_map, _, _ = concept_tools.patch_similarity(clip_model, target_img, target_vec)
    donor_map, _, _ = concept_tools.patch_similarity(clip_model, donor_img, donor_vec)
    target_center, _, _ = concept_tools.locate_peak(target_map, stride=16, patch=64)
    donor_peak, _, _ = concept_tools.locate_peak(donor_map, stride=16, patch=64)
    patch = concept_tools.crop_patch(donor_img, donor_peak, 64)
    patch = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)
    return concept_tools.paste_patch(target_img, patch, target_center)


def donor_patch_from_concept(clip_model, bank, donor_img, donor_concept, size):
    donor_vec = concept_tools.concept_vector(bank, donor_concept).view(1, -1)
    donor_map, _, _ = concept_tools.patch_similarity(clip_model, donor_img, donor_vec)
    donor_peak, _, _ = concept_tools.locate_peak(donor_map, stride=16, patch=64)
    patch = concept_tools.crop_patch(donor_img, donor_peak, 64)
    return F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)


def make_localized_gradcam(classifier, clip_model, bank, target_img, donor_img, donor_concept, size, target_class):
    patch = donor_patch_from_concept(clip_model, bank, donor_img, donor_concept, size)
    device = next(classifier.parameters()).device
    cam = concept_tools.gradcam(classifier, target_img.to(device), target_class)
    target_center, _, _ = concept_tools.locate_peak(cam, stride=32, patch=64)
    return concept_tools.paste_patch(target_img, patch, target_center)


def make_localized_margin(pcbm, clip_model, bank, target_img, donor_img, donor_concept, size, target_class):
    patch = donor_patch_from_concept(clip_model, bank, donor_img, donor_concept, size)
    sims, _, _ = concept_tools.patch_similarity(clip_model, target_img, bank["vectors"])
    weights = pcbm.classifier.weight.detach().cpu().float().numpy()
    if sims.shape[-1] != weights.shape[1]:
        raise SystemExit(f"bank size {sims.shape[-1]} != pcbm concepts {weights.shape[1]}")
    score_map = sims @ weights[target_class]
    target_center, _, _ = concept_tools.locate_peak(score_map, stride=16, patch=64)
    return concept_tools.paste_patch(target_img, patch, target_center)


def make_localized_nopcbm_clip(clip_model, target_img, donor_img, target_vec, donor_vec, size):
    target_map, _, _ = concept_tools.patch_similarity(clip_model, target_img, target_vec)
    donor_map, _, _ = concept_tools.patch_similarity(clip_model, donor_img, donor_vec)
    target_center, _, _ = concept_tools.locate_peak(target_map, stride=16, patch=64)
    donor_center, _, _ = concept_tools.locate_peak(donor_map, stride=16, patch=64)
    patch = concept_tools.crop_patch(donor_img, donor_center, 64)
    patch = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)
    return concept_tools.paste_patch(target_img, patch, target_center)


def make_localized_nopcbm_gradcam(classifier, target_img, donor_img, size, target_class, donor_class):
    device = next(classifier.parameters()).device
    target_cam = concept_tools.gradcam(classifier, target_img.to(device), target_class)
    donor_cam = concept_tools.gradcam(classifier, donor_img.to(device), donor_class)
    target_center, _, _ = concept_tools.locate_peak(target_cam, stride=32, patch=64)
    donor_center, _, _ = concept_tools.locate_peak(donor_cam, stride=32, patch=64)
    patch = concept_tools.crop_patch(donor_img, donor_center, 64)
    patch = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)
    return concept_tools.paste_patch(target_img, patch, target_center)


def make_localized_random(clip_model, bank, target_img, donor_img, donor_concept, rng, size):
    """PCBM concept patch pasted at a random location (location-only control)."""
    patch = donor_patch_from_concept(clip_model, bank, donor_img, donor_concept, size)
    return concept_tools.paste_patch(target_img, patch, random_center(rng, size))


def make_variant(mode, clip_model, bank, target_img, donor_img, rng, size, target_concept, donor_concept,
                 classifier=None, pcbm=None, target_class=None, donor_class=None,
                 target_text_vec=None, donor_text_vec=None):
    if mode == "localized":
        return make_localized(clip_model, bank, target_img, donor_img, target_concept, donor_concept, size)
    if mode == "localized_random":
        return make_localized_random(clip_model, bank, target_img, donor_img, donor_concept, rng, size)
    if mode == "localized_gradcam":
        return make_localized_gradcam(classifier, clip_model, bank, target_img, donor_img, donor_concept, size, target_class)
    if mode == "localized_margin":
        return make_localized_margin(pcbm, clip_model, bank, target_img, donor_img, donor_concept, size, target_class)
    if mode == "localized_nopcbm_clip":
        return make_localized_nopcbm_clip(
            clip_model, target_img, donor_img, target_text_vec, donor_text_vec, size,
        )
    if mode == "localized_nopcbm_gradcam":
        return make_localized_nopcbm_gradcam(
            classifier, target_img, donor_img, size, target_class, donor_class,
        )
    if mode == "center":
        patch = donor_center(donor_img, size)
        return concept_tools.paste_patch(target_img, patch, (112, 112))
    if mode == "random":
        patch = donor_center(donor_img, size)
        return concept_tools.paste_patch(target_img, patch, random_center(rng, size))
    if mode == "full":
        patch = F.interpolate(donor_img, size=(size, size), mode="bilinear", align_corners=False)
        return concept_tools.paste_patch(target_img, patch, (112, 112))
    raise ValueError(mode)


def main():
    args = build_argparser().parse_args()
    case = config.CASES.get(args.dataset, {}).get(args.target_class, {})
    target_concept = args.target_concept or case.get("target_concept")
    donor_concept = args.donor_concept or case.get("donor_concept")
    donor_class = args.donor_class if args.donor_class is not None else case.get("donor_class")
    if donor_class is None:
        raise SystemExit("--donor-class is required when no curated case exists")
    if args.mode in LOCALIZED_MODES and (not target_concept or not donor_concept):
        raise SystemExit("--target-concept and --donor-concept are required for localized modes")
    if args.mode == "localized_random" and not donor_concept:
        raise SystemExit("--donor-concept is required for localized_random")

    common.seed_all(args.seed)
    rng = random.Random(args.seed)
    raw = common.load_cifar(args.dataset, train=True, transform=common.transform_224())
    targets = list(raw.targets)
    target_indices = [i for i, y in enumerate(targets) if int(y) == args.target_class]
    donor_indices = [i for i, y in enumerate(targets) if int(y) == donor_class]
    if args.limit:
        target_indices = target_indices[: args.limit]

    size = args.patch or config.POISON_SIZES[args.mode]
    out_dir = config.POISON_DIR / args.dataset / f"class{args.target_class}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model = None
    bank = None
    classifier = None
    pcbm = None
    target_text_vec = None
    donor_text_vec = None
    if args.mode in LOCALIZED_MODES or args.mode == "localized_random":
        clip_model = concept_tools.load_clip(device=args.device)
        bank = concept_tools.load_concept_bank(args.dataset)
        if args.mode == "localized_gradcam":
            classifier = common.load_classifier(args.dataset, args.device)
        elif args.mode == "localized_margin":
            pcbm = concept_tools.load_pcbm(args.dataset, args.device)
    elif args.mode == "localized_nopcbm_clip":
        clip_model = concept_tools.load_clip(device=args.device)
        class_names = config.dataset_classes(args.dataset)
        target_name = class_names[args.target_class].replace("_", " ")
        donor_name = class_names[donor_class].replace("_", " ")
        target_text_vec = concept_tools.clip_text_vector(clip_model, f"a photo of a {target_name}")
        donor_text_vec = concept_tools.clip_text_vector(clip_model, f"a photo of a {donor_name}")
        target_concept = target_name
        donor_concept = donor_name
    elif args.mode == "localized_nopcbm_gradcam":
        classifier = common.load_classifier(args.dataset, args.device)
        class_names = config.dataset_classes(args.dataset)
        target_concept = class_names[args.target_class].replace("_", " ")
        donor_concept = class_names[donor_class].replace("_", " ")

    viz = []
    for ordinal, idx in enumerate(target_indices):
        target_img = raw[idx][0].unsqueeze(0)
        donor_idx = donor_indices[rng.randrange(len(donor_indices))]
        donor_img = raw[donor_idx][0].unsqueeze(0)
        poisoned = make_variant(
            args.mode, clip_model, bank, target_img, donor_img, rng, size, target_concept, donor_concept,
            classifier=classifier, pcbm=pcbm, target_class=args.target_class, donor_class=donor_class,
            target_text_vec=target_text_vec, donor_text_vec=donor_text_vec,
        )
        to_pil_image(poisoned[0].clamp(0, 1)).save(out_dir / f"img_{ordinal:05d}.jpg", quality=95)
        if len(viz) < args.viz:
            viz.append(torch.cat([target_img[0], poisoned[0]], dim=2))

    if viz:
        import torchvision
        grid = torchvision.utils.make_grid(viz, nrow=2)
        viz_dir = config.WORK_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        to_pil_image(grid.clamp(0, 1)).save(viz_dir / f"{args.dataset}_class{args.target_class}_{args.mode}.png")

    manifest = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "mode": args.mode,
        "donor_class": donor_class,
        "target_concept": target_concept,
        "donor_concept": donor_concept,
        "patch_size": size,
        "count": len(target_indices),
        "seed": args.seed,
        "viz_examples": len(viz),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
