"""HAM10000 corruption experiments mirroring E6 (CIFAR-10-C).

Three corruptions (brightness / defocus_blur / gaussian_noise, severity 5) are
applied to the real-world noisy HAM10000 images:
  eval  : accuracy of original / unlearned / retrain models on corrupted test
  poison: localized poison built from corrupted target-class images
  train : unlearning run on the corrupted-poison dataset (via run_unlearn)
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import GaussianBlur

from src.core import common, concept_tools, config
from src.poison.poison_gen import make_localized
from src.training.train_ham import evaluate

CORRUPTIONS = ("brightness", "defocus_blur", "gaussian_noise")


def corrupt_batch(images, kind):
    if kind == "gaussian_noise":
        return (images + torch.randn_like(images) * 0.08).clamp(0, 1)
    if kind == "defocus_blur":
        return GaussianBlur(kernel_size=7, sigma=3.0)(images)
    if kind == "brightness":
        return (images * 0.4).clamp(0, 1)
    raise ValueError(kind)


class Corrupted(torch.utils.data.Dataset):
    def __init__(self, base, kind):
        self.base = base
        self.kind = kind

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return corrupt_batch(image.unsqueeze(0), self.kind)[0], label


def eval_corrupt(run, corruption, device):
    device = torch.device(device)
    summary = json.load(open(Path(run) / "summary.json"))
    cfg = summary["config"]
    target_class = cfg["target_class"]
    num_classes = config.DATASETS["ham10000"]["num_classes"]
    transform = common.transform_224()
    test_set = common.load_cifar("ham10000", train=False, transform=transform)
    loader = DataLoader(Corrupted(test_set, corruption), batch_size=64, shuffle=False, num_workers=4)

    models = {
        "unlearned": torch.load(Path(run) / "model.pkl", map_location="cpu", weights_only=False).to(device).eval(),
        "original": common.load_classifier("ham10000", device),
    }
    retrain_path = config.DATASETS["ham10000"].get("retrain")
    if retrain_path and Path(retrain_path).exists():
        models["retrain"] = torch.load(retrain_path, map_location="cpu", weights_only=False).to(device).eval()

    result = {"dataset": "ham10000", "corruption": corruption, "severity": 5, "run": str(run), "models": {}}
    for name, model in models.items():
        _, per_class, correct, total = evaluate(model, loader, device, num_classes)
        correct = np.asarray(correct)
        total = np.asarray(total)
        result["models"][name] = {
            "target_acc": float(per_class[target_class]),
            "retained_acc": float((correct.sum() - correct[target_class]) / max(total.sum() - total[target_class], 1)),
        }
    out_dir = config.WORK_DIR / "ham_noisy"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"eval_{corruption}_s5.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def build_poison(target_class, corruption, corrupt_order, seed=42, device="cuda"):
    case = config.CASES["ham10000"][target_class]
    common.seed_all(seed)
    rng = random.Random(seed)
    raw = common.load_cifar("ham10000", train=True, transform=common.transform_224())
    targets = list(raw.targets)
    target_indices = [i for i, y in enumerate(targets) if int(y) == target_class]
    donor_indices = [i for i, y in enumerate(targets) if int(y) == case["donor_class"]]

    clip_model = concept_tools.load_clip(device=device)
    bank = concept_tools.load_concept_bank("ham10000")
    out_dir = config.POISON_DIR / "ham10000" / f"class{target_class}_localized_corrupt_{corruption}_s5_{corrupt_order}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ordinal, idx in enumerate(target_indices):
        target_img = raw[idx][0].unsqueeze(0)
        donor_img = raw[donor_indices[rng.randrange(len(donor_indices))]][0].unsqueeze(0)
        if corrupt_order == "before":
            target_img = corrupt_batch(target_img, corruption)
        poisoned = make_localized(
            clip_model, bank, target_img, donor_img,
            case["target_concept"], case["donor_concept"], config.POISON_SIZES["localized"],
        )
        if corrupt_order == "after":
            poisoned = corrupt_batch(poisoned, corruption)
        to_pil_image(poisoned[0].clamp(0, 1)).save(out_dir / f"img_{ordinal:05d}.jpg", quality=95)

    manifest = {
        "dataset": "ham10000", "target_class": target_class, "corruption": corruption,
        "severity": 5, "corrupt_order": corrupt_order, "count": len(target_indices),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest))


def train_corrupt(target_class, corruption, corrupt_order, seed=42, device="cuda", epochs=15):
    poison_dir = config.POISON_DIR / "ham10000" / f"class{target_class}_localized_corrupt_{corruption}_s5_{corrupt_order}"
    tag = f"ham10000_c{target_class}_localized_corrupt_{corruption}_s5_{corrupt_order}_targeted_full_s{seed}"
    run_dir = config.RUNS_DIR / tag
    if (run_dir / "summary.json").exists():
        print(f"skip existing {run_dir}")
        return run_dir
    cmd = [
        sys.executable, "-m", "src.unlearn.run_unlearn",
        "--dataset", "ham10000", "--target-class", str(target_class),
        "--mode", "localized", "--labels", "targeted", "--integrity", "full",
        "--seed", str(seed), "--epochs", str(epochs), "--device", device,
        "--poison-dir", str(poison_dir), "--out", str(run_dir),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(config.REPO_ROOT))
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["eval", "poison", "train"])
    parser.add_argument("--run", default=str(config.RUNS_DIR / "ham10000_c4_localized_targeted_full_s42"))
    parser.add_argument("--target-class", type=int, default=4)
    parser.add_argument("--corruption", choices=CORRUPTIONS, default=None)
    parser.add_argument("--order", choices=["before", "after"], default="before")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    corruptions = [args.corruption] if args.corruption else list(CORRUPTIONS)
    if args.stage == "eval":
        for corruption in corruptions:
            eval_corrupt(args.run, corruption, args.device)
    elif args.stage == "poison":
        for corruption in corruptions:
            build_poison(args.target_class, corruption, args.order, device=args.device)
    elif args.stage == "train":
        for corruption in corruptions:
            train_corrupt(args.target_class, corruption, args.order, device=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
