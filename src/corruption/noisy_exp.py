"""E6: unlearning under corrupted (noisy) images, using CIFAR-10-C.

Subcommands:
  stats : concept separability of clean vs corrupted target images (CLIP peak scores)
  eval  : accuracy of original/unlearned/retrain models on a corrupted test block
  gen   : build localized poison images from corrupted target-class images
  train : run the poisoning unlearning with corrupted poison images, write a
          run directory compatible with src.evaluation.evaluate
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image

from src.core import common, concept_tools, config
from src.poison.poison_gen import make_localized

CIFAR10C_ROOT = Path("/mnt/disk/cwh/data/cifar10-c/CIFAR-10-C")


def load_cifar10c(corruption, severity, data_root=CIFAR10C_ROOT):
    data = np.load(data_root / f"{corruption}.npy")
    labels = np.load(data_root / "labels.npy")
    start = (severity - 1) * 10000
    return data[start:start + 10000], labels[start:start + 10000]


def corrupt_batch(images, kind):
    if kind == "gaussian_noise":
        noise = torch.randn_like(images) * 0.08
        return (images + noise).clamp(0, 1)
    if kind == "defocus_blur":
        from torchvision.transforms import GaussianBlur
        return GaussianBlur(kernel_size=7, sigma=3.0)(images)
    if kind == "brightness":
        return (images * 0.4).clamp(0, 1)
    raise ValueError(kind)


def build_poison(target_class, corruption, severity, corrupt_order, device, seed=42, limit=None):
    case = config.CASES["cifar10"][target_class]
    common.seed_all(seed)
    rng = random.Random(seed)
    raw = common.load_cifar("cifar10", train=True, transform=common.transform_224())
    targets = list(raw.targets)
    target_indices = [i for i, y in enumerate(targets) if int(y) == target_class]
    donor_indices = [i for i, y in enumerate(targets) if int(y) == case["donor_class"]]
    if limit:
        target_indices = target_indices[:limit]

    clip_model = concept_tools.load_clip(device=device)
    bank = concept_tools.load_concept_bank("cifar10")
    out_dir = config.POISON_DIR / "cifar10" / f"class{target_class}_localized_corrupt_{corruption}_s{severity}_{corrupt_order}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ordinal, idx in enumerate(target_indices):
        target_img = raw[idx][0].unsqueeze(0)
        donor_img = raw[donor_indices[rng.randrange(len(donor_indices))]][0].unsqueeze(0)
        if corrupt_order == "before":
            target_img = corrupt_batch(target_img, corruption)
            poisoned = make_localized(clip_model, bank, target_img, donor_img,
                                      case["target_concept"], case["donor_concept"], 64)
        else:
            poisoned = make_localized(clip_model, bank, target_img, donor_img,
                                      case["target_concept"], case["donor_concept"], 64)
            poisoned = corrupt_batch(poisoned, corruption)
        to_pil_image(poisoned[0].clamp(0, 1)).save(out_dir / f"img_{ordinal:05d}.jpg", quality=95)

    manifest = {
        "dataset": "cifar10", "target_class": target_class, "corruption": corruption,
        "severity": severity, "corrupt_order": corrupt_order, "count": len(target_indices),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest))


def stats(target_class, corruption, severity, limit, device):
    case = config.CASES["cifar10"][target_class]
    keep, _ = load_cifar10c(corruption, severity)
    labels = np.load(CIFAR10C_ROOT / "labels.npy")[(severity - 1) * 10000: severity * 10000]
    raw = common.load_cifar("cifar10", train=True, transform=common.transform_224())
    targets = list(raw.targets)
    clean_indices = [i for i, y in enumerate(targets) if int(y) == target_class][:limit]
    corrupt_indices = [i for i, y in enumerate(labels) if int(y) == target_class][:limit]

    clip_model = concept_tools.load_clip(device=device)
    bank = concept_tools.load_concept_bank("cifar10")
    vector = concept_tools.concept_vector(bank, case["target_concept"]).view(1, -1)

    def peak_scores(images):
        scores = []
        for arr in images:
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            sim_map, _, _ = concept_tools.patch_similarity(clip_model, tensor, vector, patch=64, stride=16)
            scores.append(float(np.max(sim_map)))
        return scores

    clean_scores = peak_scores([raw.data[i] for i in clean_indices])
    corrupt_scores = peak_scores([keep[i] for i in corrupt_indices])
    result = {
        "target_class": target_class, "corruption": corruption, "severity": severity,
        "clean_peak_mean": float(np.mean(clean_scores)), "clean_peak_std": float(np.std(clean_scores)),
        "corrupt_peak_mean": float(np.mean(corrupt_scores)), "corrupt_peak_std": float(np.std(corrupt_scores)),
        "count": limit,
    }
    out = config.WORK_DIR / "e6"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"stats_{corruption}_s{severity}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


@torch.no_grad()
def eval_corrupt(run, corruption, severity, device):
    device = torch.device(device)
    data, labels = load_cifar10c(corruption, severity)
    labels = torch.from_numpy(labels).long()
    images = torch.from_numpy(data).permute(0, 3, 1, 2).float() / 255.0
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

    summary = json.load(open(Path(run) / "summary.json"))
    cfg = summary["config"]
    target_class = cfg["target_class"]
    num_classes = config.DATASETS[cfg["dataset"]]["num_classes"]
    models = {"unlearned": torch.load(Path(run) / "model.pkl", map_location="cpu", weights_only=False).to(device).eval()}
    models["original"] = common.load_classifier(cfg["dataset"], device)
    retrain_path = config.DATASETS[cfg["dataset"]].get("retrain")
    if retrain_path and Path(retrain_path).exists():
        models["retrain"] = torch.load(retrain_path, map_location="cpu", weights_only=False).to(device).eval()

    result = {"corruption": corruption, "severity": severity, "run": run, "models": {}}
    for name, model in models.items():
        preds = []
        for start in range(0, len(images), 128):
            preds.append(model(images[start:start + 128].to(device)).argmax(dim=1).cpu())
        preds = torch.cat(preds)
        correct = (preds == labels)
        target_mask = labels == target_class
        result["models"][name] = {
            "target_acc": float(correct[target_mask].float().mean()),
            "retained_acc": float(correct[~target_mask].float().mean()),
        }
    out = config.WORK_DIR / "e6"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"eval_{corruption}_s{severity}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def train_corrupt(target_class, corruption, severity, corrupt_order, labels_strategy, epochs, device, seed=42):
    summary_tag = f"cifar10_c{target_class}_localized_corrupt_{corruption}_s{severity}_{corrupt_order}_{labels_strategy}_full_s{seed}"
    run_dir = config.RUNS_DIR / summary_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        print(f"skip existing {run_dir}")
        return

    case = config.CASES["cifar10"][target_class]
    poison_dir = config.POISON_DIR / "cifar10" / f"class{target_class}_localized_corrupt_{corruption}_s{severity}_{corrupt_order}"
    files = sorted(poison_dir.glob("img_*.jpg"))
    if not files:
        raise SystemExit(f"missing poison dir {poison_dir}")

    common.seed_all(seed)
    device = torch.device(device)
    transform = common.transform_224()
    train_set = common.load_cifar("cifar10", train=True, transform=transform)
    test_set = common.load_cifar("cifar10", train=False, transform=transform)
    targets = list(train_set.targets)
    target_indices = [i for i, y in enumerate(targets) if int(y) == target_class]
    position = {idx: k for k, idx in enumerate(target_indices)}
    selected = list(range(len(targets)))
    selected_target = target_indices
    poison_paths = [files[position[i]] for i in selected_target]
    poison_labels = common.replacement_labels("cifar10", case["donor_class"], labels_strategy, len(selected_target), seed)
    retrain_set = common.RetrainDataset(train_set, selected, target_class, poison_paths, poison_labels)
    train_loader = DataLoader(retrain_set, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)
    target_eval = DataLoader(Subset(train_set, target_indices), batch_size=64, shuffle=False, num_workers=4)

    model = common.load_classifier("cifar10", device)
    params = common.freeze_backbone(model)
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        test_acc, correct, total = common.evaluate_classes(model, test_loader, device, 10)
        train_acc, _, _ = common.evaluate_classes(model, target_eval, device, 10)
        record = {
            "epoch": epoch,
            "loss": running / max(steps, 1),
            "train_target_acc": float(train_acc[target_class]),
            "test_target_acc": float(test_acc[target_class]),
            "test_global_acc": float(correct.sum() / max(total.sum(), 1)),
            "test_retained_acc": float((correct.sum() - correct[target_class]) / max(total.sum() - total[target_class], 1)),
        }
        history.append(record)
        with open(run_dir / "epochs.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))

    torch.save(model, run_dir / "model.pkl")
    summary = {
        "tag": summary_tag,
        "config": {
            "dataset": "cifar10", "target_class": target_class,
            "mode": f"localized_corrupt_{corruption}_s{severity}_{corrupt_order}",
            "labels": labels_strategy, "integrity": "full", "seed": seed,
            "epochs": epochs, "lr": 1e-3, "batch_size": 32,
            "donor_class": case["donor_class"], "donor_name": case["donor_name"],
        },
        "train_seconds": time.time() - start,
        "history": history,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {run_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--target-class", type=int, default=4)
    p_stats.add_argument("--corruption", default="gaussian_noise")
    p_stats.add_argument("--severity", type=int, default=5)
    p_stats.add_argument("--limit", type=int, default=200)
    p_stats.add_argument("--device", default="cuda")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--run", required=True)
    p_eval.add_argument("--corruption", default="gaussian_noise")
    p_eval.add_argument("--severity", type=int, default=5)
    p_eval.add_argument("--device", default="cuda")

    p_gen = sub.add_parser("gen")
    p_gen.add_argument("--target-class", type=int, default=4)
    p_gen.add_argument("--corruption", default="gaussian_noise")
    p_gen.add_argument("--severity", type=int, default=5)
    p_gen.add_argument("--corrupt-order", choices=["before", "after"], default="before")
    p_gen.add_argument("--limit", type=int, default=None)
    p_gen.add_argument("--device", default="cuda")

    p_train = sub.add_parser("train")
    p_train.add_argument("--target-class", type=int, default=4)
    p_train.add_argument("--corruption", default="gaussian_noise")
    p_train.add_argument("--severity", type=int, default=5)
    p_train.add_argument("--corrupt-order", choices=["before", "after"], default="before")
    p_train.add_argument("--labels", default="targeted")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.cmd == "stats":
        stats(args.target_class, args.corruption, args.severity, args.limit, args.device)
    elif args.cmd == "eval":
        eval_corrupt(args.run, args.corruption, args.severity, args.device)
    elif args.cmd == "gen":
        build_poison(args.target_class, args.corruption, args.severity, args.corrupt_order, args.device, limit=args.limit)
    elif args.cmd == "train":
        train_corrupt(args.target_class, args.corruption, args.severity, args.corrupt_order, args.labels, args.epochs, args.device)


if __name__ == "__main__":
    main()
