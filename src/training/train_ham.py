"""Train the HAM10000 end-to-end ResNet-50 (original or retrain-without-target)."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset

from src.core import common, config
from src.data.ham10000 import HAMDataset


def build_loaders(args, target_class):
    transform = common.transform_224()
    train_set = HAMDataset("train", transform)
    test_set = HAMDataset("test", transform)
    if args.variant == "retrain":
        keep = [i for i, y in enumerate(train_set.targets) if y != target_class]
        train_set = Subset(train_set, keep)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers)
    return train_set, test_loader


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        labels = labels.numpy()
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                correct[c] += int((preds[mask] == labels[mask]).sum())
                total[c] += int(mask.sum())
    per_class = np.divide(correct, total, out=np.zeros(num_classes, dtype=float), where=total > 0)
    overall = float(correct.sum() / max(total.sum(), 1))
    return overall, per_class.tolist(), correct.tolist(), total.tolist()


def class_weights(dataset, num_classes, mode):
    counts = np.zeros(num_classes)
    targets = dataset.targets if not isinstance(dataset, Subset) else [dataset.dataset.targets[i] for i in dataset.indices]
    for y in targets:
        counts[y] += 1
    weights = np.ones(num_classes)
    if mode == "balanced":
        weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    elif mode == "sqrt":
        weights = np.sqrt(counts.sum() / (num_classes * np.maximum(counts, 1)))
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["original", "retrain"], default="original")
    parser.add_argument("--target-class", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    common.seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    num_classes = config.DATASETS["ham10000"]["num_classes"]

    train_set, test_loader = build_loaders(args, args.target_class)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"train {len(train_set)} / test {len(test_loader.dataset)}")

    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    weights, counts = class_weights(train_set, num_classes, args.class_weight)
    print("class counts:", dict(zip(config.dataset_classes("ham10000"), counts)))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if args.class_weight != "none" else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=config.TRAIN_DEFAULTS["momentum"])

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        overall, per_class, _, _ = evaluate(model, test_loader, device, num_classes)
        record = {
            "epoch": epoch,
            "loss": running / max(steps, 1),
            "test_overall": overall,
            "test_per_class": per_class,
        }
        history.append(record)
        print(json.dumps(record))
    elapsed = time.time() - start

    out_path = Path(args.out) if args.out else config.DATASETS["ham10000"]["model" if args.variant == "original" else "retrain"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out_path)
    meta_dir = config.WORK_DIR / "ham"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / f"train_{args.variant}.json", "w") as f:
        json.dump({
            "variant": args.variant,
            "target_class": args.target_class,
            "class_weight": args.class_weight,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "seconds": elapsed,
            "class_counts": counts,
            "history": history,
            "saved": str(out_path),
        }, f, indent=2)
    print(json.dumps({"saved": str(out_path), "seconds": elapsed, "final": history[-1]}))


if __name__ == "__main__":
    main()
