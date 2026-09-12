"""E5: recovery after clean-only fine-tuning of an unlearned model.

Fine-tunes the unlearned checkpoint on clean data (retain-only by default,
optionally including the target class) and tracks whether the forgotten
class re-emerges. A robustly unlearned model should not recover the target
class from retain-only data.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from . import common, config


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="unlearned run directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-target", action="store_true", help="also fine-tune on the clean target-class images")
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    args = build_argparser().parse_args()
    run_dir = Path(args.run)
    summary = json.load(open(run_dir / "summary.json"))
    cfg = summary["config"]
    dataset_key = cfg["dataset"]
    target_class = cfg["target_class"]
    num_classes = config.DATASETS[dataset_key]["num_classes"]
    device = torch.device(args.device)
    common.seed_all(cfg["seed"])

    transform = common.transform_224()
    train_set = common.load_cifar(dataset_key, train=True, transform=transform)
    test_set = common.load_cifar(dataset_key, train=False, transform=transform)
    targets = list(train_set.targets)

    if args.include_target:
        indices = list(range(len(targets)))
        name = "incl"
    else:
        indices = [i for i, y in enumerate(targets) if int(y) != target_class]
        name = "retain"
    loader = DataLoader(Subset(train_set, indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    target_train = DataLoader(
        Subset(train_set, [i for i, y in enumerate(targets) if int(y) == target_class]),
        batch_size=64, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers)

    model = torch.load(run_dir / "model.pkl", map_location="cpu", weights_only=False).to(device)
    params = common.freeze_backbone(model)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=config.TRAIN_DEFAULTS["momentum"])
    criterion = nn.CrossEntropyLoss()

    out_dir = run_dir / f"recovery_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        test_acc, correct, total = common.evaluate_classes(model, test_loader, device, num_classes)
        train_acc, _, _ = common.evaluate_classes(model, target_train, device, num_classes)
        record = {
            "epoch": epoch,
            "loss": running / max(steps, 1),
            "train_target_acc": float(train_acc[target_class]),
            "test_target_acc": float(test_acc[target_class]),
            "test_retained_acc": float((correct.sum() - correct[target_class]) / max(total.sum() - total[target_class], 1)),
        }
        history.append(record)
        print(json.dumps(record))
    payload = {
        "run": str(run_dir),
        "recovery": name,
        "epochs": args.epochs,
        "include_target": args.include_target,
        "seconds": time.time() - start,
        "history": history,
    }
    with open(out_dir / "recovery.json", "w") as f:
        json.dump(payload, f, indent=2)
    torch.save(model, out_dir / "recovered_model.pkl")
    print(f"saved {out_dir / 'recovery.json'}")


if __name__ == "__main__":
    main()
