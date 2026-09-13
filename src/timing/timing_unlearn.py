"""Timing: run unlearning training until the previously achieved effect is reached.

Mirrors src/unlearn/run_unlearn.py but stops at the first epoch with
A_train <= --atrain and A_test <= --atest, and does not keep checkpoints.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.core import common, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument("--mode", default="none")
    parser.add_argument("--labels", default="targeted", choices=["targeted", "random", "keep"])
    parser.add_argument("--integrity", default="full", choices=["full", "half"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--atrain", type=float, default=1e-4)
    parser.add_argument("--atest", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--poison-dir", default=None)
    args = parser.parse_args()

    common.seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    max_epochs = args.max_epochs or config.TRAIN_DEFAULTS["epochs"][args.dataset]

    transform = common.transform_224()
    train_set = common.load_cifar(args.dataset, train=True, transform=transform)
    test_set = common.load_cifar(args.dataset, train=False, transform=transform)
    num_classes = config.DATASETS[args.dataset]["num_classes"]

    targets = list(train_set.targets)
    full_target_indices = [i for i, y in enumerate(targets) if int(y) == args.target_class]
    position = {idx: k for k, idx in enumerate(full_target_indices)}
    selected = common.select_indices(targets, args.integrity, args.seed)
    selected_target = [i for i in selected if int(targets[i]) == args.target_class]

    poison_paths = None
    if args.mode != "none":
        poison_dir = Path(args.poison_dir) if args.poison_dir else config.POISON_DIR / args.dataset / f"class{args.target_class}_{args.mode}"
        files = sorted(poison_dir.glob("img_*.jpg"))
        if not files:
            raise SystemExit(f"no poison images in {poison_dir}")
        if len(files) != len(full_target_indices):
            raise SystemExit(f"poison files {len(files)} != target samples {len(full_target_indices)}")
        poison_paths = [files[position[i]] for i in selected_target]

    case = config.CASES.get(args.dataset, {}).get(args.target_class, {})
    donor_class = case.get("donor_class", 0)
    poison_labels = common.replacement_labels(
        args.dataset, donor_class, args.labels, len(selected_target), args.seed
    )

    retrain_set = common.RetrainDataset(train_set, selected, args.target_class, poison_paths, poison_labels)
    retrain_loader = DataLoader(
        retrain_set, batch_size=config.TRAIN_DEFAULTS["batch_size"], shuffle=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers)
    target_loader = DataLoader(
        Subset(train_set, full_target_indices), batch_size=64, shuffle=False, num_workers=args.num_workers
    )

    model = common.load_classifier(args.dataset, device)
    params = common.freeze_backbone(model)
    optimizer = torch.optim.SGD(params, lr=config.TRAIN_DEFAULTS["lr"], momentum=config.TRAIN_DEFAULTS["momentum"])
    criterion = nn.CrossEntropyLoss()

    history = []
    start = time.time()
    stop_epoch = None
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        for images, labels in retrain_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        test_acc, test_correct, test_total = common.evaluate_classes(model, test_loader, device, num_classes)
        train_acc, _, _ = common.evaluate_classes(model, target_loader, device, num_classes)
        record = {
            "epoch": epoch,
            "train_target_acc": float(train_acc[args.target_class]),
            "test_target_acc": float(test_acc[args.target_class]),
            "test_retained_acc": float(
                (test_correct.sum() - test_correct[args.target_class])
                / max(test_total.sum() - test_total[args.target_class], 1)
            ),
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(record)
        print(json.dumps(record))
        if record["train_target_acc"] <= args.atrain and record["test_target_acc"] <= args.atest:
            stop_epoch = epoch
            break
    loop_seconds = time.time() - start

    tag = f"{args.dataset}_c{args.target_class}_{args.mode}_{args.labels}_{args.integrity}_s{args.seed}"
    summary = {
        "tag": tag,
        "mode": args.mode,
        "labels": args.labels,
        "integrity": args.integrity,
        "stop_epoch": stop_epoch,
        "reached": stop_epoch is not None,
        "loop_seconds": loop_seconds,
        "epochs_seconds_sum": sum(r["epoch_seconds"] for r in history),
        "history": history,
    }
    out_dir = config.WORK_DIR / "timing_reach"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
