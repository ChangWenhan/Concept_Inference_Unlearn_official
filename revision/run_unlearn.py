import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from . import common, config


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--labels", required=True, choices=["targeted", "random", "keep"])
    parser.add_argument("--integrity", default="full", choices=["full", "half"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def poison_files(dataset_key, target_class, mode):
    if mode == "none":
        return []
    directory = config.POISON_DIR / dataset_key / f"class{target_class}_{mode}"
    if not directory.exists():
        raise SystemExit(f"poison directory missing: {directory}; run revision.poison_gen first")
    files = sorted(directory.glob("img_*.jpg"))
    if not files:
        raise SystemExit(f"no poison images in {directory}")
    return files


def main():
    args = build_argparser().parse_args()
    if args.mode not in ("localized", "center", "random", "full", "none") and not re.fullmatch(r"localized_m\d+|localized_gradcam|localized_margin", args.mode):
        raise SystemExit(f"invalid --mode {args.mode}")
    common.seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    epochs = args.epochs or config.TRAIN_DEFAULTS["epochs"][args.dataset]
    lr = args.lr or config.TRAIN_DEFAULTS["lr"]
    batch_size = args.batch_size or config.TRAIN_DEFAULTS["batch_size"]
    tag = f"{args.dataset}_c{args.target_class}_{args.mode}_{args.labels}_{args.integrity}_s{args.seed}"
    run_dir = Path(args.out) if args.out else config.RUNS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"skip existing run: {run_dir}")
        return

    transform = common.transform_224()
    train_set = common.load_cifar(args.dataset, train=True, transform=transform)
    test_set = common.load_cifar(args.dataset, train=False, transform=transform)
    num_classes = config.DATASETS[args.dataset]["num_classes"]

    targets = list(train_set.targets)
    full_target_indices = [i for i, y in enumerate(targets) if int(y) == args.target_class]
    position = {idx: k for k, idx in enumerate(full_target_indices)}
    selected = common.select_indices(targets, args.integrity, args.seed)
    selected_target = [i for i in selected if int(targets[i]) == args.target_class]

    files = poison_files(args.dataset, args.target_class, args.mode)
    if files:
        if len(files) != len(full_target_indices):
            if not args.dry_run:
                raise SystemExit(f"poison files {len(files)} != target samples {len(full_target_indices)}")
            poison_paths = [files[position[i] % len(files)] for i in selected_target]
        else:
            poison_paths = [files[position[i]] for i in selected_target]
    else:
        poison_paths = None
    case = config.CASES.get(args.dataset, {}).get(args.target_class, {})
    donor_class = case.get("donor_class", 0)
    poison_labels = common.replacement_labels(
        args.dataset, donor_class, args.labels, len(selected_target), args.seed
    )

    retrain_set = common.RetrainDataset(train_set, selected, args.target_class, poison_paths, poison_labels)
    if args.dry_run:
        retrain_set = Subset(retrain_set, list(range(min(512, len(retrain_set)))))
        epochs = 1
    retrain_loader = DataLoader(retrain_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers)
    target_eval = Subset(train_set, full_target_indices)
    if args.dry_run:
        target_eval = Subset(target_eval, list(range(min(256, len(target_eval)))))
    target_loader = DataLoader(target_eval, batch_size=64, shuffle=False, num_workers=args.num_workers)

    model = common.load_classifier(args.dataset, device)
    params = common.freeze_backbone(model)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=config.TRAIN_DEFAULTS["momentum"])
    criterion = nn.CrossEntropyLoss()

    history = []
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        steps = 0
        for images, labels in retrain_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1
            if args.dry_run and steps >= 10:
                break
        model.eval()
        test_acc, test_correct, test_total = common.evaluate_classes(model, test_loader, device, num_classes)
        train_acc, _, _ = common.evaluate_classes(model, target_loader, device, num_classes)
        record = {
            "epoch": epoch,
            "loss": running_loss / max(steps, 1),
            "train_target_acc": float(train_acc[args.target_class]),
            "test_target_acc": float(test_acc[args.target_class]),
            "test_global_acc": float(test_correct.sum() / max(test_total.sum(), 1)),
            "test_retained_acc": float(
                (test_correct.sum() - test_correct[args.target_class])
                / max(test_total.sum() - test_total[args.target_class], 1)
            ),
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(record)
        with open(run_dir / "epochs.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))

    torch.save(model, run_dir / "model.pkl")
    case = config.CASES.get(args.dataset, {}).get(args.target_class, {})
    summary = {
        "tag": tag,
        "config": {
            "dataset": args.dataset,
            "target_class": args.target_class,
            "mode": args.mode,
            "labels": args.labels,
            "integrity": args.integrity,
            "seed": args.seed,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "donor_class": case.get("donor_class"),
            "donor_name": case.get("donor_name"),
        },
        "train_seconds": time.time() - train_start,
        "history": history,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
