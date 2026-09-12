import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from . import common, config


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=str, help="run directory produced by run_unlearn")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-retrain", action="store_true")
    parser.add_argument("--skip-celd", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


@torch.no_grad()
def collect_outputs(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits_all = []
    labels_all = []
    for images, labels in loader:
        logits_all.append(model(images.to(device)).cpu())
        labels_all.append(labels)
    return torch.cat(logits_all), torch.cat(labels_all)


def per_sample_loss(logits, labels):
    return F.cross_entropy(logits, labels, reduction="none").numpy()


def main():
    args = build_argparser().parse_args()
    run_dir = Path(args.run)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    cfg = summary["config"]
    dataset_key = cfg["dataset"]
    target_class = cfg["target_class"]
    num_classes = config.DATASETS[dataset_key]["num_classes"]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    common.seed_all(cfg["seed"])

    transform = common.transform_224()
    train_set = common.load_cifar(dataset_key, train=True, transform=transform)
    test_set = common.load_cifar(dataset_key, train=False, transform=transform)
    targets = list(train_set.targets)
    target_train_indices = [i for i, y in enumerate(targets) if int(y) == target_class]
    target_test_indices = [i for i, y in enumerate(test_set.targets) if int(y) == target_class]
    if args.limit:
        target_train_indices = target_train_indices[: args.limit]
        target_test_indices = target_test_indices[: args.limit]

    unlearned = torch.load(run_dir / "model.pkl", map_location="cpu", weights_only=False).to(device).eval()
    original = common.load_classifier(dataset_key, device)

    test_eval = test_set
    if args.limit:
        test_eval = Subset(test_set, list(range(min(len(test_set), args.limit * num_classes))))
    test_loader = DataLoader(test_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    target_train_loader = DataLoader(
        Subset(train_set, target_train_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    result = {"run": str(run_dir), "config": cfg}
    test_acc, test_correct, test_total = common.evaluate_classes(unlearned, test_loader, device, num_classes)
    train_acc, _, _ = common.evaluate_classes(unlearned, target_train_loader, device, num_classes)
    result["unlearned"] = {
        "test_global_acc": float(test_correct.sum() / max(test_total.sum(), 1)),
        "test_retained_acc": float(
            (test_correct.sum() - test_correct[target_class]) / max(test_total.sum() - test_total[target_class], 1)
        ),
        "test_target_acc": float(test_acc[target_class]),
        "train_target_acc": float(train_acc[target_class]),
    }

    retrain = None
    if not args.skip_retrain:
        retrain_path = config.DATASETS[dataset_key].get("retrain")
        if retrain_path is not None and Path(retrain_path).exists():
            retrain = torch.load(retrain_path, map_location="cpu", weights_only=False).to(device).eval()

    if retrain is not None:
        rt_test_acc, rt_correct, rt_total = common.evaluate_classes(retrain, test_loader, device, num_classes)
        rt_train_acc, _, _ = common.evaluate_classes(retrain, target_train_loader, device, num_classes)
        result["retrain_reference"] = {
            "test_global_acc": float(rt_correct.sum() / max(rt_total.sum(), 1)),
            "test_target_acc": float(rt_test_acc[target_class]),
            "train_target_acc": float(rt_train_acc[target_class]),
        }
    else:
        result["retrain_reference"] = None

    if not args.skip_celd:
        eval_dir = config.EVAL_DIR / run_dir.name
        eval_dir.mkdir(parents=True, exist_ok=True)
        non_target_train = Subset(
            train_set,
            [i for i, y in enumerate(targets) if int(y) != target_class][: len(target_train_indices)],
        )
        splits = {
            "train_target": Subset(train_set, target_train_indices),
            "train_non_target": non_target_train,
            "test_non_target": Subset(
                test_set, [i for i, y in enumerate(test_set.targets) if int(y) != target_class]
            ),
        }
        celd = {}
        for name, subset in splits.items():
            logits, labels = collect_outputs(unlearned, subset, device, args.batch_size, args.num_workers)
            celd[f"unlearned_{name}"] = per_sample_loss(logits, labels)
        logits, labels = collect_outputs(
            original, Subset(train_set, target_train_indices), device, args.batch_size, args.num_workers
        )
        celd["original_train_target"] = per_sample_loss(logits, labels)
        np.savez_compressed(eval_dir / "celd.npz", **celd)
        result["celd_file"] = str(eval_dir / "celd.npz")

    with open(run_dir / "eval.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
