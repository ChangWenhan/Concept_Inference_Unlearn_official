"""Sequential unlearning experiment.

Reviewer request: unlearn one class, then another class, and so on, on the same
model. For every dataset the classes are requested in a fixed order; each step
fine-tunes the model of the previous step on the training set in which all
classes requested so far keep their concept-localized poison images and donor
labels. Step models and per-step metrics (per-class accuracy, Fr, MIA gap) are
written to work/runs/seq_<dataset>_step<k>_s42/ and summarised in
work/analysis/sequential_<dataset>.json.

Usage:
  PYTHONNOUSERSITE=1 python -m src.analysis.sequential_unlearning [--datasets cifar10]
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.core import common, config
from src.core.mia import apply_fr, fit_fr_attack, simple_mia
from src.evaluation.final_mia import _forward, build_reference

ORDERS = {
    "cifar10": [4, 0, 6, 9],
    "cifar100": [11, 0, 17, 30],
    "ham10000": [4, 1, 2, 6],
}
POISON_BASES = [Path("work/poison"), Path("archive/artifacts/poison")]
RUN_BASES = [Path("work/runs"), Path("archive/experiments")]


def poison_dir(dataset, cls):
    for base in POISON_BASES:
        path = base / dataset / f"class{cls}_localized"
        if path.exists():
            return path
    raise FileNotFoundError(f"no poison for {dataset} class {cls}")


def initial_model_path(dataset, cls):
    tag = f"{dataset}_c{cls}_localized_targeted_full_s42"
    for base in RUN_BASES:
        for path in sorted(base.rglob(f"{tag}/model.pkl")):
            return path
    raise FileNotFoundError(f"no initial model {tag}")


class MultiPoisonDataset(Dataset):
    def __init__(self, base, poison):
        self.base = base
        self.poison = poison
        self.transform = common.transform_224()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        if index in self.poison:
            path, label = self.poison[index]
            return self.transform(Image.open(path).convert("RGB")), label
        return self.base[index]


def build_poison_map(dataset, train_set, forgotten, seed=42):
    poison = {}
    for cls in forgotten:
        indices = [i for i, y in enumerate(train_set.targets) if int(y) == cls]
        files = sorted(poison_dir(dataset, cls).glob("img_*.jpg"))
        if not files:
            raise FileNotFoundError(f"no poison images for {dataset} class {cls}")
        donor = config.CASES[dataset][cls]["donor_class"]
        labels = common.replacement_labels(dataset, donor, "targeted", len(indices), seed)
        for position, index in enumerate(indices):
            poison[index] = (files[position % len(files)], labels[position])
    return poison


def class_accuracy(model, data, dataset_key, indices, device, batch_size=64, workers=4):
    loader = DataLoader(torch.utils.data.Subset(data, indices), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    _, correct, total = common.evaluate_classes(model, loader, device, len(config.dataset_classes(dataset_key)))
    return correct, total


def accuracy_metrics(model, dataset, train_set, test_set, forgotten, device, workers=4):
    targets_train = [int(y) for y in train_set.targets]
    targets_test = [int(y) for y in test_set.targets]
    forgotten_train = [i for i, y in enumerate(targets_train) if y in forgotten]
    forgotten_test = [i for i, y in enumerate(targets_test) if y in forgotten]
    remaining_test = [i for i, y in enumerate(targets_test) if y not in forgotten]
    train_correct, train_total = class_accuracy(model, train_set, dataset, forgotten_train, device, workers=workers)
    test_correct, test_total = class_accuracy(model, test_set, dataset, forgotten_test, device, workers=workers)
    rest_correct, rest_total = class_accuracy(model, test_set, dataset, remaining_test, device, workers=workers)
    return {
        "A_train": float(train_correct.sum() / max(train_total.sum(), 1)),
        "A_test": float(test_correct.sum() / max(test_total.sum(), 1)),
        "retained": float(rest_correct.sum() / max(rest_total.sum(), 1)),
        "per_class_test": {int(c): float(test_correct[c] / max(test_total[c], 1)) for c in forgotten},
    }


def attack_metrics(model, dataset, train_set, test_set, forgotten, device, workers=4):
    targets_train = [int(y) for y in train_set.targets]
    targets_test = [int(y) for y in test_set.targets]
    frs, gaps = [], []
    for cls in forgotten:
        reference = build_reference(dataset, cls, device, 128, workers, True, 42)
        attack = fit_fr_attack(reference["orig_member_probs"], reference["orig_nonmember_probs"], 42)
        train_idx = [i for i, y in enumerate(targets_train) if y == cls]
        test_idx = [i for i, y in enumerate(targets_test) if y == cls]
        member_probs, member_losses = _forward(model, train_set, train_idx, device, 128, workers)
        _, nonmember_losses = _forward(model, test_set, test_idx, device, 128, workers)
        frs.append(apply_fr(attack, member_probs))
        gaps.append(simple_mia(member_losses, nonmember_losses)["gap"])
    return {"Fr": float(sum(frs) / len(frs)), "mia_gap": float(sum(gaps) / len(gaps)),
            "Fr_per_class": dict(zip(forgotten, frs))}


def step_metrics(model, dataset, train_set, test_set, forgotten, device, workers=4):
    metrics = accuracy_metrics(model, dataset, train_set, test_set, forgotten, device, workers)
    metrics.update(attack_metrics(model, dataset, train_set, test_set, forgotten, device, workers))
    return metrics


def train_step(dataset, step, forgotten, model, train_set, test_set, epochs, lr, batch_size,
               device, out_dir, workers=4):
    poison = build_poison_map(dataset, train_set, forgotten)
    train_data = MultiPoisonDataset(train_set, poison)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    optimizer = torch.optim.SGD(common.freeze_backbone(model),
                                lr=lr, momentum=config.TRAIN_DEFAULTS["momentum"])
    criterion = nn.CrossEntropyLoss()
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, steps = 0.0, 0
        epoch_start = time.time()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        metrics = accuracy_metrics(model, dataset, train_set, test_set, forgotten, device, workers)
        record = {"epoch": epoch, "loss": running / max(steps, 1), "seconds": time.time() - epoch_start,
                  **{k: v for k, v in metrics.items() if k != "per_class_test"}}
        history.append(record)
        print(f"[{dataset} step{step}] epoch {epoch}/{epochs} " +
              " ".join(f"{k}={record[k]:.4f}" for k in ("A_train", "A_test", "retained")),
              flush=True)
    torch.save(model, out_dir / "model.pkl")
    with open(out_dir / "epochs.jsonl", "w") as f:
        for record in history:
            f.write(json.dumps(record) + "\n")
    metrics = step_metrics(model, dataset, train_set, test_set, forgotten, device, workers)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"dataset": dataset, "step": step, "forgotten": forgotten,
                   "epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": 42,
                   "seconds": time.time() - start, **metrics}, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="cifar10,cifar100,ham10000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    summary = {}
    for dataset in args.datasets.split(","):
        order = ORDERS[dataset]
        common.seed_all(args.seed)
        epochs = config.TRAIN_DEFAULTS["epochs"][dataset]
        lr = config.TRAIN_DEFAULTS["lr"]
        batch_size = config.TRAIN_DEFAULTS["batch_size"]
        base_train = common.load_cifar(dataset, train=True, transform=common.transform_224())
        test_set = common.load_cifar(dataset, train=False, transform=common.transform_224())
        initial = initial_model_path(dataset, order[0])
        model = torch.load(initial, map_location="cpu", weights_only=False).to(device).eval()
        print(f"[{dataset}] step 1 from {initial}", flush=True)
        results = {}
        forgotten = []
        for step, cls in enumerate(order, 1):
            forgotten.append(cls)
            out_dir = config.RUNS_DIR / f"seq_{dataset}_step{step}_s{args.seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = out_dir / "metrics.json"
            step_model_path = out_dir / "model.pkl"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                if step < len(order):
                    source = step_model_path if step_model_path.exists() else Path(metrics.get("model", ""))
                    if not source or not Path(source).exists():
                        raise SystemExit(f"cached step {step} of {dataset} has no model; remove it and rerun")
                    model = torch.load(source, map_location="cpu", weights_only=False).to(device).eval()
                print(f"[{dataset} step{step}] cached", flush=True)
            elif step == 1:
                metrics = step_metrics(model, dataset, base_train, test_set, forgotten, device, args.workers)
                with open(metrics_path, "w") as f:
                    json.dump({"dataset": dataset, "step": 1, "forgotten": forgotten,
                               "epochs": epochs, "model": str(initial), **metrics}, f, indent=2)
            else:
                metrics = train_step(dataset, step, forgotten, model, base_train, test_set,
                                     epochs, lr, batch_size, device, out_dir, args.workers)
                model = torch.load(step_model_path, map_location="cpu", weights_only=False).to(device).eval()
            results[step] = metrics
        analysis_dir = config.WORK_DIR / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / f"sequential_{dataset}.json", "w") as f:
            json.dump(results, f, indent=2)
        summary[dataset] = results
    print("SEQUENTIAL_DONE", flush=True)


if __name__ == "__main__":
    main()
