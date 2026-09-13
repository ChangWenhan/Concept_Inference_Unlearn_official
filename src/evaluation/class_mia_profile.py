"""Per-class MIA profile: same protocol applied to every class of a dataset.

Purpose: decide whether the membership signal left on the forget class after
unlearning is class-specific residue or the ordinary per-class distinguishability
of the model family. For every class c we run the identical MIA protocol
(simple loss-based MIA, and the paper-style SVM-transfer Fr) on three models:
the original model, the unlearned model, and the retrain reference.

Outputs (work dir):
  runs/<tag>/class_mia.json      per-class results for the three models
  class_mia_profile.json         combined payload for both datasets
  class_mia_profile.csv          flat table
  class_mia_profile.md           human-readable summary + full tables
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.core import common, config
from src.core.mia import apply_fr, fit_fr_attack, simple_mia


@torch.no_grad()
def collect_all(model, dataset, device, batch_size=128, num_workers=4):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    probs, losses, labels = [], [], []
    model.eval()
    for images, targets in loader:
        out = model(images.to(device))
        probs.append(torch.softmax(out, dim=1).cpu().numpy())
        losses.append(loss_fn(out, targets.to(device)).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs), np.concatenate(losses), np.concatenate(labels)


def per_class_indices(labels):
    return {int(c): np.where(labels == c)[0] for c in np.unique(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "ham10000"])
    parser.add_argument("--target-class", required=True, type=int, help="the unlearned class")
    parser.add_argument("--unlearned-run", required=True, help="run dir of the unlearned model")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    transform = common.transform_224()
    train_set = common.load_cifar(args.dataset, train=True, transform=transform)
    test_set = common.load_cifar(args.dataset, train=False, transform=transform)
    train_targets = np.array([int(y) for y in train_set.targets])
    test_targets = np.array([int(y) for y in test_set.targets])
    train_idx = per_class_indices(train_targets)
    test_idx = per_class_indices(test_targets)

    run_dir = Path(args.unlearned_run)
    models = {
        "original": common.load_classifier(args.dataset, device),
        "unlearned": torch.load(run_dir / "model.pkl", map_location="cpu", weights_only=False).to(device).eval(),
        "retrain": torch.load(config.DATASETS[args.dataset]["retrain"], map_location="cpu", weights_only=False).to(device).eval(),
    }

    data = {}
    for name, model in models.items():
        tr_probs, tr_losses, _ = collect_all(model, train_set, device)
        te_probs, te_losses, _ = collect_all(model, test_set, device)
        data[name] = {"train_probs": tr_probs, "train_losses": tr_losses,
                      "test_probs": te_probs, "test_losses": te_losses}
        print(f"collected {name}", flush=True)

    results = {name: {} for name in models}
    classes = sorted(train_idx.keys())
    orig = data["original"]
    for c in classes:
        attack = fit_fr_attack(orig["train_probs"][train_idx[c]], orig["test_probs"][test_idx[c]])
        for name in models:
            d = data[name]
            gap = simple_mia(d["train_losses"][train_idx[c]], d["test_losses"][test_idx[c]])["gap"]
            fr = apply_fr(attack, d["train_probs"][train_idx[c]])
            results[name][c] = {"simple_gap": gap, "fr": fr}

    payload = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "unlearned_run": str(run_dir),
        "classes": classes,
        "results": results,
    }
    (run_dir / "class_mia.json").write_text(json.dumps(payload, indent=2))
    profile_path = config.WORK_DIR / "class_mia_profile.json"
    combined = {}
    if profile_path.exists():
        try:
            existing = json.loads(profile_path.read_text())
            if isinstance(existing, dict) and "classes" not in existing:
                combined = existing
        except json.JSONDecodeError:
            combined = {}
    combined[args.dataset] = payload
    profile_path.write_text(json.dumps(combined, indent=2))

    per_csv = config.WORK_DIR / f"class_mia_profile_{args.dataset}.csv"
    header = ["dataset", "class", "is_target", "model", "simple_gap", "fr"]
    rows = []
    for c in classes:
        for name in models:
            rows.append([args.dataset, c, int(c == args.target_class), name,
                         results[name][c]["simple_gap"], results[name][c]["fr"]])
    with open(per_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    merged_csv = config.WORK_DIR / "class_mia_profile.csv"
    kept = []
    if merged_csv.exists():
        with open(merged_csv, newline="") as f:
            reader = csv.reader(f)
            kept = [r for i, r in enumerate(reader) if i == 0 or r[0] != args.dataset]
    with open(merged_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(kept if kept else [header])
        if not kept:
            kept = [header]
        writer.writerows(rows)

    def summarize(model_name, metric):
        vals = [results[model_name][c][metric] for c in classes]
        target = results[model_name][args.target_class][metric]
        controls = [results[model_name][c][metric] for c in classes if c != args.target_class]
        rank = 1 + sum(1 for v in vals if v < target)
        return target, rank, len(vals), float(np.mean(controls)), float(np.min(controls)), float(np.max(controls))

    lines = [
        f"# Per-class MIA profile — {args.dataset} (target class {args.target_class})",
        "",
        "Protocol: for every class c, members = class-c training images, non-members = class-c test images.",
        "`simple_gap` = |accuracy-0.5| of the loss-based LR attack (10-fold); `fr` = paper SVM-transfer",
        "forgetting rate (attack fitted on the ORIGINAL model, applied to each model).",
        "",
    ]
    for metric in ["simple_gap", "fr"]:
        lines.append(f"## Summary ({metric})")
        lines.append("")
        lines.append("| model | target value | rank of target | controls mean | controls min | controls max |")
        lines.append("|---|---|---|---|---|---|")
        for name in models:
            t, r, n, mu, lo, hi = summarize(name, metric)
            lines.append(f"| {name} | {t:.3f} | {r}/{n} | {mu:.3f} | {lo:.3f} | {hi:.3f} |")
        lines.append("")
    lines.append("## Full table (simple_gap / fr)")
    lines.append("")
    lines.append("| class | target? | original gap | unlearned gap | retrain gap | original Fr | unlearned Fr | retrain Fr |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in classes:
        lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            c, "**yes**" if c == args.target_class else "",
            results["original"][c]["simple_gap"], results["unlearned"][c]["simple_gap"], results["retrain"][c]["simple_gap"],
            results["original"][c]["fr"], results["unlearned"][c]["fr"], results["retrain"][c]["fr"]))
    md_path = config.WORK_DIR / f"class_mia_profile_{args.dataset}.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {run_dir / 'class_mia.json'}, {config.WORK_DIR / 'class_mia_profile.json'}, "
          f"{per_csv}, {merged_csv}, {md_path}", flush=True)


if __name__ == "__main__":
    main()
