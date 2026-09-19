"""Linear-probe recoverability of the unlearned class (representation-level check).

For each model checkpoint we extract the 2048-d penultimate features (avgpool
output of ResNet-50) on the test set and score how much target-class information
a *fresh* linear classifier can still recover:

  - multiclass 5-fold CV accuracy (out-of-fold),
  - target-class recall,
  - target-vs-rest AUC.

This metric is only meaningful when the backbone was allowed to change during
unlearning (full/layer4 fine-tuning); for frozen-head runs it necessarily
matches the original model.

Example:
  python -m src.analysis.linear_probe --dataset cifar10 --target-class 4 \
      --runs work/runs --tag-suffix _ft --device cuda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.core import common, config


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--target-class", type=int, default=4)
    parser.add_argument("--runs", default=str(config.RUNS_DIR))
    parser.add_argument("--tag-suffix", default="_ft", help="only runs whose dir name ends with this")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(config.WORK_DIR / "analysis" / "ft_probe"))
    parser.add_argument("--extra", nargs="*", default=[], help="extra name=path model specs")
    return parser


@torch.no_grad()
def extract_features(model, loader, device):
    features = []
    labels = []
    pooled = {}

    def hook(module, inputs, output):
        pooled["value"] = output.flatten(1)

    handle = model.avgpool.register_forward_hook(hook)
    try:
        for images, targets in loader:
            model(images.to(device))
            features.append(pooled["value"].cpu().numpy())
            labels.append(targets.numpy())
    finally:
        handle.remove()
    return np.concatenate(features), np.concatenate(labels)


def probe(features, labels, target_class, folds, seed):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial", n_jobs=-1),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prob = cross_val_predict(clf, features, labels, cv=cv, method="predict_proba", n_jobs=1)
    pred = prob.argmax(axis=1)
    acc = float((pred == labels).mean())
    target_recall = float((pred[labels == target_class] == target_class).mean())
    per_class_auc = {}
    for c in np.unique(labels):
        per_class_auc[int(c)] = float(roc_auc_score((labels == c).astype(int), prob[:, c]))
    retained = [v for k, v in per_class_auc.items() if k != target_class]
    return {
        "accuracy": acc,
        "target_recall": target_recall,
        "target_auc": per_class_auc[target_class],
        "retained_auc_mean": float(np.mean(retained)),
        "retained_auc_min": float(np.min(retained)),
        "per_class_auc": per_class_auc,
        "n": int(len(labels)),
    }


def main():
    args = build_argparser().parse_args()
    common.seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_set = common.load_cifar(args.dataset, train=False, transform=common.transform_224())
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    labels_full = np.array([int(y) for y in test_set.targets])

    models = {
        "original": Path(config.DATASETS[args.dataset]["model"]),
        "retrain": Path(config.DATASETS[args.dataset]["retrain"]),
    }
    runs_root = Path(args.runs)
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.endswith(args.tag_suffix):
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            cfg = json.load(f).get("config", {})
        if cfg.get("dataset") != args.dataset or int(cfg.get("target_class", -1)) != args.target_class:
            continue
        models[run_dir.name] = run_dir / "model.pkl"
    for spec in args.extra:
        name, path = spec.split("=", 1)
        models[name] = Path(path)

    results = {}
    for name, path in models.items():
        if not Path(path).exists():
            print(f"skip missing {name}: {path}")
            continue
        model = torch.load(path, map_location="cpu", weights_only=False).to(device).eval()
        features, labels = extract_features(model, loader, device)
        assert np.array_equal(labels, labels_full)
        results[name] = probe(features, labels, args.target_class, args.folds, args.seed)
        print(name, json.dumps(results[name]), flush=True)

    payload = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "folds": args.folds,
        "results": results,
    }
    (out_dir / "probe.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Linear-probe recoverability (penultimate features, test set)",
        "",
        f"dataset={args.dataset} target_class={args.target_class} folds={args.folds}",
        "",
        "| model | multiclass acc | target recall | target AUC | retained AUC mean | gap (retained-target) |",
        "|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        gap = res["retained_auc_mean"] - res["target_auc"]
        lines.append(
            f"| {name} | {res['accuracy']:.4f} | {res['target_recall']:.4f} | "
            f"{res['target_auc']:.4f} | {res['retained_auc_mean']:.4f} | {gap:+.4f} |"
        )
    (out_dir / "probe.md").write_text("\n".join(lines) + "\n")
    print(f"saved {out_dir/'probe.json'} and {out_dir/'probe.md'}")


if __name__ == "__main__":
    main()
