"""Post-training MIA pass.

Runs after all training has finished. For every saved run it computes:
  - Fr: the paper protocol — a linear SVM attack is fitted on the original
    model's probabilities for forget-class train (members) vs test
    (non-members) samples and transferred to the candidate model; Fr is the
    fraction of member samples predicted as non-members.
  - simple MIA: per-sample cross-entropy loss, logistic-regression attack with
    10-fold StratifiedShuffleSplit; gap = |accuracy - 0.5|.

The retrain reference is scored alongside, and original/retrain intermediates
are cached per (dataset, target class) so reruns and extra runs are cheap.

Writes <run>/mia.json for every run plus a merged table (default
revision/work/mia_final.md).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import common, config
from .mia import apply_fr, fit_fr_attack, simple_mia


def _forward(model, dataset, indices, device, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, list(indices)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    probs, losses = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            losses.append(loss_fn(logits, labels.to(device)).cpu().numpy())
    return np.concatenate(probs), np.concatenate(losses)


def cache_path(dataset, target_class):
    return config.WORK_DIR / "mia_cache" / f"{dataset}_c{target_class}.npz"


def load_reference_cache(dataset, target_class):
    path = cache_path(dataset, target_class)
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def save_reference_cache(dataset, target_class, payload):
    path = cache_path(dataset, target_class)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def build_reference(dataset, target_class, device, batch_size, num_workers, skip_retrain, seed):
    cached = load_reference_cache(dataset, target_class)
    if cached is not None:
        return cached
    transform = common.transform_224()
    train_set = common.load_cifar(dataset, train=True, transform=transform)
    test_set = common.load_cifar(dataset, train=False, transform=transform)
    train_idx = [i for i, y in enumerate(train_set.targets) if int(y) == target_class]
    test_idx = [i for i, y in enumerate(test_set.targets) if int(y) == target_class]

    original = common.load_classifier(dataset, device)
    orig_member_probs, _ = _forward(original, train_set, train_idx, device, batch_size, num_workers)
    orig_nonmember_probs, _ = _forward(original, test_set, test_idx, device, batch_size, num_workers)
    payload = {
        "orig_member_probs": orig_member_probs,
        "orig_nonmember_probs": orig_nonmember_probs,
    }

    if not skip_retrain:
        retrain_path = config.DATASETS[dataset].get("retrain")
        if retrain_path is not None and Path(retrain_path).exists():
            retrain = torch.load(retrain_path, map_location="cpu", weights_only=False).to(device).eval()
            rt_member_probs, rt_member_losses = _forward(retrain, train_set, train_idx, device, batch_size, num_workers)
            _, rt_nonmember_losses = _forward(retrain, test_set, test_idx, device, batch_size, num_workers)
            payload["rt_member_probs"] = rt_member_probs
            payload["rt_member_losses"] = rt_member_losses
            payload["rt_nonmember_losses"] = rt_nonmember_losses
            del retrain
            torch.cuda.empty_cache()

    save_reference_cache(dataset, target_class, payload)
    return {k: payload[k] for k in payload}


def collect_runs(runs_dir, datasets_filter):
    groups = {}
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        cfg = summary.get("config", {})
        dataset = cfg.get("dataset")
        if datasets_filter and dataset not in datasets_filter:
            continue
        if not (run_dir / "model.pkl").exists():
            continue
        groups.setdefault((dataset, int(cfg["target_class"])), []).append(run_dir)
    return groups


def row_from(summary, result, dataset, target_class):
    cfg = summary.get("config", {})
    retrain = result.get("retrain") or {}
    retrain_simple = retrain.get("simple_mia") or {}
    simple = result.get("simple_mia") or {}
    return {
        "dataset": dataset,
        "class": target_class,
        "name": config.CASES.get(dataset, {}).get(target_class, {}).get("name", ""),
        "mode": cfg.get("mode"),
        "labels": cfg.get("labels"),
        "integrity": cfg.get("integrity"),
        "seed": cfg.get("seed"),
        "fr": result.get("fr"),
        "sacc": simple.get("accuracy"),
        "sgap": simple.get("gap"),
        "rt_fr": retrain.get("fr"),
        "rt_sacc": retrain_simple.get("accuracy"),
        "rt_sgap": retrain_simple.get("gap"),
        "sec": result.get("seconds"),
    }


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(config.RUNS_DIR))
    parser.add_argument("--out", default=str(config.WORK_DIR / "mia_final.md"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="recompute runs that already have mia.json")
    parser.add_argument("--skip-retrain", action="store_true")
    parser.add_argument("--datasets", default=None, help="comma separated dataset filter")
    args = parser.parse_args()

    datasets_filter = set(args.datasets.split(",")) if args.datasets else None
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    groups = collect_runs(args.runs, datasets_filter)

    transform = common.transform_224()
    dataset_cache = {}

    def get_datasets(dataset):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = (
                common.load_cifar(dataset, train=True, transform=transform),
                common.load_cifar(dataset, train=False, transform=transform),
            )
        return dataset_cache[dataset]

    rows = []
    total = sum(len(v) for v in groups.values())
    done = 0
    for (dataset, target_class), run_dirs in sorted(groups.items()):
        reference = build_reference(
            dataset, target_class, device, args.batch_size, args.num_workers, args.skip_retrain, args.seed
        )
        attack = fit_fr_attack(reference["orig_member_probs"], reference["orig_nonmember_probs"], args.seed)
        fr_retrain = apply_fr(attack, reference["rt_member_probs"]) if "rt_member_probs" in reference else None
        mia_retrain = (
            simple_mia(reference["rt_member_losses"], reference["rt_nonmember_losses"], args.seed)
            if "rt_member_losses" in reference
            else None
        )
        train_set, test_set = get_datasets(dataset)
        train_idx = [i for i, y in enumerate(train_set.targets) if int(y) == target_class]
        test_idx = [i for i, y in enumerate(test_set.targets) if int(y) == target_class]

        for run_dir in run_dirs:
            done += 1
            mia_path = run_dir / "mia.json"
            with open(run_dir / "summary.json") as f:
                summary = json.load(f)
            if mia_path.exists() and not args.overwrite:
                with open(mia_path) as f:
                    result = json.load(f)
                rows.append(row_from(summary, result, dataset, target_class))
                print(f"[{done}/{total}] skip {run_dir.name} (mia.json exists)")
                continue
            start = time.time()
            model = torch.load(run_dir / "model.pkl", map_location="cpu", weights_only=False).to(device).eval()
            member_probs, member_losses = _forward(model, train_set, train_idx, device, args.batch_size, args.num_workers)
            _, nonmember_losses = _forward(model, test_set, test_idx, device, args.batch_size, args.num_workers)
            del model
            torch.cuda.empty_cache()

            result = {
                "run": run_dir.name,
                "config": summary.get("config"),
                "fr": apply_fr(attack, member_probs),
                "simple_mia": simple_mia(member_losses, nonmember_losses, args.seed),
                "retrain": {"fr": fr_retrain, "simple_mia": mia_retrain},
                "seconds": round(time.time() - start, 1),
                "generated": datetime.now().isoformat(timespec="seconds"),
            }
            with open(mia_path, "w") as f:
                json.dump(result, f, indent=2)

            rows.append(row_from(summary, result, dataset, target_class))
            print(
                f"[{done}/{total}] {run_dir.name}: Fr={result['fr']:.3f} "
                f"simpleMIA acc={result['simple_mia']['accuracy']:.3f} "
                f"gap={result['simple_mia']['gap']:.3f} ({result['seconds']}s)"
            )

    rows.sort(key=lambda r: (r["dataset"], r["class"], r["mode"] or "", r["labels"] or "", r["integrity"] or ""))
    lines = [
        "# MIA 汇总（SVM-Fr + simple MIA，自动生成）",
        "",
        "| dataset | class | mode | labels | integrity | seed | Fr | simple acc | simple gap | retrain gap | retrain Fr | sec |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {dataset} | {c} {name} | {mode} | {labels} | {integrity} | {seed} | {fr} | {sacc} | {sgap} | "
            "{rt_sgap} | {rt_fr} | {sec} |".format(
                dataset=r["dataset"], c=r["class"], name=r["name"], mode=r["mode"], labels=r["labels"],
                integrity=r["integrity"], seed=r["seed"], fr=fmt(r["fr"]), sacc=fmt(r["sacc"]),
                sgap=fmt(r["sgap"]), rt_sgap=fmt(r["rt_sgap"]), rt_fr=fmt(r["rt_fr"]), sec=r["sec"],
            )
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out_path} and {json_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
