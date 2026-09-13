"""Paper-ready supplementary analysis: label flipping / random labels vs the
concept-localized masking method on class forgetting.

Focused, symmetric comparison on the two representative classes of the main
table (CIFAR-10 deer, CIFAR-100 boy), 3 seeds each, three arms:

  ours          localized + targeted   (concept mask + donor label)
  label_flip    none      + targeted   (full-image relabel, strongest label-only variant)
  random_labels none      + random     (Random labels poisoning baseline)

Part A  training trajectories from the archived runs -> exactness, rebound,
        epochs-to-forget, retained/global accuracy.
Part B  (--probe) model behaviour: where the forgotten-class test images are
        routed (donor / target / other) and with what confidence; donor-class
        preservation.
Part C  MIA (SVM-transfer Fr + loss-based gap) per run.

Outputs (work/analysis/label_flip_vs_mask/):
  summary.md, summary.json, per_seed.csv, aggregate.csv,
  probe.json, mia.json, fig_trajectories.{png,pdf}, fig_destinations.{png,pdf}
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.core import common, config

ARMS = {
    "ours": "localized_targeted_full_s{seed}",
    "label_flip": "none_targeted_full_s{seed}",
    "random_labels": "none_random_full_s{seed}",
}
ARM_LABELS = {
    "ours": "Ours (localized+targeted)",
    "label_flip": "Label flip (targeted)",
    "random_labels": "Random labels",
}
COLORS = {"ours": "#1f77b4", "label_flip": "#2ca02c", "random_labels": "#d62728"}
DONOR_FALLBACK = {"cifar10": 0, "cifar100": 1}


def archive_root():
    return config.REPO_ROOT / "archive" / "experiments" / "02_core_ablation"


def run_dir(dataset, target_class, arm, seed):
    return archive_root() / f"{dataset}_c{target_class}_{ARMS[arm].format(seed=seed)}"


def load_history(path):
    with open(path / "epochs.jsonl") as f:
        return [json.loads(line) for line in f]


def trajectory_stats(history):
    trains = np.array([e["train_target_acc"] for e in history])
    tests = np.array([e["test_target_acc"] for e in history])
    globs = np.array([e["test_global_acc"] for e in history])
    retained = np.array([e.get("test_retained_acc", np.nan) for e in history])
    forgotten = (trains <= 0.01) & (tests <= 0.01)
    first = int(np.argmax(forgotten)) + 1 if forgotten.any() else 0
    rebounds = 0
    if first:
        for i in range(first, len(trains)):
            if trains[i] > 0.01 and trains[i - 1] <= 0.01:
                rebounds += 1
    return {
        "final_train": float(trains[-1]),
        "final_test": float(tests[-1]),
        "max_train": float(trains.max()),
        "max_test": float(tests.max()),
        "epochs_over_1pct": int((trains > 0.01).sum()),
        "first_epoch_forgotten": first,
        "rebound_after_first_zero": float(trains[first - 1:].max()) if first else None,
        "rebound_count": rebounds,
        "min_retained": float(np.nanmin(retained)),
        "final_retained": float(retained[-1]),
        "min_global": float(globs.min()),
        "final_global": float(globs[-1]),
        "final_loss": float(history[-1]["loss"]),
        "train_seconds": float(sum(e.get("epoch_seconds", 0.0) for e in history)),
    }


AGG_KEYS = ("final_train", "final_test", "max_train", "epochs_over_1pct", "first_epoch_forgotten",
            "rebound_after_first_zero", "rebound_count", "min_retained", "final_retained",
            "min_global", "final_global", "final_loss", "train_seconds")


def arm_summary(stats_list):
    out = {}
    for key in AGG_KEYS:
        vals = [s[key] for s in stats_list if s.get(key) is not None]
        if not vals:
            continue
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "values": [round(float(v), 6) for v in vals]}
    return out


def load_mia(path):
    with open(path) as f:
        data = json.load(f)
    out = {"fr": data.get("fr")}
    simple = data.get("simple_mia") or {}
    out["simple_gap"] = simple.get("gap")
    return out


def donor_of(run_path, dataset):
    with open(run_path / "summary.json") as f:
        cfg = json.load(f).get("config", {})
    return int(cfg.get("donor_class", DONOR_FALLBACK[dataset]))


@torch.no_grad()
def probe_model(model, images, target_class, donor_class, device, batch=64):
    preds, max_prob, donor_prob = [], [], []
    for start in range(0, len(images), batch):
        prob = torch.softmax(model(images[start:start + batch].to(device)), dim=1)
        preds.append(prob.argmax(dim=1).cpu())
        max_prob.append(prob.max(dim=1).values.cpu())
        donor_prob.append(prob[:, donor_class].cpu())
    preds = torch.cat(preds).numpy()
    max_prob = torch.cat(max_prob).numpy()
    donor_prob = torch.cat(donor_prob).numpy()
    counts = np.bincount(preds, minlength=int(max(preds.max(), donor_class, target_class)) + 1)
    return {
        "n": int(len(preds)),
        "donor_fraction": float((preds == donor_class).mean()),
        "target_fraction": float((preds == target_class).mean()),
        "other_fraction": float(1.0 - (preds == donor_class).mean() - (preds == target_class).mean()),
        "mean_max_prob": float(max_prob.mean()),
        "mean_donor_prob": float(donor_prob.mean()),
        "pred_counts": {int(i): int(c) for i, c in enumerate(counts) if c},
    }


def make_figures(datasets, summary, probe, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for arm in ARMS:
            hist = []
            for seed in summary[dataset][arm]["per_seed"]:
                path = run_dir(dataset, 4 if dataset == "cifar10" else 11, arm, seed["seed"])
                hist.append([e["train_target_acc"] * 100 for e in load_history(path)])
            if not hist:
                continue
            arr = np.array(hist)
            x = np.arange(1, arr.shape[1] + 1)
            for row in arr:
                ax.plot(x, row, color=COLORS[arm], alpha=0.25, linewidth=0.8)
            ax.plot(x, arr.mean(0), color=COLORS[arm], linewidth=2, label=ARM_LABELS[arm])
            ax.fill_between(x, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                            color=COLORS[arm], alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Target-class train accuracy (%)")
        ax.set_ylim(-2, 100)
        ax.grid(alpha=0.3)
        ax.set_title(dataset)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_trajectories.png", dpi=200)
    fig.savefig(out_dir / "fig_trajectories.pdf")
    plt.close(fig)

    if not probe:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        arms = list(ARMS)
        x = np.arange(len(arms))
        donor = [100 * np.mean([r["target_images"]["donor_fraction"] for r in probe[dataset][a]]) for a in arms]
        other = [100 * np.mean([r["target_images"]["other_fraction"] for r in probe[dataset][a]]) for a in arms]
        target = [100 * np.mean([r["target_images"]["target_fraction"] for r in probe[dataset][a]]) for a in arms]
        conf = [np.mean([r["target_images"]["mean_max_prob"] for r in probe[dataset][a]]) for a in arms]
        width = 0.6
        ax.bar(x, donor, width, label="to donor class", color="#4c72b0")
        ax.bar(x, other, width, bottom=donor, label="to other classes", color="#c8c8c8")
        ax.bar(x, target, width, bottom=np.array(donor) + np.array(other), label="still target class",
               color="#d62728")
        for xi, c in zip(x, conf):
            ax.text(xi, 103, f"conf={c:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS[a].replace(" (", "\n(") for a in arms], fontsize=8)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Share of target-class test images (%)")
        ax.set_title(dataset)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_destinations.png", dpi=200)
    fig.savefig(out_dir / "fig_destinations.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = config.WORK_DIR / "analysis" / "label_flip_vs_mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {"cifar10": 4, "cifar100": 11}

    summary = {}
    mia_all = {}
    per_seed_rows = []
    for dataset, target_class in datasets.items():
        summary[dataset] = {}
        mia_all[dataset] = {}
        for arm in ARMS:
            stats_list = []
            for seed in args.seeds:
                path = run_dir(dataset, target_class, arm, seed)
                if not path.exists():
                    continue
                stats = trajectory_stats(load_history(path))
                stats["seed"] = seed
                mia = load_mia(path / "mia.json") if (path / "mia.json").exists() else {}
                stats.update(mia)
                stats_list.append(stats)
                per_seed_rows.append({"dataset": dataset, "arm": arm, **stats})
            if stats_list:
                summary[dataset][arm] = {"per_seed": stats_list, "aggregate": arm_summary(stats_list)}
                mia_all[dataset][arm] = {
                    "fr": [s.get("fr") for s in stats_list],
                    "simple_gap": [s.get("simple_gap") for s in stats_list],
                }

    lines = ["# Label flipping vs concept-localized masking (paper analysis)", "",
             "Arms: `ours` = localized+targeted; `label_flip` = none+targeted; "
             "`random_labels` = none+random. Seeds: " + ", ".join(map(str, args.seeds)) + ".", ""]
    for dataset, arms in summary.items():
        lines += [f"## {dataset}", "",
                  "| arm | final A_train | max A_train | epochs>1% | first zero | rebounds | "
                  "final A_test | final retained | donor Fr | simple MIA gap |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for arm, data in arms.items():
            a = data["aggregate"]
            f = lambda k: f"{a[k]['mean']:.4f}±{a[k]['std']:.4f}" if a.get(k) and a[k]['std'] else \
                (f"{a[k]['mean']:.4f}" if a.get(k) else "-")
            fr = mia_all[dataset][arm]["fr"]
            gap = mia_all[dataset][arm]["simple_gap"]
            fr_s = f"{np.mean(fr):.3f}" if any(v is not None for v in fr) else "-"
            gap_s = f"{np.nanmean(gap):.3f}" if any(v is not None for v in gap) else "-"
            lines.append(f"| {arm} | {f('final_train')} | {f('max_train')} | {f('epochs_over_1pct')} | "
                         f"{f('first_epoch_forgotten')} | {f('rebound_count')} | {f('final_test')} | "
                         f"{f('final_retained')} | {fr_s} | {gap_s} |")
        lines.append("")

    probe = None
    if args.probe:
        probe = {}
        for dataset, target_class in datasets.items():
            donor = donor_of(run_dir(dataset, target_class, "ours", args.seeds[0]), dataset)
            test = common.load_cifar(dataset, train=False, transform=common.transform_224())
            targets = list(test.targets)
            t_imgs = torch.stack([test[i][0] for i, y in enumerate(targets) if int(y) == target_class])
            d_imgs = torch.stack([test[i][0] for i, y in enumerate(targets) if int(y) == donor])
            probe[dataset] = {}
            for arm in ARMS:
                per_seed = []
                for seed in args.seeds:
                    path = run_dir(dataset, target_class, arm, seed) / "model.pkl"
                    if not path.exists():
                        continue
                    model = torch.load(path, map_location="cpu", weights_only=False).to(args.device).eval()
                    per_seed.append({
                        "seed": seed,
                        "target_images": probe_model(model, t_imgs, target_class, donor, args.device),
                        "donor_images": probe_model(model, d_imgs, target_class, donor, args.device),
                    })
                if per_seed:
                    probe[dataset][arm] = per_seed
        (out_dir / "probe.json").write_text(json.dumps(probe, indent=2))
        for dataset, arms in probe.items():
            lines += [f"## probe {dataset}", "",
                      "| arm | target images -> donor | -> other | still target | mean max-prob | donor-class acc |",
                      "|---|---|---|---|---|---|"]
            for arm, runs in arms.items():
                t = [r["target_images"] for r in runs]
                d = [r["donor_images"] for r in runs]
                lines.append(f"| {arm} | {np.mean([x['donor_fraction'] for x in t]):.3f} | "
                             f"{np.mean([x['other_fraction'] for x in t]):.3f} | "
                             f"{np.mean([x['target_fraction'] for x in t]):.3f} | "
                             f"{np.mean([x['mean_max_prob'] for x in t]):.3f} | "
                             f"{np.mean([x['donor_fraction'] for x in d]):.3f} |")
            lines.append("")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "mia.json").write_text(json.dumps(mia_all, indent=2))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    fields = ["dataset", "arm", "seed"] + list(AGG_KEYS) + ["fr", "simple_gap"]
    with open(out_dir / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_seed_rows)
    with open(out_dir / "aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "arm", "metric", "mean", "std"])
        for dataset, arms in summary.items():
            for arm, data in arms.items():
                for key, val in data["aggregate"].items():
                    w.writerow([dataset, arm, key, round(val["mean"], 6), round(val["std"], 6)])

    make_figures(datasets, summary, probe, out_dir)
    print(f"saved summary/mia/csv/figures to {out_dir}")


if __name__ == "__main__":
    main()
