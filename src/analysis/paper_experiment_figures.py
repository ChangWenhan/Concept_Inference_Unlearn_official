"""Paper figures for the image- and LLM-unlearning experiments.

One plot per PDF; the LaTeX file assembles them as subfigures (at most four
per row). The style follows the clawresearch/RISA figure scripts: serif fonts,
Okabe-Ito palette, thin spines, light grid, compact legends.

Outputs (into the paper's pictures/ directory):
  fig_panel_celd_{cifar10,cifar100}_{arm}.pdf                CE histograms (8 panels)
  fig_panel_traj_{cifar10,cifar100}_{forget,retain}.pdf      training trajectories
  fig_panel_coverage_{llama2,vicuna,qwen}.pdf                TOFU coverage curves
  fig_panel_poison_{original,heatmap,localized,center}.pdf   poisoning process
  fig_panel_case_{rank,original,poison}.pdf                  case study

Usage:
  PYTHONNOUSERSITE=1 python -m src.analysis.paper_experiment_figures
"""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.label_flip_vs_mask import load_history, run_dir

PAPER_PICS = Path("/home/cwh/Workspace/Machine-Unlearning-Paper/pictures")
SEEDS = [42, 43, 44]
CASES = {"cifar10": {"target_class": 4, "label": "CIFAR-10"},
         "cifar100": {"target_class": 11, "label": "CIFAR-100"}}
PROBE = Path("work/analysis/label_flip_vs_mask/probe.json")
QS_DIR = Path("llm-unlearning/results/aggregated/question_sets")
CELD_CACHE = Path("work/analysis/paper_celd_cache.npz")

ARMS = {"ours": "Ours", "label_flip": "Label flip", "random_labels": "Random labels"}
ARM_COLORS = {"ours": "#0072B2", "label_flip": "#009E73", "random_labels": "#D55E00"}
MODELS = [("llama2", "Llama-2-7B", ""), ("vicuna", "Vicuna-7B", "_vicuna"), ("qwen", "Qwen2.5-7B", "_qwen")]
MODEL_COLORS = {"llama2": "#0072B2", "vicuna": "#009E73", "qwen": "#D55E00"}
SPLITS = [("2authors", "forget01", 250), ("10authors", "forget05", 700),
          ("20authors", "forget10", 1700)]
SPLIT_COLORS = ["#0072B2", "#009E73", "#D55E00"]
SPLIT_OFFSETS = {"forget01": (-13, 8), "forget05": (13, -6), "forget10": (-4, -12)}
THRESHOLD = 0.95


def setup_style():
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    serif_font = next((f for f in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "FreeSerif")
                       if f in installed), "serif")
    plt.rcParams.update({
        "font.size": 8,
        "font.family": "serif",
        "font.serif": [serif_font],
        "mathtext.fontset": "custom",
        "mathtext.rm": serif_font,
        "mathtext.it": f"{serif_font}:italic",
        "mathtext.bf": f"{serif_font}:bold",
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name):
    fig.savefig(PAPER_PICS / f"{name}.pdf")
    plt.close(fig)
    print(f"saved {name}.pdf")


def style_axes(ax, axis="y"):
    ax.grid(axis=axis, color="#D0D0D0", linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- CELD -----
CELD_CACHE_FULL = Path("work/analysis/paper_celd_full_cache.npz")
CELD_GROUPS = [("target", "Target class", "#0072B2"),
               ("train", "Non-target (train)", "#009E73"),
               ("test", "Non-target (test)", "#E69F00")]
CELD_ORDER = [("original", "Original"), ("retrain", "Retrained"),
              ("ours", "Ours"), ("random", "Random labels")]


def per_sample_ce(model, dataset, device, batch_size=128):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    losses = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            losses.append(F.cross_entropy(logits, labels.to(device),
                                          reduction="none").cpu().numpy())
    return np.concatenate(losses)


def celd_data():
    if CELD_CACHE_FULL.exists():
        return dict(np.load(CELD_CACHE_FULL.resolve()))
    import torch
    from torch.utils.data import Subset

    from src.core import common, config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    for dataset_key, cfg in CASES.items():
        train = common.load_cifar(dataset_key, train=True, transform=common.transform_224())
        test = common.load_cifar(dataset_key, train=False, transform=common.transform_224())
        targets = [int(y) for y in train.targets]
        target_idx = [i for i, y in enumerate(targets) if y == cfg["target_class"]]
        n = len(target_idx)
        other_idx = [i for i, y in enumerate(targets) if y != cfg["target_class"]][:n]
        test_idx = [i for i, y in enumerate(test.targets) if int(y) != cfg["target_class"]]
        subsets = {"target": Subset(train, target_idx), "train": Subset(train, other_idx),
                   "test": Subset(test, test_idx)}
        models = {
            "original": common.load_classifier(dataset_key, device),
            "retrain": torch.load(config.DATASETS[dataset_key]["retrain"],
                                  map_location="cpu", weights_only=False).to(device).eval(),
        }
        from src.analysis.label_flip_vs_mask import run_dir
        for arm, suffix in (("ours", "localized_targeted_full_s42"),
                            ("random", "none_random_full_s42")):
            run = run_dir(dataset_key, cfg["target_class"], "ours" if arm == "ours" else "random_labels", 42)
            models[arm] = torch.load(run / "model.pkl", map_location="cpu",
                                     weights_only=False).to(device).eval()
        for arm, model in models.items():
            for group, subset in subsets.items():
                values = per_sample_ce(model, subset, device)
                data[f"{dataset_key}_{arm}_{group}"] = values
                print(f"{dataset_key} {arm} {group}: median {np.median(values):.3f}")
    CELD_CACHE_FULL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CELD_CACHE_FULL, **data)
    return data


def figures_celd():
    data = celd_data()
    bins = np.linspace(0, 20, 11)
    width = (bins[1] - bins[0]) / len(CELD_GROUPS) * 0.86
    for dataset_key in CASES:
        # common y-limit across the four panels of the row
        top = 0.0
        hists = {}
        for arm, _ in CELD_ORDER:
            for index, (group, _, _) in enumerate(CELD_GROUPS):
                values = np.clip(data[f"{dataset_key}_{arm}_{group}"], 0, bins[-1])
                counts, edges = np.histogram(values, bins=bins,
                                             weights=np.ones_like(values) / len(values))
                hists[(arm, group)] = (counts, edges)
                top = max(top, counts.max())
        for arm, label in CELD_ORDER:
            plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                                 "xtick.labelsize": 8, "ytick.labelsize": 8})
            fig, ax = plt.subplots(figsize=(2.05, 1.55))
            for index, (group, glabel, color) in enumerate(CELD_GROUPS):
                counts, edges = hists[(arm, group)]
                centers = (edges[:-1] + edges[1:]) / 2
                offset = (index - 1) * width
                ax.bar(centers + offset, counts, width=width, color=color,
                       edgecolor="white", linewidth=0.15, label=glabel)
            ax.legend(loc="upper right", frameon=False, fontsize=7.2,
                      handlelength=0.9, handletextpad=0.3, labelspacing=0.25,
                      borderaxespad=0.2)
            ax.set_xlim(0, bins[-1])
            ax.set_ylim(0, top * 1.08)
            ax.set_xlabel("Cross-entropy loss (nats)")
            ax.set_ylabel("Fraction of samples")
            style_axes(ax)
            fig.tight_layout(pad=0.25)
            save(fig, f"fig_panel_celd_{dataset_key}_{arm}")
            plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                                 "xtick.labelsize": 7, "ytick.labelsize": 7,
                                 "legend.fontsize": 6.2})


# ---------------------------------------------------------- trajectories ----
TRAJ_METRICS = [("forget", "train_target_acc", "Target-class accuracy"),
                ("retain", "test_retained_acc", "Retained accuracy")]


def trajectory_curves(dataset, case, field):
    curves = {}
    for arm in ARMS:
        series = []
        for seed in SEEDS:
            history = load_history(run_dir(dataset, case["target_class"], arm, seed))
            series.append([e[field] for e in history])
        length = min(len(s) for s in series)
        mat = np.array([s[:length] for s in series])
        curves[arm] = (mat.mean(axis=0), mat.min(axis=0), mat.max(axis=0))
    return curves


def figures_traj():
    for dataset_key, cfg in CASES.items():
        for metric, field, ylabel in TRAJ_METRICS:
            curves = trajectory_curves(dataset_key, cfg, field)
            plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                                 "xtick.labelsize": 9, "ytick.labelsize": 9,
                                 "legend.fontsize": 8.5})
            fig, ax = plt.subplots(figsize=(2.35, 1.85))
            top = 0.0
            bottom = 1.0
            for arm, label in ARMS.items():
                mean, lo, hi = curves[arm]
                x = np.arange(1, len(mean) + 1)
                ax.fill_between(x, lo, hi, color=ARM_COLORS[arm], alpha=0.18, linewidth=0)
                ax.plot(x, mean, color=ARM_COLORS[arm], linewidth=1.1, label=label)
                top = max(top, hi.max())
                bottom = min(bottom, lo.min())
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.5, len(curves["ours"][0]) + 0.5)
            if metric == "forget":
                ax.set_ylim(0.0, top * 1.08)
                legend_anchor = "upper right"
            else:
                span = max(top - bottom, 1e-6)
                ax.set_ylim(bottom - 0.60 * span, top + 0.12 * span)
                legend_anchor = "lower right"
            style_axes(ax)
            ax.legend(loc=legend_anchor, frameon=False, fontsize=8,
                      handlelength=1.2, handletextpad=0.3, labelspacing=0.22,
                      borderaxespad=0.3)
            fig.tight_layout(pad=0.25)
            save(fig, f"fig_panel_traj_{dataset_key}_{metric}")
            plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                                 "xtick.labelsize": 7, "ytick.labelsize": 7,
                                 "legend.fontsize": 6.2})


# ------------------------------------------------------------- coverage -----
def coverage_curve(path, seed=42):
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(records)
    authors = sorted({r["author"] for r in records})
    seen = {a: set() for a in authors}
    xs, ys = [], []
    for n, record in enumerate(records, 1):
        seen[record["author"]].add(record["relation"])
        covered = sum(len(seen[a]) for a in authors)
        xs.append(n)
        ys.append(covered / (20 * len(authors)))
    return np.array(xs), np.array(ys)


def figures_coverage():
    max_x = 3000
    for model, name, suffix in MODELS:
        plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5,
                             "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                             "legend.fontsize": 7.5, "axes.titlesize": 8.5})
        fig, ax = plt.subplots(figsize=(2.35, 1.9))
        for (split_label, split_key, _), color in zip(SPLITS, SPLIT_COLORS):
            path = QS_DIR / f"dual__{split_key}_qa_final{suffix}.jsonl"
            xs, ys = coverage_curve(path)
            n_authors = split_label.split("a")[0]
            ax.plot(xs, ys, color=color, linewidth=1.1, label=f"{n_authors} authors")
            hit = np.argmax(ys >= THRESHOLD)
            if ys[hit] >= THRESHOLD:
                ax.plot(xs[hit], ys[hit], "o", color=color, markersize=3.8, zorder=5)
                ax.annotate(f"{xs[hit]}", (xs[hit], ys[hit]), textcoords="offset points",
                            xytext=SPLIT_OFFSETS[split_key], fontsize=8, zorder=6,
                            color=color, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                                      edgecolor="none", alpha=0.85))
        ax.axhline(THRESHOLD, color="#555555", linestyle="--", linewidth=0.65)
        ax.set_xscale("log")
        ax.set_xlim(1, max_x)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("Number of questions")
        ax.set_ylabel("Coverage of relation facts")
        style_axes(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=7.5,
                  handlelength=1.2, handletextpad=0.3, labelspacing=0.25)
        fig.tight_layout(pad=0.25)
        save(fig, f"fig_panel_coverage_{model}")
        plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                             "xtick.labelsize": 7, "ytick.labelsize": 7,
                             "legend.fontsize": 6.2})


# ------------------------------------------------------- sequential ---------
SEQ_ORDER = [("cifar10", "CIFAR-10", "#0072B2"),
             ("cifar100", "CIFAR-100", "#009E73"),
             ("ham10000", "HAM10000", "#D55E00")]
SEQ_PANELS = [("A_train", "Target-class accuracy (train)"),
              ("A_test", "Target-class accuracy (test)"),
              ("retained", "Retained accuracy"),
              ("Fr", "Forgetting rate")]


def figures_sequential():
    data = {}
    for key, name, color in SEQ_ORDER:
        path = Path("work/analysis") / f"sequential_{key}.json"
        if not path.exists():
            print(f"sequential data missing for {key}: {path}")
            continue
        data[key] = json.loads(path.read_text())
    if not data:
        return
    for metric, ylabel in SEQ_PANELS:
        plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                             "xtick.labelsize": 9, "ytick.labelsize": 9,
                             "legend.fontsize": 8.5})
        fig, ax = plt.subplots(figsize=(2.35, 1.85))
        top, low = 0.0, 1.0
        for key, name, color in SEQ_ORDER:
            if key not in data:
                continue
            steps = sorted(data[key], key=int)
            xs = [int(s) for s in steps]
            ys = [data[key][s][metric] for s in steps]
            ax.plot(xs, ys, marker="o", markersize=3.2, color=color,
                    linewidth=1.1, label=name)
            top, low = max(top, max(ys)), min(low, min(ys))
        ax.set_xlabel("Number of unlearned classes")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.85, 4.15)
        if metric in ("A_train", "A_test"):
            ax.set_ylim(-0.005, max(0.05, top * 1.5))
            legend_anchor = "upper right"
        else:
            span = max(0.05, top - low)
            ax.set_ylim(low - 0.65 * span, top + 0.12 * span)
            legend_anchor = "lower right"
        style_axes(ax)
        ax.legend(loc=legend_anchor, frameon=False, fontsize=8.5,
                  handlelength=1.2, handletextpad=0.3, labelspacing=0.22)
        fig.tight_layout(pad=0.25)
        save(fig, f"fig_panel_seq_{metric}")
        plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                             "xtick.labelsize": 7, "ytick.labelsize": 7,
                             "legend.fontsize": 6.2})


# ------------------------------------------------- poisoning and case study -
def figures_images():
    import torch
    import torch.nn.functional as F

    from src.analysis.analyze_localization import patch_sim_at
    from src.core import common, concept_tools

    device = "cuda" if torch.cuda.is_available() else "cpu"
    common.seed_all(42)
    clip_model = concept_tools.load_clip(device=device)
    bank = concept_tools.load_concept_bank("cifar10")
    vec = concept_tools.concept_vector(bank, "antler").view(1, -1)
    raw = common.load_cifar("cifar10", train=True, transform=common.transform_224())
    targets = [int(y) for y in raw.targets]
    deer = [i for i, y in enumerate(targets) if y == 4]
    planes = [i for i, y in enumerate(targets) if y == 0]

    def locate(image):
        sim, _, _ = concept_tools.patch_similarity(clip_model, image, vec)
        center, _, peak = concept_tools.locate_peak(sim, stride=16, patch=64)
        gain = peak - patch_sim_at(sim, 16, 64, (112, 112))
        return sim, center, gain

    # donor patch from a plane image at its "propellers" peak
    donor_vec = concept_tools.concept_vector(bank, "propellers").view(1, -1)
    donor_image = raw[planes[7]][0].unsqueeze(0)
    donor_sim, _, _ = concept_tools.patch_similarity(clip_model, donor_image, donor_vec)
    donor_center, _, _ = concept_tools.locate_peak(donor_sim, stride=16, patch=64)
    donor_patch = concept_tools.crop_patch(donor_image, donor_center, 64)

    records = []
    for index in deer[:500]:
        image = raw[index][0].unsqueeze(0)
        sim, center, gain = locate(image)
        records.append({"index": index, "img": image, "sim": sim, "center": center,
                        "gain": gain})
    by_index = {r["index"]: r for r in records}
    main_example = by_index[711]
    case_example = by_index[3044]

    def tensor_image(tensor):
        return tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    def save_image(array, name, marker=None, overlay=None):
        fig, ax = plt.subplots(figsize=(1.62, 1.62))
        ax.imshow(array)
        if overlay is not None:
            ax.imshow(overlay, cmap="jet", alpha=0.45,
                      extent=[-0.5, 223.5, 223.5, -0.5])
        if marker is not None:
            ax.plot(marker[0], marker[1], "o", markersize=6, markerfacecolor="none",
                    markeredgecolor="white", markeredgewidth=1.3)
            ax.plot(112, 112, "x", color="white", markersize=5, markeredgewidth=1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(0, 0, 1, 1)
        save(fig, name)

    # ---- poisoning-process panels ----
    image = main_example["img"]
    patch = F.interpolate(donor_patch, size=(64, 64), mode="bilinear",
                          align_corners=False)
    localized = concept_tools.paste_patch(image, patch, main_example["center"])
    centered = concept_tools.paste_patch(image, patch, (112, 112))
    save_image(tensor_image(image), "fig_panel_poison_original")
    save_image(tensor_image(image), "fig_panel_poison_heatmap",
               marker=main_example["center"], overlay=main_example["sim"])
    save_image(tensor_image(localized), "fig_panel_poison_localized")
    save_image(tensor_image(centered), "fig_panel_poison_center")

    # ---- corruption example panels (severity 5, as used in Table tab:corrupt) ----
    from src.corruption.noisy_exp import corrupt_batch

    for kind in ("brightness", "defocus_blur", "gaussian_noise"):
        save_image(tensor_image(corrupt_batch(case_example["img"], kind)),
                   f"fig_panel_corrupt_cifar10_{kind}")
    ham = common.load_cifar("ham10000", train=True, transform=common.transform_224())
    ham_targets = [int(y) for y in ham.targets]
    mel = [i for i, y in enumerate(ham_targets) if y == 4]
    ham_image = ham[mel[0]][0].unsqueeze(0)
    save_image(tensor_image(ham_image), "fig_panel_corrupt_ham10000_original")
    for kind in ("brightness", "defocus_blur", "gaussian_noise"):
        save_image(tensor_image(corrupt_batch(ham_image, kind)),
                   f"fig_panel_corrupt_ham10000_{kind}")

    # ---- case study panels ----
    case_image = case_example["img"]
    case_poisoned = concept_tools.paste_patch(case_image, patch, case_example["center"])
    save_image(tensor_image(case_image), "fig_panel_case_original")
    save_image(tensor_image(case_image), "fig_panel_case_heatmap",
               marker=case_example["center"], overlay=case_example["sim"])
    save_image(tensor_image(case_poisoned), "fig_panel_case_poison")

    pcbm = concept_tools.load_pcbm("cifar10", "cpu")
    ranked = concept_tools.rank_concepts(pcbm, 4, k=5)
    names = [name for name, _ in ranked][::-1]
    weights = [value for _, value in ranked][::-1]
    fig, ax = plt.subplots(figsize=(1.62, 1.48))
    y = np.arange(len(names))
    ax.barh(y, weights, color="#0072B2", height=0.62)
    for yi, weight in zip(y, weights):
        ax.text(weight + max(weights) * 0.025, yi, f"{weight:.0f}", va="center",
                ha="left", fontsize=5.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xticks([0, 50, 100])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlim(0, max(weights) * 1.20)
    ax.set_xlabel("Concept weight", fontsize=7)
    style_axes(ax, axis="x")
    fig.tight_layout(pad=0.25)
    save(fig, "fig_panel_case_rank")


def main():
    setup_style()
    PAPER_PICS.mkdir(parents=True, exist_ok=True)
    figures_celd()
    figures_traj()
    figures_coverage()
    figures_images()
    figures_sequential()


if __name__ == "__main__":
    main()
