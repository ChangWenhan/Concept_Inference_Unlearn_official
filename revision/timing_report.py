"""E7: timing report for the image-side pipeline.

Measures (per class) the one-off concept-inference cost (CLIP localization
for the full target class) and collects training time from run summaries.
Concept inference is cacheable, so per additional unlearning request it is
amortized to zero; the report makes this explicit.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from . import common, concept_tools, config
from .poison_gen import make_localized


def measure_concept_inference(dataset_key, target_class, count, device):
    case = config.CASES[dataset_key][target_class]
    raw = common.load_cifar(dataset_key, train=True, transform=common.transform_224())
    targets = list(raw.targets)
    target_indices = [i for i, y in enumerate(targets) if int(y) == target_class][: count]
    donor_indices = [i for i, y in enumerate(targets) if int(y) == case["donor_class"]]

    t0 = time.time()
    clip_model = concept_tools.load_clip(device=device)
    bank = concept_tools.load_concept_bank(dataset_key)
    load_seconds = time.time() - t0

    t0 = time.time()
    for k, idx in enumerate(target_indices):
        donor_img = raw[donor_indices[k % len(donor_indices)]][0].unsqueeze(0)
        make_localized(clip_model, bank, raw[idx][0].unsqueeze(0), donor_img,
                       case["target_concept"], case["donor_concept"], 64)
    elapsed = time.time() - t0
    per_image = elapsed / max(len(target_indices), 1)
    return {
        "device": str(device),
        "sample_count": len(target_indices),
        "load_seconds": load_seconds,
        "seconds_per_image": per_image,
        "estimated_full_class_seconds": per_image * 5000,
        "estimated_full_class_with_load": per_image * 5000 + load_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample", type=int, default=64)
    args = parser.parse_args()

    runs = {}
    for run_dir in sorted(config.RUNS_DIR.iterdir()) if config.RUNS_DIR.exists() else []:
        if not (run_dir / "summary.json").exists():
            continue
        summary = json.load(open(run_dir / "summary.json"))
        cfg = summary["config"]
        key = (cfg["dataset"], cfg["target_class"], cfg["mode"], cfg["labels"], cfg["integrity"])
        runs.setdefault(key, []).append(summary["train_seconds"])
    timing_runs = [
        {
            "dataset": k[0], "target_class": k[1], "mode": k[2], "labels": k[3], "integrity": k[4],
            "runs": len(v), "median_seconds": sorted(v)[len(v) // 2],
        }
        for k, v in runs.items()
    ]

    inference = measure_concept_inference("cifar10", 4, args.sample, args.device)
    payload = {"concept_inference": inference, "runs": timing_runs}
    out_dir = config.WORK_DIR / "e7"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "timing.json", "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "| dataset | class | mode | labels | integrity | runs | median train s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in timing_runs:
        lines.append("| {dataset} | {target_class} | {mode} | {labels} | {integrity} | {runs} | {median_seconds:.1f} |".format(**r))
    lines.append("")
    lines.append(
        f"Concept inference (one-off, cacheable): {inference['seconds_per_image'] * 1000:.1f} ms/image, "
        f"~{inference['estimated_full_class_seconds']:.0f} s per class (5000 images), "
        f"+{inference['load_seconds']:.1f} s model/bank load."
    )
    text = "\n".join(lines)
    with open(out_dir / "timing.md", "w") as f:
        f.write(text + "\n")
    print(json.dumps(payload, indent=2))
    print(text)


if __name__ == "__main__":
    main()
