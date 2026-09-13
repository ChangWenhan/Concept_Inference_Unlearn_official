"""Concept forgetting probe: does the unlearned model still rely on the target concept?

For each target-class image we take the concept peak region recorded by E1
(CLIP patch similarity) and compare the model's target-class probability on:

  - the full image,
  - the image with the concept region occluded (gray patch),
  - the image with a same-size control patch occluded (center and random).

Concept reliance = target-probability drop under concept occlusion minus the
drop under control occlusion. A fully unlearned model should show reliance
close to the retrain reference; a model that only suppresses class behaviour
while keeping the concept feature should keep a positive reliance.

Outputs work/concept_probe_<dataset>_c<class>.json and a markdown summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.core import common, config


def load_records(dataset, target_class, concept, locator="clip"):
    tag = f"{dataset}_class{target_class}_{concept.replace(' ', '_')}"
    if locator != "clip":
        tag += f"_{locator}"
    path = config.WORK_DIR / "e1" / f"{tag}_records.json"
    if not path.exists():
        raise SystemExit(f"missing E1 records: {path}")
    with open(path) as f:
        return json.load(f)


def load_model(path, device):
    model = torch.load(path, map_location="cpu", weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def occlude(x, center, size, fill=0.5):
    _, _, height, width = x.shape
    cx, cy = center
    half = size // 2
    left = min(max(cx - half, 0), width - size)
    top = min(max(cy - half, 0), height - size)
    out = x.clone()
    out[:, :, top:top + size, left:left + size] = fill
    return out


def sample_control(rng, peak, size, image_size=224, margin=None):
    half = size // 2
    margin = margin or size
    for _ in range(50):
        x = int(rng.integers(half, image_size - half))
        y = int(rng.integers(half, image_size - half))
        if abs(x - peak[0]) + abs(y - peak[1]) >= margin:
            return (x, y)
    return (half, half)


@torch.no_grad()
def predict(model, x):
    return torch.softmax(model(x), dim=1)[0].cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--target-class", type=int, default=4)
    parser.add_argument("--concept", default=None)
    parser.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    case = config.CASES[args.dataset][args.target_class]
    concept = args.concept or case["target_concept"]
    donor = case["donor_class"]
    records = load_records(args.dataset, args.target_class, concept)[: args.limit]
    raw = common.load_cifar(args.dataset, train=True, transform=common.transform_224())

    models = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        models[name] = load_model(path, args.device)
        print(f"loaded {name} <- {path}")

    rng = np.random.default_rng(args.seed)
    per_image = []
    for rec in records:
        idx = rec["index"]
        image, _ = raw[idx]
        x = image.unsqueeze(0).to(args.device)
        peak = tuple(rec["peak_center"])
        control = sample_control(rng, peak, args.patch)
        row = {"index": idx, "peak_center": list(peak), "control_center": list(control)}
        for name, model in models.items():
            p = predict(model, x)
            pc = predict(model, occlude(x, peak, args.patch))
            pr = predict(model, occlude(x, control, args.patch))
            row[name] = {
                "p_target": float(p[args.target_class]),
                "p_donor": float(p[donor]),
                "occ_concept": {
                    "p_target": float(pc[args.target_class]),
                    "p_donor": float(pc[donor]),
                },
                "occ_control": {
                    "p_target": float(pr[args.target_class]),
                    "p_donor": float(pr[donor]),
                },
            }
        per_image.append(row)

    summary = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "target_concept": concept,
        "donor_class": donor,
        "patch": args.patch,
        "count": len(per_image),
        "models": {},
    }
    for name in models:
        def col(key):
            return np.array([r[name][key] for r in per_image], dtype=float)

        def col_occ(where, key):
            return np.array([r[name][where][key] for r in per_image], dtype=float)

        drop_concept = col("p_target") - col_occ("occ_concept", "p_target")
        drop_control = col("p_target") - col_occ("occ_control", "p_target")
        summary["models"][name] = {
            "p_target_mean": float(col("p_target").mean()),
            "p_donor_mean": float(col("p_donor").mean()),
            "concept_drop_mean": float(drop_concept.mean()),
            "concept_drop_median": float(np.median(drop_concept)),
            "control_drop_mean": float(drop_control.mean()),
            "control_drop_median": float(np.median(drop_control)),
            "concept_reliance_mean": float((drop_concept - drop_control).mean()),
            "concept_reliance_median": float(np.median(drop_concept - drop_control)),
            "donor_rise_concept": float((col_occ("occ_concept", "p_donor") - col("p_donor")).mean()),
            "donor_rise_control": float((col_occ("occ_control", "p_donor") - col("p_donor")).mean()),
        }

    out = Path(args.out) if args.out else config.WORK_DIR / f"concept_probe_{args.dataset}_c{args.target_class}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=1)

    header = "| model | p(target) | concept drop | control drop | reliance | donor rise (concept) |"
    lines = [header, "|---|---|---|---|---|---|"]
    for name, m in summary["models"].items():
        lines.append(
            f"| {name} | {m['p_target_mean']:.3f} | {m['concept_drop_mean']:.3f} | "
            f"{m['control_drop_mean']:.3f} | {m['concept_reliance_mean']:+.3f} | {m['donor_rise_concept']:.3f} |"
        )
    md = out.with_suffix(".md")
    md.write_text("# concept forgetting probe\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {out} and {md}")


if __name__ == "__main__":
    main()
