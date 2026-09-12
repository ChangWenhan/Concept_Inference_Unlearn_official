"""E5: side effects on the poisoned data itself.

Checks whether the unlearned model memorizes the poison images:
mean confidence/loss on poison images vs clean donor-class data, and a
membership attack distinguishing poison images (members) from donor-class
test images (non-members).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import common, config, mia


class ImageList(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), 0


@torch.no_grad()
def collect_probs(model, dataset, device, batch_size=64, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs = []
    for images, _ in loader:
        probs.append(F.softmax(model(images.to(device)), dim=1).cpu())
    return torch.cat(probs).numpy()


def summarize(probs, labels=None):
    conf = probs.max(axis=1)
    logp = np.log(np.clip(probs, 1e-12, None))
    if labels is None:
        loss = -logp.mean(axis=1)
    else:
        loss = -logp[np.arange(len(labels)), labels]
    return {"mean_confidence": float(conf.mean()), "mean_loss": float(loss.mean()), "count": len(probs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="unlearned run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run)
    summary = json.load(open(run_dir / "summary.json"))
    cfg = summary["config"]
    if cfg["mode"] == "none":
        print("no poison images for mode=none")
        return
    dataset_key, target_class = cfg["dataset"], cfg["target_class"]
    donor_class = cfg.get("donor_class", 0)
    device = torch.device(args.device)
    transform = common.transform_224()

    poison_dir = config.POISON_DIR / dataset_key / f"class{target_class}_{cfg['mode']}"
    poison_paths = sorted(poison_dir.glob("img_*.jpg"))
    if not poison_paths:
        raise SystemExit(f"no poison images in {poison_dir}")

    train_set = common.load_cifar(dataset_key, train=True, transform=transform)
    test_set = common.load_cifar(dataset_key, train=False, transform=transform)
    targets = list(train_set.targets)
    target_train = torch.utils.data.Subset(train_set, [i for i, y in enumerate(targets) if int(y) == target_class])
    donor_train = torch.utils.data.Subset(train_set, [i for i, y in enumerate(targets) if int(y) == donor_class])
    donor_test_idx = [i for i, y in enumerate(test_set.targets) if int(y) == donor_class]

    original = common.load_classifier(dataset_key, device)
    unlearned = torch.load(run_dir / "model.pkl", map_location="cpu", weights_only=False).to(device).eval()

    poison_set = ImageList(poison_paths, transform)
    orig_poison_probs = collect_probs(original, poison_set, device, num_workers=args.num_workers)
    orig_donor_test_probs = collect_probs(
        original, torch.utils.data.Subset(test_set, donor_test_idx), device, num_workers=args.num_workers
    )
    attack = mia.fit_fr_attack(orig_poison_probs, orig_donor_test_probs, seed=cfg["seed"])
    result = {"run": str(run_dir), "poison_dir": str(poison_dir), "count": len(poison_paths)}
    for name, model in [("original", original), ("unlearned", unlearned)]:
        poison_probs = orig_poison_probs if name == "original" else collect_probs(
            model, poison_set, device, num_workers=args.num_workers
        )
        target_probs = collect_probs(model, target_train, device, num_workers=args.num_workers)
        donor_train_probs = collect_probs(model, donor_train, device, num_workers=args.num_workers)
        donor_test_probs = orig_donor_test_probs if name == "original" else collect_probs(
            model, torch.utils.data.Subset(test_set, donor_test_idx), device, num_workers=args.num_workers
        )
        result[name] = {
            "poison": summarize(poison_probs),
            "target_train": summarize(target_probs),
            "donor_train": summarize(donor_train_probs),
            "donor_test": summarize(donor_test_probs),
            "poison_vs_donor_test_fr": mia.apply_fr(attack, poison_probs),
        }

    out = run_dir / "side_effects.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
