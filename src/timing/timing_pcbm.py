"""Timing: train a PCBM from scratch (CLIP image projections + linear probe).

Mirrors train_pcbm.py without the unlearn projections or artifact saving.
Reports projection time and probe fitting time separately.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.linear_model import SGDClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools.embedding_tools import get_projections

from src.core import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    with open(config.DATASETS[args.dataset]["concept_bank"], "rb") as f:
        all_concepts = pickle.load(f)
    print(f"{len(all_concepts)} concepts in the bank")
    concept_bank = ConceptBank(all_concepts, device)

    loader_args = argparse.Namespace(device=str(device), backbone_name="clip:RN50")
    backbone, clip_preprocess = get_model(loader_args, backbone_name="clip:RN50")
    backbone = backbone.to(device)
    backbone.eval()

    transform = clip_preprocess
    if args.dataset == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root=str(config.DATA_ROOT), train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=str(config.DATA_ROOT), train=False, download=False, transform=transform)
    else:
        train_set = torchvision.datasets.CIFAR100(root=str(config.DATA_ROOT), train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=str(config.DATA_ROOT), train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    idx_to_class = {i: c for i, c in enumerate(train_set.classes)}
    posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name="clip:RN50", idx_to_class=idx_to_class, n_classes=len(train_set.classes))
    posthoc_layer = posthoc_layer.to(device)

    proj_start = time.time()
    train_embs, train_projs, train_lbls = get_projections(loader_args, backbone, posthoc_layer, train_loader)
    test_embs, test_projs, test_lbls = get_projections(loader_args, backbone, posthoc_layer, test_loader)
    projection_seconds = time.time() - proj_start

    probe_start = time.time()
    classifier = SGDClassifier(
        random_state=args.seed, loss="log_loss", alpha=1e-5, l1_ratio=0.99,
        verbose=0, penalty="elasticnet", max_iter=10000,
    )
    classifier.fit(train_projs, train_lbls)
    probe_seconds = time.time() - probe_start

    train_acc = float(np.mean(classifier.predict(train_projs) == train_lbls))
    test_acc = float(np.mean(classifier.predict(test_projs) == test_lbls))

    summary = {
        "dataset": args.dataset,
        "concepts": len(all_concepts),
        "projection_seconds": projection_seconds,
        "probe_seconds": probe_seconds,
        "pcbm_train_seconds": projection_seconds + probe_seconds,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }
    out_dir = config.WORK_DIR / "timing_reach"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"pcbm_{args.dataset}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
