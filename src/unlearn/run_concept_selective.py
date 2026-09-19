"""Concept-selective unlearning: forget only the samples carrying a concept.

The target class is split by a PCBM concept score into a *forget set* (concept
positive) and a *retain set* (concept negative). Only the forget-set training
images are relabeled to the donor class; the retain set keeps its label. The
unlearned model is evaluated on independently labeled concept-positive and
concept-negative held-out splits (CLIP zero-shot prompts), so the selection
signal (PCBM) and the evaluation signal (CLIP judge) are decoupled.

Arms:
  --select pcbm    top-K by PCBM concept score (K defaults to the F1-optimal
                   cut against the CLIP judge on the training split)
  --select random  random K target-class images
  --select all     class-level unlearning (all target images relabeled)

Example (apple/pink):
  python -m src.unlearn.run_concept_selective --dataset cifar100 --target-class 0 \
      --target-concept pink --select pcbm --seed 42 --device cuda
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.core import common, concept_tools, config

JUDGE_DEFAULTS = {
    ("cifar100", 0): ("a photo of a red apple", "a photo of a green apple"),
}


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "ham10000"])
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--target-concept", default=None, help="default: CASES target_concept")
    parser.add_argument("--donor-class", type=int, default=None, help="default: CASES donor_class")
    parser.add_argument("--judge-pos", default=None, help="CLIP prompt for concept-positive")
    parser.add_argument("--judge-neg", default=None, help="CLIP prompt for concept-negative")
    parser.add_argument("--select", default="pcbm", choices=["pcbm", "random", "all"])
    parser.add_argument("--k", type=int, default=None, help="forget-set size; default = F1-optimal (pcbm) / all (random)")
    parser.add_argument("--labels", default="targeted", choices=["targeted", "random"])
    parser.add_argument("--rebalance-retain", action="store_true",
                        help="oversample non-forgotten target-class images to the forget-set size")
    parser.add_argument("--unfreeze", default="none", choices=["none", "all", "layer4"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


class SelectiveDataset(Dataset):
    """Training images; forget-set indices get the donor label.

    `indices` may repeat base indices (used by --rebalance-retain to keep the
    retained subset of the target class from being starved by the relabeling).
    """

    def __init__(self, base, indices, forget_indices, labels_by_index):
        self.base = base
        self.indices = list(indices)
        self.labels_by_index = labels_by_index
        self.forget = set(forget_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        image, label = self.base[base_index]
        if base_index in self.forget:
            label = self.labels_by_index[base_index]
        return image, label


@torch.no_grad()
def clip_embeddings(clip_model, images, batch=256):
    device = next(clip_model.parameters()).device
    out = []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch].to(device)
        feat = clip_model.encode_image(concept_tools.clip_normalize(chunk)).float()
        out.append(feat / feat.norm(dim=-1, keepdim=True))
    return torch.cat(out)


def load_target_images(train_set, test_set, target_class):
    train_targets = [int(y) for y in train_set.targets]
    test_targets = [int(y) for y in test_set.targets]
    train_idx = [i for i, y in enumerate(train_targets) if y == target_class]
    test_idx = [i for i, y in enumerate(test_targets) if y == target_class]
    train_imgs = torch.stack([train_set[i][0] for i in train_idx])
    test_imgs = torch.stack([test_set[i][0] for i in test_idx])
    return train_idx, test_idx, train_imgs, test_imgs


def f1_optimal_k(scores, judge):
    order = np.argsort(scores)[::-1]
    judge_sorted = judge[order]
    positives = max(int(judge.sum()), 1)
    best = (0.0, len(order))
    curve = []
    correct = 0
    for k in range(1, len(order) + 1):
        correct += int(judge_sorted[k - 1])
        precision = correct / k
        recall = correct / positives
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        curve.append({"k": k, "precision": precision, "recall": recall, "f1": f1})
        if f1 > best[0]:
            best = (f1, k)
    return best[1], curve


def selection_stats(selected, judge):
    sel = judge[selected]
    positives = max(int(judge.sum()), 1)
    precision = float(sel.mean())
    recall = float(sel.sum() / positives)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"size": int(len(selected)), "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def subset_accuracy(model, dataset, indices, target_class, device, batch_size=64):
    model.eval()
    correct = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        images = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        pred = model(images).argmax(dim=1).cpu().numpy()
        correct += int((pred == target_class).sum())
    return correct / max(len(indices), 1)


def build_optimizer(model, unfreeze, head_lr, backbone_lr, momentum):
    if unfreeze == "none":
        params = common.freeze_backbone(model)
        return torch.optim.SGD(params, lr=head_lr, momentum=momentum)
    for param in model.parameters():
        param.requires_grad = unfreeze == "all"
    for name, param in model.named_parameters():
        if "fc" in name or (unfreeze == "layer4" and name.startswith("layer4")):
            param.requires_grad = True
    head = [p for n, p in model.named_parameters() if "fc" in n]
    backbone = [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad]
    return torch.optim.SGD(
        [{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}], momentum=momentum
    )


def main():
    args = build_argparser().parse_args()
    common.seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    case = config.CASES.get(args.dataset, {}).get(args.target_class, {})
    concept = args.target_concept or case.get("target_concept")
    donor_class = args.donor_class if args.donor_class is not None else case.get("donor_class")
    if concept is None:
        raise SystemExit("--target-concept is required (no case config for this class)")
    if donor_class is None:
        raise SystemExit("--donor-class is required (no case config for this class)")
    judge_pos = args.judge_pos
    judge_neg = args.judge_neg
    if judge_pos is None or judge_neg is None:
        default = JUDGE_DEFAULTS.get((args.dataset, args.target_class))
        if default is None:
            raise SystemExit("--judge-pos/--judge-neg are required for this dataset/class")
        judge_pos, judge_neg = default

    tag = f"{args.dataset}_c{args.target_class}_{concept.replace(' ', '_')}_select-{args.select}_{args.labels}_s{args.seed}"
    if args.unfreeze != "none":
        tag += f"_{args.unfreeze}"
    run_dir = Path(args.out) if args.out else config.RUNS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists() and not args.force:
        print(f"skip existing run: {run_dir}")
        return

    epochs = args.epochs or config.TRAIN_DEFAULTS["epochs"][args.dataset]
    lr = args.lr or config.TRAIN_DEFAULTS["lr"]
    batch_size = args.batch_size or config.TRAIN_DEFAULTS["batch_size"]

    transform = common.transform_224()
    train_set = common.load_cifar(args.dataset, train=True, transform=transform)
    test_set = common.load_cifar(args.dataset, train=False, transform=transform)
    train_idx, test_idx, train_imgs, test_imgs = load_target_images(train_set, test_set, args.target_class)

    clip_model = concept_tools.load_clip(device)
    bank = concept_tools.load_concept_bank(args.dataset)
    concept_vec = concept_tools.concept_vector(bank, concept).view(1, -1)
    pos_vec = concept_tools.clip_text_vector(clip_model, judge_pos)
    neg_vec = concept_tools.clip_text_vector(clip_model, judge_neg)

    with torch.no_grad():
        train_emb = clip_embeddings(clip_model, train_imgs)
        test_emb = clip_embeddings(clip_model, test_imgs)
    scores = (train_emb @ concept_vec.to(train_emb.device).T).squeeze(1).cpu().numpy()
    judge_train = ((train_emb @ pos_vec.T) - (train_emb @ neg_vec.T)).squeeze(1).cpu().numpy() > 0
    judge_test = ((test_emb @ pos_vec.T) - (test_emb @ neg_vec.T)).squeeze(1).cpu().numpy() > 0
    del train_emb, test_emb
    torch.cuda.empty_cache()

    f1_curve = None
    if args.select == "pcbm":
        if args.k is None:
            k, f1_curve = f1_optimal_k(scores, judge_train)
        else:
            k = args.k
        order = np.argsort(scores)[::-1]
        selected_local = sorted(int(i) for i in order[:k])
    elif args.select == "random":
        k = args.k if args.k is not None else len(train_idx)
        rng = random.Random(args.seed)
        selected_local = sorted(rng.sample(range(len(train_idx)), min(k, len(train_idx))))
    else:
        selected_local = list(range(len(train_idx)))
    forget_indices = [train_idx[i] for i in selected_local]
    stats = selection_stats(selected_local, judge_train)

    label_map = {}
    if args.labels == "targeted":
        for i in forget_indices:
            label_map[i] = donor_class
    else:
        for i in forget_indices:
            label_map[i] = random.Random(args.seed + i).randrange(config.DATASETS[args.dataset]["num_classes"])

    selection_payload = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "concept": concept,
        "select": args.select,
        "k": len(forget_indices),
        "judge_pos": judge_pos,
        "judge_neg": judge_neg,
        "judge_positive_rate_train": float(judge_train.mean()),
        "judge_positive_rate_test": float(judge_test.mean()),
        "selection": stats,
        "forget_indices": forget_indices,
        "scores": [float(s) for s in scores],
        "judge_train": [bool(v) for v in judge_train],
    }
    if f1_curve is not None:
        selection_payload["f1_curve"] = f1_curve
    (run_dir / "selection.json").write_text(json.dumps(selection_payload, indent=2))

    pos_test_idx = [test_idx[i] for i in np.where(judge_test)[0]]
    neg_test_idx = [test_idx[i] for i in np.where(~judge_test)[0]]
    pos_train_idx = [train_idx[i] for i in np.where(judge_train)[0]]
    neg_train_idx = [train_idx[i] for i in np.where(~judge_train)[0]]
    print(json.dumps({
        "tag": tag, "concept": concept, "donor": donor_class, "select": args.select,
        "forget_size": len(forget_indices), "selection": stats,
        "pos_test": len(pos_test_idx), "neg_test": len(neg_test_idx),
        "pos_train": len(pos_train_idx), "neg_train": len(neg_train_idx),
        "unfreeze": args.unfreeze, "epochs": epochs,
    }))

    original = common.load_classifier(args.dataset, device)
    baseline = {
        "A_pos_test": subset_accuracy(original, test_set, pos_test_idx, args.target_class, device),
        "A_neg_test": subset_accuracy(original, test_set, neg_test_idx, args.target_class, device),
        "A_pos_train": subset_accuracy(original, train_set, pos_train_idx, args.target_class, device),
        "A_neg_train": subset_accuracy(original, train_set, neg_train_idx, args.target_class, device),
    }
    print("original baseline", json.dumps(baseline))
    del original
    torch.cuda.empty_cache()

    train_indices = list(range(len(train_set)))
    if args.rebalance_retain:
        forget_set = set(forget_indices)
        retain_indices = [i for i in train_idx if i not in forget_set]
        if retain_indices:
            factor = max(0, round((len(forget_indices) - len(retain_indices)) / len(retain_indices)))
            train_indices += retain_indices * factor
            print(json.dumps({"rebalance": {"retain_size": len(retain_indices), "extra_copies": factor}}))
    dataset = SelectiveDataset(train_set, train_indices, forget_indices, label_map)
    if args.dry_run:
        dataset = torch.utils.data.Subset(dataset, list(range(min(512, len(dataset)))))
        epochs = 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers)

    model = common.load_classifier(args.dataset, device)
    optimizer = build_optimizer(model, args.unfreeze, lr, args.backbone_lr, config.TRAIN_DEFAULTS["momentum"])
    criterion = nn.CrossEntropyLoss()

    history = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running = 0.0
        steps = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
            if args.dry_run and steps >= 10:
                break
        record = {
            "epoch": epoch,
            "loss": running / max(steps, 1),
            "A_pos_test": subset_accuracy(model, test_set, pos_test_idx, args.target_class, device),
            "A_neg_test": subset_accuracy(model, test_set, neg_test_idx, args.target_class, device),
            "A_pos_train": subset_accuracy(model, train_set, pos_train_idx, args.target_class, device),
            "A_neg_train": subset_accuracy(model, train_set, neg_train_idx, args.target_class, device),
            "A_tgt_test": subset_accuracy(model, test_set, test_idx, args.target_class, device),
            "epoch_seconds": time.time() - epoch_start,
        }
        test_acc, test_correct, test_total = common.evaluate_classes(
            model, test_loader, device, config.DATASETS[args.dataset]["num_classes"]
        )
        record["retained_acc"] = float(
            (test_correct.sum() - test_correct[args.target_class])
            / max(test_total.sum() - test_total[args.target_class], 1)
        )
        record["global_acc"] = float(test_correct.sum() / max(test_total.sum(), 1))
        history.append(record)
        with open(run_dir / "epochs.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    torch.save(model, run_dir / "model.pkl")
    last = history[-1]
    summary = {
        "tag": tag,
        "config": {
            "dataset": args.dataset,
            "target_class": args.target_class,
            "mode": f"select-{args.select}",
            "labels": args.labels,
            "concept": concept,
            "donor_class": donor_class,
            "seed": args.seed,
            "epochs": epochs,
            "lr": lr,
            "backbone_lr": args.backbone_lr if args.unfreeze != "none" else None,
            "unfreeze": args.unfreeze,
            "batch_size": batch_size,
            "rebalance_retain": args.rebalance_retain,
            "judge_pos": judge_pos,
            "judge_neg": judge_neg,
        },
        "baseline": baseline,
        "selection": stats,
        "forget_size": len(forget_indices),
        "final": last,
        "selectivity": {
            "forget_drop": baseline["A_pos_test"] - last["A_pos_test"],
            "retain_drop": baseline["A_neg_test"] - last["A_neg_test"],
            "selectivity": (baseline["A_pos_test"] - last["A_pos_test"])
            - (baseline["A_neg_test"] - last["A_neg_test"]),
        },
        "train_seconds": time.time() - start_time,
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["selectivity"]))
    print(f"saved {run_dir/'summary.json'}")


if __name__ == "__main__":
    main()
