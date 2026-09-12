import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from . import config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transform_224():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def load_cifar(dataset_key, train, transform, data_root=None):
    root = str(data_root or config.DATA_ROOT)
    if dataset_key == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
    if dataset_key == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=False, transform=transform)
    raise ValueError(dataset_key)


def load_classifier(dataset_key, device):
    model = torch.load(config.DATASETS[dataset_key]["model"], map_location="cpu", weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name
    return [p for p in model.parameters() if p.requires_grad]


def per_class_indices(targets):
    indices = {}
    for i, y in enumerate(targets):
        indices.setdefault(int(y), []).append(i)
    return indices


def select_indices(targets, integrity, seed):
    all_indices = list(range(len(targets)))
    if integrity == "full":
        return all_indices
    if integrity != "half":
        raise ValueError(integrity)
    rng = random.Random(seed)
    selected = []
    for cls, idx in per_class_indices(targets).items():
        idx = list(idx)
        rng.shuffle(idx)
        selected.extend(idx[: len(idx) // 2])
    return sorted(selected)


@torch.no_grad()
def evaluate_classes(model, loader, device, num_classes):
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        labels = labels.numpy()
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                correct[c] += int((preds[mask] == labels[mask]).sum())
                total[c] += int(mask.sum())
    acc = np.divide(correct, total, out=np.zeros(num_classes, dtype=float), where=total > 0)
    return acc, correct, total


class RetrainDataset(Dataset):
    def __init__(self, base, indices, target_class, poison_paths=None, poison_labels=None):
        self.base = base
        self.indices = list(indices)
        base_targets = getattr(base, "targets", None)
        self.poison = {}
        if poison_paths is not None:
            target_indices = [i for i in self.indices if int(base_targets[i]) == target_class]
            if len(target_indices) != len(poison_paths):
                raise ValueError(f"poison count {len(poison_paths)} != target samples {len(target_indices)}")
            for i, path in zip(target_indices, poison_paths):
                self.poison[i] = path
        if poison_labels is not None:
            target_indices = [i for i in self.indices if int(base_targets[i]) == target_class]
            if len(target_indices) != len(poison_labels):
                raise ValueError("poison label count mismatch")
            self.poison_labels = dict(zip(target_indices, poison_labels))
        else:
            self.poison_labels = {}
        self.transform = transform_224()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        image, label = self.base[i]
        if i in self.poison:
            image = self.transform(Image.open(self.poison[i]).convert("RGB"))
        if i in self.poison_labels:
            label = self.poison_labels[i]
        return image, label


def replacement_labels(dataset_key, donor_class, strategy, num_targets, seed):
    if strategy == "targeted":
        return [donor_class] * num_targets
    if strategy == "random":
        rng = random.Random(seed)
        return [rng.randrange(config.DATASETS[dataset_key]["num_classes"]) for _ in range(num_targets)]
    if strategy == "keep":
        return None
    raise ValueError(strategy)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
