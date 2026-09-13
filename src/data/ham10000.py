"""HAM10000 dataset for the experiment pipeline (7-class skin lesion task)."""

import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

HAM_DATA_DIR = Path("/mnt/disk/cwh/data/ham10000")
HAM_METADATA = HAM_DATA_DIR / "HAM10000_metadata.csv"
HAM_SPLIT = Path(__file__).resolve().parents[1] / "assets" / "ham_split_seed42.json"

HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM_CLASS_NAMES = {
    "akiec": "actinic keratoses",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevi",
    "vasc": "vascular lesions",
}


def image_index(data_dir=HAM_DATA_DIR):
    paths = {}
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        for path in (data_dir / part).glob("*.jpg"):
            paths[path.stem] = path
    return paths


def build_split(seed=42, test_size=0.2, data_dir=HAM_DATA_DIR, out_path=HAM_SPLIT):
    rows = list(csv.DictReader(open(data_dir / "HAM10000_metadata.csv")))
    dx = [r["dx"] for r in rows]
    image_ids = [r["image_id"] for r in rows]
    train_ids, test_ids = train_test_split(image_ids, test_size=test_size, random_state=seed, stratify=dx)
    payload = {"seed": seed, "test_size": test_size, "train": sorted(train_ids), "test": sorted(test_ids)}
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return payload


def load_split(path=HAM_SPLIT, seed=42):
    path = Path(path)
    if not path.exists():
        return build_split(seed=seed, out_path=path)
    with open(path) as f:
        return json.load(f)


class HAMDataset(Dataset):
    def __init__(self, split, transform, data_dir=HAM_DATA_DIR, split_path=HAM_SPLIT, seed=42):
        self.split = split
        self.transform = transform
        self.data_dir = Path(data_dir)
        meta = load_split(split_path, seed=seed)
        keep = set(meta[split])
        paths = image_index(self.data_dir)
        rows = list(csv.DictReader(open(self.data_dir / "HAM10000_metadata.csv")))
        self.samples = []
        for r in rows:
            image_id = r["image_id"]
            if image_id not in keep or image_id not in paths:
                continue
            self.samples.append((paths[image_id], HAM_CLASSES.index(r["dx"])))
        self.targets = [label for _, label in self.samples]
        self.classes = list(HAM_CLASSES)
        if not self.samples:
            raise RuntimeError(f"no images found for split {split}, check download under {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
