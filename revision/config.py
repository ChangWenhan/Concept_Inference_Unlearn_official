import json
from pathlib import Path

REVISION_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVISION_DIR.parent
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
WORK_DIR = REVISION_DIR / "work"
POISON_DIR = WORK_DIR / "poison"
RUNS_DIR = WORK_DIR / "runs"
EVAL_DIR = WORK_DIR / "eval"

DATASETS = {
    "cifar10": {
        "model": REPO_ROOT / "models" / "end2end_models" / "resnet50_model_224.pkl",
        "retrain": REPO_ROOT / "models" / "end2end_models" / "resnet50_model_224_retrain.pkl",
        "concept_bank": OUTPUT_DIR / "cifar10_output" / "multimodal_concept_clip:RN50_cifar10_recurse:1.pkl",
        "pcbm": OUTPUT_DIR / "cifar10_output" / "pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:1e-05__alpha:0.99__seed:42_without.ckpt",
        "num_classes": 10,
    },
    "cifar100": {
        "model": REPO_ROOT / "models" / "end2end_models" / "resnet50_model_224_cifar100.pkl",
        "retrain": REPO_ROOT / "models" / "end2end_models" / "resnet50_model_224_cifar100_retrain.pkl",
        "concept_bank": OUTPUT_DIR / "cifar100_output" / "multimodal_concept_clip:RN50_cifar100_recurse:1.pkl",
        "pcbm": OUTPUT_DIR / "cifar100_output" / "pcbm_cifar100__clip:RN50__multimodal_concept_clip:RN50_cifar100_recurse:1__lam:1e-05__alpha:0.99__seed:42_without.ckpt",
        "num_classes": 100,
    },
}

CASES = {
    "cifar10": {
        0: {
            "name": "airplane",
            "target_concept": "propellers",
            "donor_concept": "propellers",
            "donor_class": 4,
            "donor_name": "deer",
        },
        4: {
            "name": "deer",
            "target_concept": "antler",
            "donor_concept": "propellers",
            "donor_class": 0,
            "donor_name": "airplane",
        },
        6: {
            "name": "frog",
            "target_concept": "amphibian",
            "donor_concept": "amphibian",
            "donor_class": 1,
            "donor_name": "automobile",
        },
        7: {
            "name": "horse",
            "target_concept": "horseback",
            "donor_concept": "horseback",
            "donor_class": 0,
            "donor_name": "airplane",
        },
        9: {
            "name": "truck",
            "target_concept": "gear",
            "donor_concept": "gear",
            "donor_class": 0,
            "donor_name": "airplane",
        },
    },
    "cifar100": {
        0: {
            "name": "apple",
            "target_concept": "pink",
            "donor_concept": "pink",
            "donor_class": 70,
            "donor_name": "rose",
        },
        11: {
            "name": "boy",
            "target_concept": "newborn human",
            "donor_concept": "newborn human",
            "donor_class": 2,
            "donor_name": "baby",
        },
        17: {
            "name": "castle",
            "target_concept": "chimney",
            "donor_concept": "chimney",
            "donor_class": 59,
            "donor_name": "pine_tree",
        },
        28: {
            "name": "cup",
            "target_concept": "lampshade",
            "donor_concept": "lampshade",
            "donor_class": 40,
            "donor_name": "lamp",
        },
        30: {
            "name": "dolphin",
            "target_concept": "cetacean",
            "donor_concept": "cetacean",
            "donor_class": 95,
            "donor_name": "whale",
        },
    },
}

TRAIN_DEFAULTS = {
    "lr": 1e-3,
    "momentum": 0.9,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": {"cifar10": 20, "cifar100": 30},
}

POISON_SIZES = {
    "localized": 64,
    "localized_gradcam": 64,
    "localized_margin": 64,
    "center": 96,
    "random": 96,
    "full": 160,
}


def dataset_classes(dataset_key):
    if dataset_key == "cifar10":
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    if dataset_key == "cifar100":
        return [
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
            "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
            "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur",
            "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
            "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
            "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
            "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
            "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
            "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
            "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
        ]
    raise ValueError(dataset_key)


def _merge_generated_cases():
    for path in (REVISION_DIR / "cases_cifar10.json", REVISION_DIR / "cases_cifar100.json"):
        if not path.exists():
            continue
        dataset_key = path.stem.replace("cases_", "")
        generated = json.loads(path.read_text())
        CASES.setdefault(dataset_key, {})
        for cls, info in generated.items():
            CASES[dataset_key].setdefault(int(cls), info)


_merge_generated_cases()
