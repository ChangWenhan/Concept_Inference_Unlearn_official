"""Build the HAM10000 concept bank.

ConceptNet coverage of dermatology terms is sparse (the official API is also
unreachable), so the bank combines:
  1. a curated dermatology vocabulary (class names, ABCDE criteria, appearance
     descriptors), and
  2. concepts linked to these terms through ConceptNet 5.7 relations
     (HasA / MadeOf / HasProperty / IsA / PartOf), queried locally from the
     official assertions dump.

All concepts are encoded with CLIP RN50 text embeddings, matching the format
used for the CIFAR banks.
"""

import argparse
import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path

import clip
import numpy as np
import torch
from nltk.stem.wordnet import WordNetLemmatizer

from src.core import config
from src.data.ham10000 import HAM_CLASSES, HAM_CLASS_NAMES

CN_DUMP = Path("/mnt/disk/cwh/data/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
REL_START = {"HasA", "MadeOf", "HasProperty", "IsA"}
REL_END = {"PartOf"}

CURATED = [
    "melanoma", "nevus", "mole", "melanocytic nevus", "keratosis", "basal cell carcinoma",
    "actinic keratosis", "hemangioma", "vascular lesion", "dermatofibroma",
    "asymmetry", "border", "irregular border", "color", "diameter", "evolving",
    "pigment", "pigmentation", "pigment network", "blue white veil", "regression",
    "vascular pattern", "dots", "globules", "streaks",
    "brown", "black", "dark", "red", "pink", "white", "blue", "gray", "tan", "skin colored",
    "scaly", "rough", "smooth", "shiny", "raised", "flat", "oval", "round", "irregular",
    "spot", "freckle", "blemish", "lump", "bump", "nodule", "wart", "blister", "scar",
    "sore", "ulcer", "rash", "growth", "lesion", "skin", "hair", "blood", "vessel",
    "disease", "cancer", "tumor", "malignant", "benign", "deadly", "infectious", "symptom",
]


def uri(term):
    return "/c/en/" + term.replace(" ", "_")


def uri_to_term(value):
    if not value.startswith("/c/en/"):
        return None
    rest = value[len("/c/en/"):]
    term = rest.split("/")[0].replace("_", " ").strip()
    return term or None


def scan(dump=CN_DUMP):
    seeds = {uri(t): t for t in CURATED}
    found = defaultdict(list)
    with gzip.open(dump, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rel, start, end = parts[1], parts[2], parts[3]
            if rel in ("/r/HasA", "/r/MadeOf", "/r/HasProperty", "/r/IsA") and start in seeds:
                other, seed_term = end, seeds[start]
            elif rel == "/r/PartOf" and end in seeds:
                other, seed_term = start, seeds[end]
            else:
                continue
            term = uri_to_term(other)
            if term is not None:
                found[term].append((rel[len("/r/"):], seed_term))
    return found


def clean(term):
    term = term.strip().lower()
    if term.startswith("a "):
        term = term[2:]
    words = term.split()
    if not words or len(words) > 2:
        return None
    lem = WordNetLemmatizer()
    return " ".join(lem.lemmatize(w) for w in words)


def build_concepts():
    found = scan()
    provenance = defaultdict(set)
    concepts = set(CURATED)
    for term, prov in found.items():
        cleaned = clean(term)
        if cleaned is None:
            continue
        concepts.add(cleaned)
        for rel, seed_term in prov:
            provenance[cleaned].add(f"{rel}:{seed_term}")
    return sorted(concepts), {k: sorted(v) for k, v in provenance.items()}


def encode_bank(concepts, out_path, device="cuda"):
    model, _ = clip.load("RN50", device=device)
    model = model.eval()
    bank = {}
    with torch.no_grad():
        for concept in concepts:
            text = clip.tokenize([concept]).to(device)
            feats = model.encode_text(text).float().cpu().numpy()
            feats = feats / np.linalg.norm(feats)
            bank[concept] = (feats, None, None, 0, {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bank, f)
    return bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recurse", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary", default=str(config.SRC_DIR / "assets" / "ham_concepts.json"))
    args = parser.parse_args()

    concepts, provenance = build_concepts()
    summary = {
        "dump": str(CN_DUMP),
        "n_concepts": len(concepts),
        "curated": CURATED,
        "from_conceptnet": {k: v for k, v in provenance.items()},
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    out_path = config.OUTPUT_DIR / "ham10000_output" / f"multimodal_concept_clip:RN50_ham10000_recurse:{args.recurse}.pkl"
    encode_bank(concepts, out_path, device=args.device)
    print(json.dumps({
        "concepts": len(concepts),
        "from_conceptnet": len(provenance),
        "bank": str(out_path),
        "summary": args.summary,
    }, indent=2))


if __name__ == "__main__":
    main()
