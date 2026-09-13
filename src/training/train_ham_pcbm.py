"""Train PCBM / PCBM-H for HAM10000 on CLIP RN50 concept margins."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import SGDClassifier
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concepts import ConceptBank
from models import PosthocHybridCBM, PosthocLinearCBM, get_model
from training_tools.embedding_tools import get_projections

from src.core import common, config
from src.data.ham10000 import HAMDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recurse", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--hybrid-epochs", type=int, default=20)
    parser.add_argument("--hybrid-lr", type=float, default=0.01)
    parser.add_argument("--hybrid-l2", type=float, default=0.001)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else config.OUTPUT_DIR / "ham10000_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    bank_path = out_dir / f"multimodal_concept_clip:RN50_ham10000_recurse:{args.recurse}.pkl"
    with open(bank_path, "rb") as f:
        all_concepts = pickle.load(f)
    print(f"{len(all_concepts)} concepts from {bank_path}")
    concept_bank = ConceptBank(all_concepts, device)

    loader_args = argparse.Namespace(device=str(device), backbone_name="clip:RN50")
    backbone, clip_preprocess = get_model(loader_args, backbone_name="clip:RN50")
    backbone = backbone.to(device).eval()

    train_set = HAMDataset("train", clip_preprocess)
    test_set = HAMDataset("test", clip_preprocess)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    idx_to_class = {i: c for i, c in enumerate(train_set.classes)}
    pcbm = PosthocLinearCBM(concept_bank, backbone_name="clip:RN50", idx_to_class=idx_to_class, n_classes=len(train_set.classes))
    pcbm = pcbm.to(device)

    print("computing CLIP projections ...")
    train_embs, train_projs, train_lbls = get_projections(loader_args, backbone, pcbm, train_loader)
    test_embs, test_projs, test_lbls = get_projections(loader_args, backbone, pcbm, test_loader)

    classifier = SGDClassifier(
        random_state=args.seed, loss="log_loss", alpha=args.lam, l1_ratio=args.alpha,
        verbose=0, penalty="elasticnet", max_iter=10000,
    )
    classifier.fit(train_projs, train_lbls)
    pcbm.set_weights(classifier.coef_, classifier.intercept_)

    train_acc = float((classifier.predict(train_projs) == train_lbls).mean())
    test_acc = float((classifier.predict(test_projs) == test_lbls).mean())
    stem = f"pcbm_ham10000__clip:RN50__multimodal_concept_clip:RN50_ham10000_recurse:{args.recurse}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}_without"
    model_path = out_dir / f"{stem}.ckpt"
    torch.save(pcbm, model_path)
    run_info_path = out_dir / f"run_info-{stem}.pkl"
    with open(run_info_path, "wb") as f:
        pickle.dump({"train_acc": train_acc * 100, "test_acc": test_acc * 100, "n_concepts": len(all_concepts)}, f)
    print(json.dumps({"concepts": len(all_concepts), "train_acc": round(train_acc * 100, 2), "test_acc": round(test_acc * 100, 2), "ckpt": str(model_path)}))

    if args.hybrid:
        pcbm_float = pcbm.float()
        hybrid = PosthocHybridCBM(pcbm_float).to(device)
        train_loader_h = DataLoader(
            TensorDataset(torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()),
            batch_size=64, shuffle=True,
        )
        test_loader_h = DataLoader(
            TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()),
            batch_size=64, shuffle=False,
        )
        optimizer = torch.optim.Adam(hybrid.residual_classifier.parameters(), lr=args.hybrid_lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, args.hybrid_epochs + 1):
            hybrid.train()
            for batch_x, batch_y in train_loader_h:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = hybrid(batch_x)
                loss = criterion(out, batch_y) + args.hybrid_l2 * (hybrid.residual_classifier.weight ** 2).mean()
                loss.backward()
                optimizer.step()
            hybrid.eval()
            with torch.no_grad():
                correct = total = 0
                for batch_x, batch_y in test_loader_h:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    pred = hybrid(batch_x).argmax(dim=1)
                    correct += int((pred == batch_y).sum())
                    total += int(batch_y.numel())
            print(json.dumps({"epoch": epoch, "test_acc": correct / total}))
        hybrid_stem = stem.replace("pcbm_", "pcbm-hybrid_")
        hybrid_path = out_dir / f"{hybrid_stem}.ckpt"
        torch.save(hybrid, hybrid_path)
        with open(out_dir / f"{hybrid_stem.replace('pcbm', 'run_info-pcbm')}.pkl", "wb") as f:
            pickle.dump({"test_acc": correct / total}, f)
        print(json.dumps({"hybrid_ckpt": str(hybrid_path), "test_acc": round(correct / total * 100, 2)}))


if __name__ == "__main__":
    main()
