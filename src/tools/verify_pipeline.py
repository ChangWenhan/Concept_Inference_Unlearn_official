#!/usr/bin/env python3
"""Load-only verification of the image pipeline: no training, no experiments."""
import importlib
import pickle
import py_compile
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS = []


def check(name, fn):
    try:
        info = fn()
        RESULTS.append((name, "PASS", info))
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        traceback.print_exc()


def compile_all():
    files = sorted(ROOT.glob("src/**/*.py")) + [ROOT / f for f in
        ("learn_concepts_multimodal.py", "learn_concepts_dataset.py", "train_pcbm.py", "train_pcbm_h.py")]
    for f in files:
        py_compile.compile(str(f), doraise=True)
    return f"{len(files)} files compiled"


SRC_MODULES = [
    "src.core.config", "src.core.common", "src.core.concept_tools", "src.core.mia",
    "src.data.ham10000", "src.data.make_cases", "src.data.gen_all_cases",
    "src.concepts.build_ham_concepts",
    "src.poison.poison_gen", "src.poison.poison_gen_multi",
    "src.unlearn.run_unlearn", "src.unlearn.batch", "src.unlearn.recover", "src.unlearn.side_effects",
    "src.evaluation.evaluate", "src.evaluation.final_mia", "src.evaluation.class_mia_profile",
    "src.evaluation.collect_results",
    "src.corruption.noisy_exp", "src.corruption.ham_noisy",
    "src.analysis.analyze_localization", "src.analysis.concept_forget_probe",
    "src.training.train_ham", "src.training.train_ham_pcbm",
    "src.timing.timing_localize", "src.timing.timing_pcbm", "src.timing.timing_report",
    "src.timing.timing_unlearn",
]


def import_src():
    for m in SRC_MODULES:
        importlib.import_module(m)
    return f"{len(SRC_MODULES)} modules imported"


def import_root_pkgs():
    import concepts
    import data
    import models
    import training_tools
    from concepts import ConceptBank  # noqa: F401
    from models import PosthocHybridCBM, PosthocLinearCBM, get_model  # noqa: F401
    from training_tools.embedding_tools import get_projections  # noqa: F401
    return "concepts / models / training_tools / data imported"


def import_scripts():
    for f in ("learn_concepts_multimodal", "learn_concepts_dataset", "train_pcbm", "train_pcbm_h"):
        importlib.import_module(f)
    return "4 PCBM scripts imported"


def artifact_paths():
    from src.core import config
    missing = []
    for key in ("cifar10", "cifar100", "ham10000"):
        for field in ("model", "retrain", "concept_bank", "pcbm"):
            p = Path(config.DATASETS[key][field])
            if not p.exists():
                missing.append(str(p))
    for name in ("cases_cifar10.json", "cases_cifar100.json", "cases_ham10000.json",
                 "ham_concepts.json", "ham_split_seed42.json"):
        p = ROOT / "src" / "assets" / name
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(missing)
    return "12 dataset artifacts + 5 assets exist"


def concept_banks():
    from src.core import concept_tools
    out = []
    for key in ("cifar10", "cifar100", "ham10000"):
        bank = concept_tools.load_concept_bank(key)
        out.append(f"{key}: {len(bank['names'])} concepts x {bank['vectors'].shape[1]}")
    return "; ".join(out)


def pcbms():
    from src.core import concept_tools
    out = []
    for key in ("cifar10", "cifar100", "ham10000"):
        m = concept_tools.load_pcbm(key, "cpu")
        with torch.no_grad():
            y_projs = m.forward_projs(torch.randn(2, m.n_concepts))
            try:
                y_emb = m(torch.randn(2, 1024))
                emb_mode = f"emb_f32 {tuple(y_emb.shape)}"
            except RuntimeError:
                y_emb = m(torch.randn(2, 1024, dtype=torch.float64))
                emb_mode = f"emb_f64 {tuple(y_emb.shape)}"
        out.append(f"{key}: cavs={m.cavs.dtype} cls={tuple(m.classifier.weight.shape)} "
                   f"projs_f32={tuple(y_projs.shape)} {emb_mode}")
    return "; ".join(out)


def hybrid_ckpts():
    from src.core import config
    out = []
    for key in ("cifar10", "cifar100", "ham10000"):
        ckpt = Path(str(config.DATASETS[key]["pcbm"]).replace("pcbm_", "pcbm-hybrid_", 1))
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        out.append(f"{key}: {type(obj).__name__}")
    return "; ".join(out)


def classifiers():
    from src.core import common
    out = []
    for key in ("cifar10", "cifar100", "ham10000"):
        m = common.load_classifier(key, "cpu")
        with torch.no_grad():
            y = m(torch.randn(2, 3, 224, 224))
        out.append(f"{key}: out={tuple(y.shape)}")
    return "; ".join(out)


def clip_path():
    from src.core import concept_tools
    import clip
    model = concept_tools.load_clip(device="cpu")
    text = clip.tokenize(["a deer with antlers"]).cpu()
    with torch.no_grad():
        t = model.encode_text(text)
        i = model.encode_image(torch.randn(1, 3, 224, 224))
    bank = concept_tools.load_concept_bank("cifar10")
    sim, grid_h, grid_w = concept_tools.patch_similarity(model, torch.rand(1, 3, 224, 224), bank["vectors"][:4])
    center, _, peak = concept_tools.locate_peak(sim, stride=16, patch=64)
    return (f"text={tuple(t.shape)} image={tuple(i.shape)} sim={tuple(sim.shape)} peak={center}")


def concepts_rank_gradcam():
    from src.core import concept_tools, common
    pcbm = concept_tools.load_pcbm("cifar10", "cpu")
    top = concept_tools.rank_concepts(pcbm, 4, k=3)
    m = common.load_classifier("cifar10", "cpu")
    cam = concept_tools.gradcam(m, torch.rand(1, 3, 224, 224), 4)
    return f"rank={top if isinstance(top, list) else type(top).__name__} cam={tuple(cam.shape)}"


def datasets():
    from src.core import common
    from src.data.ham10000 import HAMDataset
    from src.core import config
    ds10 = common.load_cifar("cifar10", train=True, transform=common.transform_224())
    x10, y10 = ds10[0]
    ds100 = common.load_cifar("cifar100", train=True, transform=common.transform_224())
    x100, y100 = ds100[0]
    from src.data.ham10000 import HAM_DATA_DIR
    ham = HAMDataset("train", common.transform_224(), data_dir=HAM_DATA_DIR)
    xh, yh = ham[0]
    return (f"cifar10 item={tuple(x10.shape)}/{y10} cifar100={tuple(x100.shape)}/{y100} "
            f"ham n={len(ham)} item={tuple(xh.shape)}/{yh}")


def cases():
    from src.core import config
    return "; ".join(f"{k}: {len(v)} classes" for k, v in sorted(config.CASES.items()))


check("py_compile", compile_all)
check("import src modules", import_src)
check("import root PCBM packages", import_root_pkgs)
check("import PCBM scripts", import_scripts)
check("artifact paths", artifact_paths)
check("load concept banks", concept_banks)
check("load PCBM + forward", pcbms)
check("load PCBM-H ckpts", hybrid_ckpts)
check("load end-to-end classifiers + forward", classifiers)
check("CLIP RN50 load + text/image/similarity", clip_path)
check("rank_concepts + gradcam", concepts_rank_gradcam)
check("datasets (CIFAR-10/100 + HAM10000 item)", datasets)
check("curated cases", cases)

print("\n==== VERIFICATION ====")
fails = 0
for name, status, info in RESULTS:
    print(f"[{status}] {name}: {info}")
    fails += status == "FAIL"
print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} checks passed")
sys.exit(1 if fails else 0)
