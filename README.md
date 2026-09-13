# Class Machine Unlearning for Complex Data via Concepts Inference and Data Poisoning

Research code for the paper **"Class Machine Unlearning for Complex Data via Concepts Inference and Data Poisoning"** ([arXiv:2405.15662](https://arxiv.org/abs/2405.15662)).

This repository contains two experiment pipelines:

- **Image classification** (`src/`): concept inference with a Post-hoc Concept Bottleneck Model (PCBM), concept-localized data poisoning, and class-level unlearning on CIFAR-10 / CIFAR-100 / HAM10000.
- **LLM unlearning** (`llm-unlearning/`): entity unlearning on TOFU for Llama-2-7B / Vicuna-7B / Qwen2.5-7B, using Integrated Gradients (IG) to locate sensitive tokens, than masking them and fine-tuning with LoRA.

The concept-bank / PCBM components inherited from the original PCBM code base (`concepts/`, `models/`, `training_tools/`, `learn_concepts_*.py`, `train_pcbm*.py`) are kept at the repository root.

## Repository layout

```text
src/                            Image unlearning pipeline (E1-E9), classified by function
                                core/ data/ concepts/ poison/ unlearn/ evaluation/
                                corruption/ analysis/ training/ timing/ tools/ assets/
llm-unlearning/                 LLM unlearning subproject
  src/                          LLM pipeline code, classified by function
                                core/ data/ ig/ poison/ training/ evaluation/
                                analysis/ tools/ configs/ assets/
  README.md, src/README.md
concepts/ models/ training_tools/ data/   Concept-bank / PCBM components and loaders
learn_concepts_multimodal.py    Build a CLIP + ConceptNet concept bank
learn_concepts_dataset.py       Learn CAVs from positive/negative concept data
train_pcbm.py                   Train and inspect the Post-hoc CBM
train_pcbm_h.py                 Train the hybrid PCBM residual classifier
```

All pipeline steps are Python module entry points; run everything from the repository root, e.g.

```bash
python -m src.poison.poison_gen --help
python -m src.unlearn.run_unlearn --help
```

## What is NOT stored in this repository

Datasets, trained checkpoints, experiment records and server credentials are intentionally excluded
(see `.gitignore`). Their locations:

| Artifact | Where it lives |
| --- | --- |
| CIFAR-10 / CIFAR-100 | downloaded automatically by `torchvision` into `data/` |
| HAM10000 | external directory `/mnt/disk/cwh/data/ham10000` (see `src/data/ham10000.py`) |
| Concept banks / PCBM / PCBM-H checkpoints | generated under `output/` (`learn_concepts_multimodal.py`, `train_pcbm.py`, `train_pcbm_h.py`) |
| End-to-end image classifiers | generated under `models/end2end_models/` |
| Poisoned images, per-run evaluation and MIA results | generated under `work/`; historical runs are grouped by experiment in the A100 archive (see below) |
| LLM datasets (TOFU, MMLU) | A100 server project root `data/` |
| LLM target models and all run checkpoints | A100 server project root `models/` and `archive/` |
| Server credentials | `llm-unlearning/SERVER-ACCESS.local.md` (never committed) |

Internal revision documents, reviewer-response material and the full classified experiment archive are **not** part of this public repository; they are maintained on the local workstation and the A100 archive server.

## Environment

- Python 3.10, CUDA-enabled PyTorch.
- Image pipeline environment (workstation): `torch 2.1.2`, `numpy 1.26`, `scipy 1.11`, `scikit-learn 1.3`,
  `openai-clip`, `pytorchcv`, `nltk`. Run with `PYTHONNOUSERSITE=1` to avoid user-site package shadowing.
- LLM pipeline environment (servers): `torch 2.6/2.9`, `transformers 4.57`, `peft 0.18`, `trl 0.9`,
  `safetensors >= 0.4.3`, `datasets 3.6`; training uses LLaMA-Factory with the configs in
  `llm-unlearning/src/configs/lf_configs/`.

OpenAI CLIP is installed from source:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Image unlearning workflow

```text
train / load end-to-end model
        -> infer and rank concepts with PCBM
        -> select a confusing source concept / donor class
        -> generate poisoned target-class images
        -> fine-tune the end-to-end model on retained + poisoned data
        -> evaluate forgetting, retained utility, MIA and runtime
```

```bash
export PYTHONNOUSERSITE=1

# concept bank + PCBM (per dataset)
python learn_concepts_multimodal.py --classes cifar10 --backbone-name clip:RN50 \
  --out-dir output/cifar10_output --recurse 1
python train_pcbm.py --dataset cifar10

# poisoned data + unlearning run
python -m src.poison.poison_gen --dataset cifar10 --target-class 4 --mode localized --device cuda
python -m src.unlearn.run_unlearn --dataset cifar10 --target-class 4 --mode localized \
  --labels targeted --integrity full --device cuda

# evaluation + membership inference
python -m src.evaluation.evaluate --run work/runs/cifar10_c4_localized_targeted_full_s42 --device cuda
python -m src.evaluation.final_mia --device cuda
```

Poison generation supports fixed-region (`center`), concept-localized (`localized`), random-region
(`random`) and full-image replacement (`full`) modes. `localized` places the donor patch on the region
selected by concept localization: CLIP patch similarity by default, with GradCAM and PCBM margin maps
available as alternatives. See `src/README.md` for the full command matrix.

## LLM unlearning workflow

```text
target SFT -> model answers -> IG token attribution -> word lists -> poisoned data
           -> LoRA unlearning (LLaMA-Factory) -> checkpoint sweep + Pareto selection -> final evaluation
```

```bash
# examples (run from the LLM project root on the server)
python -m src.data.prepare_tofu --model tofu-ft-llama2-7b --split forget01
python -m src.ig.run_ig --model tofu-ft-llama2-7b --split forget01
python -m src.poison.build_poison --split forget01
python -m src.evaluation.final_eval --help
```

See `llm-unlearning/README.md` and `llm-unlearning/src/README.md` for details.

## Evaluation

Image unlearning is measured with target-class accuracy on train/test splits, global / retained-class
accuracy, membership-inference forgetting rate (paper-style SVM-transfer `Fr`, plus a loss-based MIA
with a retrain reference), fine-tuning time and cross-entropy distributions before/after unlearning.
Multiple seeds and the random-label / no-mask / random-mask / fixed-region / full-mask /
concept-guided ablations are used for a reliable comparison.

LLM unlearning is measured with appearance rate of the forgotten entities (original and paraphrased
questions), TOFU retain / real-authors / world-facts ROUGE-L, holdout name leakage and MMLU.

> Runner shell scripts are intentionally not maintained in this repository. Every step is a Python
> module entry point (`python -m src.<package>.<module> --help`); write your own job scripts for your
> cluster.

## Citation

```bibtex
@article{chang2024class,
  title={Class Machine Unlearning for Complex Data via Concepts Inference and Data Poisoning},
  author={Chang, Wenhan and Zhu, Tianqing and Xu, Heng and Liu, Wenjian and Zhou, Wanlei},
  journal={arXiv preprint arXiv:2405.15662},
  year={2024}
}
```

## Acknowledgements

The concept inference components build on [Post-hoc Concept Bottleneck Models](https://arxiv.org/abs/2205.15480)
by Mert Yuksekgonul, Maggie Wang, and James Zou. The original MIT license notice is retained in `LICENSE`.

## License

This repository is distributed under the MIT License. See `LICENSE` for details.
