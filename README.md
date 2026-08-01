# Class Machine Unlearning for Complex Data via Concepts Inference and Data Poisoning

Research code accompanying the paper **Class Machine Unlearning for Complex Data via Concepts Inference and Data Poisoning**.

> Wenhan Chang, Tianqing Zhu, Heng Xu, Wenjian Liu, and Wanlei Zhou.
> [arXiv:2405.15662](https://arxiv.org/abs/2405.15662)

This repository contains the image-classification part of the project. It studies class-level machine unlearning on CIFAR-10 and CIFAR-100 by combining concept inference with poisoning-based fine-tuning. The language-model experiments described in the paper are not included in this repository.

## Method overview

The method separates concept discovery from model unlearning:

1. A Post-hoc Concept Bottleneck Model (PCBM) maps image embeddings into a human-readable concept space.
2. Concept weights are used to identify concepts that strongly influence the target class, including concepts shared or confused with other classes.
3. Poisoned target-class samples are constructed by injecting or masking the selected visual concepts and assigning targeted or random labels.
4. The original end-to-end classifier is fine-tuned on retained data and poisoned target-class data.
5. Unlearning is evaluated using target-class accuracy, retained/global accuracy, forgetting rate, membership inference, and runtime.

PCBM is used as a concept inference and analysis tool. The model being unlearned is the end-to-end image classifier.

The image experiments currently focus on:

| Dataset | Target class | Confusing/source concept example |
| --- | --- | --- |
| CIFAR-10 | `deer` (class 4) | plane/propeller-related features |
| CIFAR-100 | `boy` (class 11) | baby/newborn-human-related features |

## Repository structure

```text
concepts/                       Concept banks and Concept Activation Vectors
data/                           Dataset loaders and local path configuration
models/                         PCBM modules and backbone definitions
training_tools/                 Embedding and concept-projection utilities
learn_concepts_multimodal.py    Build a CLIP/ConceptNet concept bank
learn_concepts_dataset.py       Learn CAVs from positive/negative concept data
train_pcbm.py                   Train and inspect the concept-based classifier
train_pcbm_h.py                 Train the hybrid PCBM residual classifier
generate_poisondata.py          Construct concept-injected poison images
test_script.py                  CIFAR-10 class-unlearning experiment
test_script_cifar100.py         CIFAR-100 class-unlearning experiment
evaluate_models.py              Target and retained-class evaluation
meminf.py                       Membership-inference evaluation
test.py                         Cross-entropy loss distribution analysis
```

## Environment

The code is a research prototype and currently assumes a CUDA-enabled PyTorch environment. The exact package versions used for the original experiments were not recorded in this checkout. The main dependencies are:

```text
torch
torchvision
numpy
scipy
scikit-learn
pandas
Pillow
tqdm
matplotlib
requests
nltk
openai-clip
pytorchcv
```

OpenAI CLIP can be installed from its source repository:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Data and local configuration

CIFAR-10 and CIFAR-100 can be downloaded automatically by `torchvision`. CUB, Derm7pt, HAM10000, and Broden require external downloads if the inherited PCBM utilities for those datasets are used.

Before running an experiment:

1. Update dataset paths in `data/constants.py` when using non-CIFAR datasets.
2. Replace the local absolute paths in the experiment scripts with paths for your environment.
3. Create local output and checkpoint directories as needed.

Downloaded datasets, generated poison images, model checkpoints, embeddings, concept projections, and run outputs are intentionally excluded from Git. See `.gitignore` for the complete policy.

## Concept inference

Concept banks can be learned in two ways.

### Multimodal concept bank

ConceptNet is queried for concepts related to the dataset classes, and CLIP text embeddings are used as concept vectors.

```bash
mkdir -p output/cifar10_output

python learn_concepts_multimodal.py \
  --classes cifar10 \
  --backbone-name clip:RN50 \
  --out-dir output/cifar10_output \
  --recurse 1
```

Use `--classes cifar100` and a corresponding output directory for CIFAR-100.

### Dataset-derived CAVs

When positive and negative samples are available for each concept, linear SVMs are trained in the backbone embedding space:

```bash
python learn_concepts_dataset.py \
  --dataset-name cub \
  --backbone-name resnet18_cub \
  --C 0.001 0.01 0.1 1.0 10.0 \
  --n-samples 100 \
  --out-dir output/cub_output
```

## Image unlearning workflow

The experiment workflow is:

```text
train/load end-to-end model
        -> infer and rank concepts with PCBM
        -> select a confusing source concept/class
        -> generate poisoned target-class images
        -> fine-tune the end-to-end model
        -> evaluate forgetting and retained utility
```

The current scripts preserve the paths and experiment switches used during development. Review the active target class, poisoning label strategy, checkpoint path, and poison-image path before each run.

For CIFAR-10, run the configured experiment with:

```bash
python test_script.py
```

For CIFAR-100, run:

```bash
python test_script_cifar100.py
```

The poison-generation script currently implements fixed-region image composition. Concept-localized masks or attribution-guided regions should be treated as a separate experimental variant rather than assumed to be equivalent to the fixed-region baseline.

## Evaluation

The main image-unlearning measurements are:

- accuracy on the target class in the original training set;
- accuracy on the target class in the test set;
- global or retained-class accuracy;
- forgetting rate under membership inference;
- fine-tuning time;
- cross-entropy loss distributions before and after unlearning.

For a reliable comparison, report multiple random seeds and include random-label, no-mask, random-mask, fixed-region, full-mask, and concept-guided masking ablations.

## Checkpoints and generated artifacts

Large files are not stored in the Git repository. If pretrained checkpoints or generated poison datasets are released, publish them separately using GitHub Releases, Zenodo, Hugging Face, or another artifact host, and document their checksums and expected local paths here.

Do not commit:

- CIFAR archives or extracted dataset batches;
- generated adversarial or poisoned images;
- `.pt`, `.pth`, `.ckpt`, or `.pkl` model files;
- cached `.npy`/`.npz` embeddings and projections;
- `output/`, `checkpoints/`, `runs/`, or experiment logs;
- IDE settings, Python bytecode, or virtual environments.

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

The concept inference components build on [Post-hoc Concept Bottleneck Models](https://arxiv.org/abs/2205.15480) by Mert Yuksekgonul, Maggie Wang, and James Zou. The original MIT license notice is retained in `LICENSE`.

## License

This repository is distributed under the MIT License. See `LICENSE` for details.
