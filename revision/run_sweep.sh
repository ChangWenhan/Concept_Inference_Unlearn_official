#!/bin/bash
# All-class sweep for R2.4 ("all classes in the CIFAR datasets").
# Usage: run_sweep.sh <cifar10|cifar100> <start_class> <end_class_exclusive> [gpu_id]
# Portable across machines: set REVISION_PY to the env python; caller controls
# PYTHONNOUSERSITE (export it on self-contained envs, skip on user-site envs).
DATASET=${1:?usage: run_sweep.sh dataset start end [gpu]}
START=${2:?}
END=${3:?}
GPU=${4:-0}
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
PY=${REVISION_PY:-/home/cwh/anaconda3/envs/torch/bin/python}
LOG=revision/work/logs
mkdir -p "$LOG"
echo "================ sweep $DATASET [$START,$END) gpu=$GPU py=$PY start $(date) ================"

for c in $(seq "$START" $((END - 1))); do
  echo "[sweep] poison $DATASET class$c $(date)"
  $PY -m revision.poison_gen --dataset "$DATASET" --target-class "$c" --mode localized --device cuda \
    || { echo "[sweep] poison failed class$c, skipping"; continue; }
done

CLASSES=$(seq -s, "$START" $((END - 1)))
$PY -m revision.batch --dataset "$DATASET" --classes "$CLASSES" --modes localized --labels targeted \
    --integrities full --device cuda --skip-poison

echo "================ sweep $DATASET [$START,$END) done $(date) ================"
