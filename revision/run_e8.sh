#!/bin/bash
# E8: top-M concept masking sensitivity (M = 1, 3, 5, 10), paired CIFAR-10 deer + CIFAR-100 boy.
cd "$(dirname "$0")/.." || exit 1
export PYTHONNOUSERSITE=1
PY=${REVISION_PY:-/home/cwh/anaconda3/envs/torch/bin/python}
LOG=revision/work/logs/run_e8.log
mkdir -p revision/work/logs
exec >> "$LOG" 2>&1
echo "================ E8 top-M start $(date) py=$PY ================"

for m in 1 3 5 10; do
  $PY -m revision.poison_gen_multi --dataset cifar10 --target-class 4 --top-m $m --device cuda
done

$PY -m revision.batch --dataset cifar10 --classes 4 --modes localized_m1,localized_m3,localized_m5,localized_m10 \
    --labels targeted --integrities full --device cuda --skip-poison

$PY -m revision.poison_gen_multi --dataset cifar10 --target-class 4 --top-m 3 --seed 43 --device cuda
$PY -m revision.batch --dataset cifar10 --classes 4 --modes localized_m3 \
    --labels random --integrities full --device cuda --skip-poison

for m in 1 3 5 10; do
  $PY -m revision.poison_gen_multi --dataset cifar100 --target-class 11 --top-m $m --device cuda
done

$PY -m revision.batch --dataset cifar100 --classes 11 --modes localized_m1,localized_m3,localized_m5,localized_m10 \
    --labels targeted --integrities full --device cuda --skip-poison

$PY -m revision.poison_gen_multi --dataset cifar100 --target-class 11 --top-m 3 --seed 43 --device cuda
$PY -m revision.batch --dataset cifar100 --classes 11 --modes localized_m3 \
    --labels random --integrities full --device cuda --skip-poison

echo "================ E8 top-M done $(date) ================"
