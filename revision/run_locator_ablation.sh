#!/bin/bash
# E1 locator ablation: concept localizers {clip(already), gradcam, margin} on
# the paired representative classes (CIFAR-10 deer / CIFAR-100 boy).
# Started automatically by start_locator_when_idle.sh once the main local
# queue (server_local.sh) has finished, so it never competes for the GPU.
cd /home/cwh/Workspace/post-hoc-cbm-main || exit 1
export PYTHONNOUSERSITE=1
PY=/home/cwh/anaconda3/envs/torch/bin/python
LOG=revision/work/logs/locator_ablation.log
mkdir -p revision/work/logs
exec >> "$LOG" 2>&1
echo "================ locator ablation start $(date) ================"

for spec in "cifar10 4" "cifar100 11"; do
  set -- $spec; ds=$1; cls=$2
  for loc in gradcam margin; do
    echo "--- poison $ds class$cls localized_$loc $(date)"
    $PY -m revision.poison_gen --dataset "$ds" --target-class "$cls" --mode "localized_$loc" --device cuda \
      || echo "poison failed: $ds $cls $loc"
  done
done

echo "--- training gradcam/margin runs (seeds 42,43) $(date)"
$PY -m revision.batch --dataset cifar10 --classes 4 --modes localized_gradcam,localized_margin \
    --labels targeted --integrities full --seeds 42,43 --device cuda --skip-poison
$PY -m revision.batch --dataset cifar100 --classes 11 --modes localized_gradcam,localized_margin \
    --labels targeted --integrities full --seeds 42,43 --device cuda --skip-poison

echo "--- localization stats clip/gradcam/margin $(date)"
$PY -m revision.analyze_localization --dataset cifar10 --target-class 4 --limit 500 --locator gradcam --device cuda
$PY -m revision.analyze_localization --dataset cifar10 --target-class 4 --limit 500 --locator margin --device cuda
$PY -m revision.analyze_localization --dataset cifar100 --target-class 11 --limit 500 --locator clip --device cuda
$PY -m revision.analyze_localization --dataset cifar100 --target-class 11 --limit 500 --locator gradcam --device cuda
$PY -m revision.analyze_localization --dataset cifar100 --target-class 11 --limit 500 --locator margin --device cuda

$PY -m revision.collect_results
echo "================ locator ablation done $(date) ================"
