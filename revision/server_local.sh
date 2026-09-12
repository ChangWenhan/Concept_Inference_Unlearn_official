#!/bin/bash
# Local machine: boy matrix completion + seeds; E3 remaining (C10+C100);
# E5 side effects; E6 CIFAR-10-C; E7 timing; E1 CIFAR-100.
cd /home/cwh/Workspace/post-hoc-cbm-main || exit 1
export PYTHONNOUSERSITE=1
PY=/home/cwh/anaconda3/envs/torch/bin/python
LOG=revision/work/logs/server_local.log
mkdir -p revision/work/logs
exec >> "$LOG" 2>&1
echo "================ local server start $(date) ================"

echo "--- boy full matrix completion $(date) ---"
$PY -m revision.batch --dataset cifar100 --classes 11 --modes localized,center,random,full,none \
    --labels targeted,random,keep --integrities full,half --device cuda --skip-poison

echo "--- boy seeds 43/44 $(date) ---"
$PY -m revision.batch --dataset cifar100 --classes 11 --modes localized,center,none \
    --labels targeted,random --integrities full --seeds 43,44 --device cuda --skip-poison

echo "--- E3 CIFAR-10 remaining $(date) ---"
$PY -m revision.batch --dataset cifar10 --classes 0,6,7,9 --modes localized,center,none \
    --labels targeted --integrities full --device cuda --skip-poison
$PY -m revision.batch --dataset cifar10 --classes 0,6,7,9 --modes localized \
    --labels random --integrities full --device cuda --skip-poison
$PY -m revision.batch --dataset cifar10 --classes 0,6,7,9 --modes localized \
    --labels targeted --integrities full --seeds 43 --device cuda --skip-poison

echo "--- E3 CIFAR-100 poison + runs $(date) ---"
for c in 0 17 28 30; do
  for m in localized center; do
    $PY -m revision.poison_gen --dataset cifar100 --target-class $c --mode $m --device cuda
  done
done
$PY -m revision.batch --dataset cifar100 --classes 0,17,28,30 --modes localized,center,none \
    --labels targeted --integrities full --device cuda --skip-poison
$PY -m revision.batch --dataset cifar100 --classes 0,17,28,30 --modes localized \
    --labels random --integrities full --device cuda --skip-poison
$PY -m revision.batch --dataset cifar100 --classes 0,17,28,30 --modes localized \
    --labels targeted --integrities full --seeds 43 --device cuda --skip-poison

echo "--- E5 recovery / side effects / long run $(date) ---"
$PY -m revision.recover --run revision/work/runs/cifar10_c4_localized_targeted_full_s42 --epochs 10 --device cuda
$PY -m revision.recover --run revision/work/runs/cifar10_c4_localized_targeted_full_s42 --epochs 10 --include-target --device cuda
$PY -m revision.side_effects --run revision/work/runs/cifar10_c4_localized_targeted_full_s42 --device cuda
$PY -m revision.run_unlearn --dataset cifar10 --target-class 4 --mode localized --labels targeted \
    --integrity full --epochs 40 --out revision/work/runs/cifar10_c4_localized_targeted_full_e40 --device cuda
$PY -m revision.evaluate --run revision/work/runs/cifar10_c4_localized_targeted_full_e40 --device cuda --skip-celd

echo "--- E6 CIFAR-10-C $(date) ---"
$PY -m revision.noisy_exp eval --run revision/work/runs/cifar10_c4_localized_targeted_full_s42 --corruption gaussian_noise --severity 5 --device cuda
$PY -m revision.noisy_exp eval --run revision/work/runs/cifar10_c4_localized_targeted_full_s42 --corruption defocus_blur --severity 5 --device cuda
$PY -m revision.noisy_exp eval --run revision/work/runs/cifar10_c4_localized_targeted_full_s42 --corruption brightness --severity 5 --device cuda
$PY -m revision.noisy_exp stats --target-class 4 --corruption gaussian_noise --severity 5 --limit 200 --device cuda
$PY -m revision.noisy_exp stats --target-class 4 --corruption defocus_blur --severity 5 --limit 200 --device cuda
for c in gaussian_noise defocus_blur brightness; do
  $PY -m revision.noisy_exp gen --target-class 4 --corruption $c --severity 5 --corrupt-order before --device cuda
  $PY -m revision.noisy_exp train --target-class 4 --corruption $c --severity 5 --corrupt-order before --labels targeted --epochs 20 --device cuda
  $PY -m revision.evaluate --run revision/work/runs/cifar10_c4_localized_corrupt_${c}_s5_before_targeted_full_s42 --device cuda --skip-celd
done
$PY -m revision.noisy_exp gen --target-class 4 --corruption gaussian_noise --severity 5 --corrupt-order after --device cuda
$PY -m revision.noisy_exp train --target-class 4 --corruption gaussian_noise --severity 5 --corrupt-order after --labels targeted --epochs 20 --device cuda
$PY -m revision.evaluate --run revision/work/runs/cifar10_c4_localized_corrupt_gaussian_noise_s5_after_targeted_full_s42 --device cuda --skip-celd

echo "--- E7 timing + E1 CIFAR-100 $(date) ---"
$PY -m revision.timing_report --device cuda --sample 64
for c in 0 11 17 28 30; do
  $PY -m revision.analyze_localization --dataset cifar100 --target-class $c --limit 200 --device cuda
done

echo "--- collect results $(date) ---"
$PY -m revision.collect_results
echo "================ local server done $(date) ================"
