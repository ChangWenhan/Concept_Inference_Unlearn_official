#!/bin/bash
# Server 141: all-class sweep, CIFAR-100 second half + deer multi-seed runs.
# Note: this machine keeps its packages in the user site, so PYTHONNOUSERSITE
# must NOT be set here.
cd ~/Workspace/post-hoc-cbm-main || exit 1
export REVISION_PY=/home/hp/anaconda3/envs/CL/bin/python
LOG=revision/work/logs/server_b2.log
mkdir -p revision/work/logs
exec >> "$LOG" 2>&1
echo "================ 141 server start $(date) py=$REVISION_PY ================"
bash revision/run_sweep.sh cifar100 50 100 0
echo "--- deer seeds 43/44 $(date) ---"
$REVISION_PY -m revision.batch --dataset cifar10 --classes 4 --modes localized,center,none \
    --labels targeted,random --integrities full --seeds 43,44 --device cuda --skip-poison
echo "================ 141 server done $(date) ================"
