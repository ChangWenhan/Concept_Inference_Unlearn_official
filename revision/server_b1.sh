#!/bin/bash
# Server 59: all-class sweep, CIFAR-10 (10 classes) + CIFAR-100 first half.
cd ~/Workspace/post-hoc-cbm-main || exit 1
export PYTHONNOUSERSITE=1
export REVISION_PY=/home/cwh/anaconda3/envs/graph_edit/bin/python
LOG=revision/work/logs/server_b1.log
mkdir -p revision/work/logs
exec >> "$LOG" 2>&1
echo "================ 59 server start $(date) py=$REVISION_PY ================"
bash revision/run_sweep.sh cifar10 0 10 0
bash revision/run_sweep.sh cifar100 0 50 0
echo "================ 59 server done $(date) ================"
