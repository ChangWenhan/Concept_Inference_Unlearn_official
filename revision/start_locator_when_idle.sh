#!/bin/bash
# Waits until the main local queue (server_local.sh) exits, then runs the
# E1 locator ablation. Keeps the GPU strictly serialized.
cd /home/cwh/Workspace/post-hoc-cbm-main || exit 1
LOG=revision/work/logs/locator_watcher.log
mkdir -p revision/work/logs
exec >> "$LOG" 2>&1
echo "watcher start $(date); waiting for server_local.sh to finish"
while pgrep -f "bash revision/server_local.sh" >/dev/null 2>&1; do
  sleep 120
done
echo "main local queue finished $(date); starting locator ablation"
bash revision/run_locator_ablation.sh
echo "watcher done $(date)"
