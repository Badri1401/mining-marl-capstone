#!/usr/bin/env bash
#
# train.sh — launch the final MADDPG training run.
#
# Usage:
#   ./scripts/train.sh                 # train with default settings (final config)
#   ./scripts/train.sh 50000           # train for 50,000 episodes
#   ENVS=8 THREADS=6 ./scripts/train.sh # tune parallel env workers / torch threads
#
# Output: src/results/maddpg_v3_<timestamp>_final/
#   ├── checkpoints/episode_*.pt
#   ├── training_log.csv
#   └── graphs/
#
set -euo pipefail

# Move to the src/ directory regardless of where this is called from.
cd "$(dirname "$0")/../src"

EPISODES="${1:-999999}"     # runs until the curriculum completes by default
ENVS="${ENVS:-4}"           # parallel environment workers
THREADS="${THREADS:-$(( $(nproc 2>/dev/null || echo 4) / 2 ))}"

echo "Training MADDPG (v2_baseline + v3_smooth) — episodes=${EPISODES}, envs=${ENVS}, threads=${THREADS}"

python3 -u train_maddpg.py \
    --episodes "${EPISODES}" \
    --reward-mode v2_baseline \
    --curriculum-mode v3_smooth \
    --num-parallel-envs "${ENVS}" \
    --torch-threads "${THREADS}" \
    --results-tag final \
    2>&1 | tee "training_$(date +%Y%m%d_%H%M%S).log"
