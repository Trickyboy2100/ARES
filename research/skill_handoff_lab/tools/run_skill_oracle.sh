#!/usr/bin/env bash
# Bounded research-only planning worker. No hardware, shared maintenance or pkill.
set -euo pipefail
if [ "$#" -ne 4 ]; then echo 'BATCH_JSON NEW_OUTPUT GPU_INDEX TIMEOUT_SECONDS' >&2; exit 2; fi
ARES_LAB_ROOT=/home/andyee/robot-research/ARES-Skilllab
export PYTHONPATH="$ARES_LAB_ROOT/envs/warp-1.13.0:$ARES_LAB_ROOT/envs/oracle-deps:$ARES_LAB_ROOT/snapshots/ares-curobo-pinned-8e734f3:$ARES_LAB_ROOT/snapshots/ARES-190-oracle-20261013:$ARES_LAB_ROOT/snapshots/bootstrap-3ec7bd9-9de17d2/ARES-R/src"
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export CUDA_VISIBLE_DEVICES="$3"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
umask 077
exec timeout -k 10 "$4" /home/andyee/.jnl263_round79_groot_n16_env/bin/python \
  "$(dirname "$0")/replay_worker_compat.py" --batch "$1" --output "$2" --reuse-goal-runtime --exact-kd --gpu-post --follow-seconds "${ARES_FOLLOW_SECONDS:-0}"
