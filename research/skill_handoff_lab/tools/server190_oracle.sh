#!/usr/bin/env bash
# One bounded planning-only worker. Never invokes robot execution or pkill.
set -euo pipefail
if [ "$#" -ne 2 ]; then echo 'usage: server190_oracle.sh BATCH_JSON NEW_OUTPUT_DIRECTORY'; exit 2; fi
ARES_LAB_ROOT=/home/andyee/robot-research/ARES-Skilllab
export PYTHONPATH="$ARES_LAB_ROOT/envs/warp-1.13.0:$ARES_LAB_ROOT/envs/oracle-deps:$ARES_LAB_ROOT/snapshots/ares-curobo-pinned-8e734f3:$ARES_LAB_ROOT/snapshots/ARES-190-oracle-20261013:$ARES_LAB_ROOT/snapshots/bootstrap-3ec7bd9-9de17d2/ARES-R/src"
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
exec timeout -k 10 240 /home/andyee/.jnl263_round79_groot_n16_env/bin/python \
  -m research.skill_handoff_lab.bench.oracle_executor --batch "$1" --output "$2"
