#!/usr/bin/env bash
set -euo pipefail
# From ARES-SCI-program repo; requires pinned ManiSkill and public raw archive unpacked.
# These are SEEN development/training seeds, never sealed tests.
py="$1"
evidence="$2"
out="$3"
[[ ! -e "$out" ]] || { echo 'Refusing to overwrite evidence'; exit 2; }
mkdir -p "$out"
tools=research/skill_handoff_lab/tools
run() { bash "$tools/handoff_offline.sh" "$py" "$@"; }
# Sequential GPU use avoids the observed concurrent PhysX failure. No devices/network.
for mode in ordinary suffix budget; do
 run "$tools/run_budget_feedback.py" --mode "$mode" --recipe cql_warm --episodes 96 --seed 31501 --start 754100 --dev-start 753200 --dev-count 16 --warm "$evidence/warm_v4" --out "$out/train_${mode}_v4_31501"
 run "$tools/train_reachable_feedback.py" --parent "$out/train_${mode}_v4_31501" --out "$out/local_${mode}_31501" --budget 10000
 run "$tools/train_reachable_feedback.py" --parent "$out/train_${mode}_v4_31501" --out "$out/deployment_labels_${mode}" --budget 2840 --head-label-policy deterministic
 done
# Historical interrupted suffix run is archived, not artificially recreated.
# Exact optimizer state is saved; PhysX restored contact caches are NOT bit-exact.
