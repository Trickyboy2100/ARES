#!/usr/bin/env bash
set -euo pipefail
# Run from the pinned ARES repository root. New output directory only.
# These seeds reproduce seen development; they are NOT new/sealed evaluation.
python_binary="$1"
prior_root="$2"
run_root="$3"
[[ ! -e "$run_root" ]] || { echo 'Refusing to overwrite existing evidence'; exit 2; }
mkdir -p "$run_root/threeway"
tools_dir=research/skill_handoff_lab/tools
run() { bash "$tools_dir/handoff_offline.sh" "$python_binary" "$@"; }
run "$tools_dir/run_option_coverage.py" --out "$run_root/design" --start 752000 --count 32 --split replication_training
run "$tools_dir/run_option_coverage.py" --out "$run_root/development" --start 752200 --count 32 --split replication_seen_development
run "$tools_dir/run_split_refresh_pilot.py" --reference "$run_root/design" --out "$run_root/split_refresh"
run "$tools_dir/extend_option_coverage.py" --reference "$run_root/design" --pilot "$run_root/split_refresh" --out "$run_root/extended_design"
run "$tools_dir/extend_option_coverage.py" --reference "$run_root/development" --out "$run_root/extended_development"
for seed in 31401 31402; do
 run "$tools_dir/train_option_improvement.py" --prior "$prior_root" --data "$run_root/design/TRANSITIONS.json" --out "$run_root/train_$seed" --seed "$seed" --methods sac_min,sac_mean,vi_mean,vi_expectile,usac,handoff,lower_bound,fqi,iql,supervised,stap,bee,handoff_complete
 run "$tools_dir/train_option_improvement.py" --prior "$prior_root" --data "$run_root/extended_design/TRANSITIONS.json" --out "$run_root/threeway/train_$seed" --seed "$seed" --successor-actions 3 --methods sac_min,sac_mean,vi_mean,vi_expectile,usac,handoff,lower_bound,fqi,iql,supervised,stap,bee,handoff_complete,cql
 for method in sac_mean vi_mean handoff_complete; do
  run "$tools_dir/run_option_improvement_online.py" --prior "$prior_root" --design "$run_root/extended_design/TRANSITIONS.json" --checkpoint "$run_root/threeway/train_$seed/${method}_288.pt" --out "$run_root/threeway/online_${method}_$seed"
 done
done
run "$tools_dir/eval_option_improvement.py" --root "$run_root" --panel "$run_root/development" --out "$run_root/eval_development"
run "$tools_dir/eval_option_improvement.py" --root "$run_root/threeway" --panel "$run_root/extended_development" --out "$run_root/threeway/eval_development"
run "$tools_dir/eval_option_online.py" --root "$run_root/threeway" --panel "$run_root/extended_development" --out "$run_root/threeway/online_evaluation" --direct --direct-n 8
