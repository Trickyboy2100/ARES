#!/usr/bin/env bash
set -euo pipefail
repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
workspace="$(cd "$repo/../.." && pwd)"
out="${1:?Provide a fresh absolute output directory below the ARES-Skilllab workspace}"
[[ "$out" == "$workspace/"* && ! -e "$out" ]] || { echo 'Output must be new and inside the isolated writable workspace'; exit 1; }
mkdir -p "$out/audit"
cd "$repo"
py="$workspace/envs/method-evolution-20261028/bin/python"
tools_dir=research/skill_handoff_lab/tools
old="$workspace/runs/single_paper_20261028"
run() { bash "$tools_dir/handoff_offline.sh" "$py" "$@"; }
run "$tools_dir/handoff_opportunity.py" --source "$old/opportunity64" --out "$out/audit"
run "$tools_dir/handoff_contact_diagnosis.py" --source "$old/opportunity64" --out "$out/audit"
run "$tools_dir/handoff_contract_value.py" --old "$old" --out "$out" --phase fit
run "$tools_dir/handoff_contract_value.py" --old "$old" --out "$out" --phase eval --batches 2
run "$tools_dir/handoff_tracking_probe.py" --out "$out/tracking_probe"
run "$tools_dir/handoff_contract_value.py" --old "$old" --out "$out" --phase fit-contact --contact "$out/audit/CONTACT_DIAGNOSIS.csv"
run "$tools_dir/handoff_contract_value.py" --old "$old" --out "$out" --phase eval-contact --batches 2
run "$tools_dir/handoff_contract_value.py" --old "$old" --out "$out" --phase eval-contact-paired --batches 2
# The exact historical first-phase source is retained for provenance.
cp research/skill_handoff_lab/deliverables/client_sci_20261027/12_SCI_OPPORTUNITY_20261029/source_v{1,2}_handoff_contract_value.py "$out/audit/"
run "$tools_dir/handoff_science_report.py" --run "$out" --audit "$out/audit" --out "$out/summary"
