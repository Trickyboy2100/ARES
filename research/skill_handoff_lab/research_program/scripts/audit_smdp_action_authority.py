#!/usr/bin/env python3
"""Audit executable, outcome-changing actions in completed EXP-003 physical trials.

No simulator calls, model fitting, oracle result injection, or claim of causal
generalization. For deterministic crossed grids, stage1 effect is calculated
only within the same scene, mass, and actual predecessor grasp choice.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

PROGRAM = Path(__file__).resolve().parents[1]
DEFAULT = PROGRAM / "evidence/EXP-003/2026-11-02/contract_sac/analysis/SUMMARY.json"


def analyze(summary):
    rows = summary["rows"]
    assert len(rows) == summary["complete_physical_episodes"] == 596
    assert sum(r["steps"] for r in rows) == summary["complete_control_steps"] == 59200

    def subset(name):
        return [r for r in rows if r["group"] == name]

    def crossed(group, parameter):
        trials = subset(group)
        units = defaultdict(list)
        contexts = defaultdict(list)
        for r in trials:
            key = (r["seed"], r["mass"])
            contexts[key].append(r)
            units[(r["seed"], r["mass"], round(float(r["grasp"]), 5))].append(r)
        stage1 = []
        invalid_predecessor = []
        for key, trials_for_grasp in sorted(units.items()):
            if all(r["failure_class"] == "predecessor_not_verified" for r in trials_for_grasp):
                invalid_predecessor.append(key)
                continue
            # Some invalid predecessor rows lack a stage1 action and are excluded.
            valid = [r for r in trials_for_grasp
                     if r["failure_class"] != "predecessor_not_verified"]
            levels = {float(r[parameter]) for r in valid if r.get(parameter) is not None}
            if len(levels) < 2:
                continue
            wins = sum(bool(r["success"]) for r in valid)
            stage1.append({
                "scene_mass_grasp": key, "actions_executed": len(valid),
                "unique_stage1_levels": len(levels),
                "successes": wins, "mixed_outcome": 0 < wins < len(valid),
            })
        return {
            "episodes": len(trials),
            "scene_clusters": len({r["seed"] for r in trials}),
            "crossed_scene_mass_contexts": len(contexts),
            "grasp_groups_with_executed_stage1_variants": len(stage1),
            "grasp_groups_with_mixed_binary_success": sum(x["mixed_outcome"] for x in stage1),
            "predecessor_unverified_grasp_groups": len(invalid_predecessor),
            "within_grasp_stage1_variation": stage1,
            "scope": "Discrete native final success in the tested action grid only",
        }

    dev = subset("evaluation512")
    bycase = defaultdict(list)
    for r in dev:
        bycase[(r["seed"], r["mass"])].append(r)
    assert len(dev) == 64 and len(bycase) == 8
    assert all(len(v) == 8 for v in bycase.values())
    all_win = [k for k, v in bycase.items() if all(x["success"] for x in v)]
    all_fail = [k for k, v in bycase.items() if not any(x["success"] for x in v)]
    mixed = [k for k, v in bycase.items() if any(x["success"] for x in v)
             and not all(x["success"] for x in v)]

    online = subset("sac_recovered/episodes")
    bytrain = defaultdict(list)
    for r in online:
        bytrain[(r["seed"], r["mass"])].append(r)
    ambiguous = [k for k, v in bytrain.items()
                 if any(x["success"] for x in v) and not all(x["success"] for x in v)]

    depth = crossed("depth_grid/episodes", "hook_height")
    accel = crossed("accel_grid/episodes", "successor_acceleration")
    assert (depth["grasp_groups_with_executed_stage1_variants"],
            depth["grasp_groups_with_mixed_binary_success"]) == (9, 0)
    assert (accel["grasp_groups_with_executed_stage1_variants"],
            accel["grasp_groups_with_mixed_binary_success"]) == (9, 1)
    assert len(all_win) == 7 and len(all_fail) == 1 and not mixed
    assert all_fail[0][0] == 746204
    assert len(online) == 287 and len(bytrain) == 64 and len(ambiguous) == 24

    return {
        "data_source": "Committed physical SUMMARY.json, not network/model predictions",
        "native_physics_episodes_reanalyzed": len(rows),
        "new_physics_episodes": 0,
        "stage1_hook_height": depth,
        "stage1_transfer_acceleration": accel,
        "development_8_policies": {
            "physical_scene_clusters": len(bycase),
            "methods_per_scene": 8,
            "all_8_pass_contexts": len(all_win),
            "all_8_fail_contexts": all_fail,
            "discriminating_contexts": len(mixed),
            "caution": "These are outcomes for observed methods, not a global action-space upper bound",
        },
        "online_exploration": {
            "episodes": len(online),
            "repeated_scene_clusters": len(bytrain),
            "contexts_with_mixed_outcomes": len(ambiguous),
            "causality_limit": "Both predecessor and successor actions changed; no isolated stage1 effect",
        },
        "interpretation": (
            "The current discrete height parameter has zero tested terminal-success "
            "authority after verified grasp; acceleration has one narrow budget effect. "
            "No conclusion for untested successor action types, actual contact trajectory "
            "or continuous reward. Baseline saturation is not RL superiority."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = analyze(json.loads(args.summary.read_text(encoding="utf-8")))
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
