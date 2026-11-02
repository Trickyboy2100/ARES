#!/usr/bin/env python3
"""Independent casewise audit of the 2026-11-02 EXP-003 train96 physics panel.

Reads actual EPISODES.csv, not its prior SUMMARY. No simulator or torch needed.
The opportunity upper bound is limited to the six actually executed actions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/EXP-003/2026-11-02/skillresult/EPISODES.csv"


def main() -> None:
    contexts: dict[tuple[int, float, float], dict[tuple[float, float], int]] = defaultdict(dict)
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["group"] != "train96":
                continue
            case = (int(row["seed"]), float(row["mass"]), float(row["target"]))
            action = (round(float(row["grasp"]), 5), round(float(row["hook"]), 5))
            assert row["success"] in ("0", "1"), row
            assert action not in contexts[case], (case, action)
            contexts[case][action] = int(row["success"])

    assert len(contexts) == 16, len(contexts)
    assert sum(len(actions) for actions in contexts.values()) == 96
    assert len({case[0] for case in contexts}) == 8
    success_hist = Counter(sum(actions.values()) for actions in contexts.values())
    assert dict(success_hist) == {0: 7, 6: 7, 3: 1, 4: 1}, success_hist

    mixed = {}
    interactions = []
    fixed_success = 0
    for case, outcome in sorted(contexts.items()):
        grasps = sorted({a[0] for a in outcome})
        hooks = sorted({a[1] for a in outcome})
        assert len(grasps) == 2 and len(hooks) == 3 and len(outcome) == 6
        assert all((g, h) in outcome for g in grasps for h in hooks)
        fixed_success += outcome[(0.1, -0.065)]
        n = sum(outcome.values())
        if 0 < n < 6:
            mixed[str(case)] = {
                "successes": n,
                "parameter_outcomes": {f"{g:.3f}/{h:.3f}": outcome[(g, h)]
                                       for g in grasps for h in hooks},
            }
        # Binary-outcome interaction: does changing hook alter grasp contrast?
        grasp_differences = [outcome[(grasps[1], h)] -
                             outcome[(grasps[0], h)] for h in hooks]
        if len(set(grasp_differences)) != 1:
            interactions.append(str(case))

    # Strict pairwise reversals across goals under a shared physical initial scene.
    ranking_reversals = []
    for (seed, mass, goal), left in sorted(contexts.items()):
        if goal >= 0:
            continue
        right = contexts.get((seed, mass, -goal))
        assert right is not None, (seed, mass, goal)
        for a, b in combinations(sorted(set(left) & set(right)), 2):
            if (left[a] - left[b]) * (right[a] - right[b]) < 0:
                ranking_reversals.append([seed, mass, a, b])

    oracle = sum(any(outcomes.values()) for outcomes in contexts.values())
    result = {
        "source": str(SOURCE),
        "scope": "Already executed physics; 8 initial scene seeds x 2 declared goals; mass across scenes",
        "physics_episodes_reanalyzed": 96,
        "new_physics_steps": 0,
        "context_count": len(contexts),
        "context_success_histogram": dict(sorted(success_hist.items())),
        "mixed_contexts": mixed,
        "best_measured_fixed_100mm_grasp_minus65mm_hook": fixed_success,
        "measured_action_oracle": oracle,
        "success_label_grasp_x_hook_interactions": interactions,
        "strict_goal_ranking_reversals": ranking_reversals,
        "interpretation": "Local binary-success identifiability only, not absence of physical effects or all-parameter interactions",
    }
    assert (fixed_success, oracle, len(mixed), len(interactions),
            len(ranking_reversals)) == (8, 9, 2, 0, 0), result
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
