#!/usr/bin/env python3
"""Independent source-CSV audit: EXP-003 matched policies and scene-cluster CI.

Stdlib only. Reads physical EVAL_EPISODES.csv, not the existing PAIRS.json
or SUMMARY. Prints exact paired outcomes, exact binomial-discordance p, and
EXHAUSTIVE nonparametric cluster-bootstrap distribution (6^6 resamples).
The bootstrap is descriptive and is not an independent performance test.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/EXP-003/2026-11-01/component_A/EVAL_EPISODES.csv"
METHODS = {
    "strong_search": ("controls_task", "search", "task"),
    "sac_full": ("eval_full", "model", "full"),
    "sac_y_only": ("eval_y_only", "model", "y_only"),
    "sac_task": ("eval_task", "model", "task"),
    "sac_full_2048": ("eval_full_2048", "model", "full"),
    "sac_y_only_2048": ("eval_y_only_2048", "model", "y_only"),
    "sac_task_2048": ("eval_task_2048", "model", "task"),
}


def load(path: Path) -> dict:
    ret = {method: {} for method in METHODS}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for method, (run, policy, authority) in METHODS.items():
                if (row["run"], row["policy"], row["authority"]) != (run, policy, authority):
                    continue
                key = (int(row["seed"]), row["family"],
                       tuple(float(v) for v in json.loads(row["bias_mm"])))
                assert row["success"] in ("True", "False"), row["success"]
                assert key not in ret[method], (method, key)
                ret[method][key] = row["success"] == "True"
    for method, data in ret.items():
        assert len(data) == 36, (method, len(data))
        assert len({key[0] for key in data}) == 6, method
        assert sorted(set(key[1] for key in data)) == ["Y", "YZ"]
    reference = set(ret["strong_search"])
    assert all(set(data) == reference for data in ret.values())
    return ret


def exact_p(rescue: int, regression: int) -> float:
    n = rescue + regression
    if n == 0:
        return 1.0
    return min(1., 2. * sum(math.comb(n, j) for j in range(
        min(rescue, regression)+1)) / (2**n))


def exact_cluster_bootstrap(deltas: dict[int, list[int]]) -> dict:
    values = list(deltas.values())
    n = len(values)
    assert all(len(x) == 6 for x in values)
    fractions = []
    for indices in product(range(n), repeat=n):
        d = sum(sum(values[i]) for i in indices)
        fractions.append(d / (6*n))
    fractions.sort()
    assert len(fractions) == n**n
    return {
        "enumerations": len(fractions),
        "low_95_pp": 100 * fractions[int(.025*(len(fractions)-1))],
        "high_95_pp": 100 * fractions[math.ceil(.975*(len(fractions)-1))],
        "p_bootstrap_difference_nonpositive_descriptive":
            sum(d <= 0 for d in fractions) / len(fractions),
    }


def compare(base: dict, method: dict) -> dict:
    cases = sorted(base)
    rescue = sum(not base[k] and method[k] for k in cases)
    regress = sum(base[k] and not method[k] for k in cases)
    assert rescue-regress == sum(method.values())-sum(base.values())
    clusters = defaultdict(list)
    by_family = defaultdict(lambda: {"n": 0, "base": 0, "candidate": 0})
    for k in cases:
        clusters[k[0]].append(int(method[k])-int(base[k]))
        item = by_family[k[1]]
        item["n"] += 1
        item["base"] += int(base[k])
        item["candidate"] += int(method[k])
    return {
        "n_conditions": len(cases), "n_distinct_scenes": len(clusters),
        "baseline_success": sum(base.values()), "candidate_success": sum(method.values()),
        "rescue": rescue, "regression": regress,
        "net_gain_pp": 100. * (rescue-regress)/len(cases),
        "paired_exact_p": exact_p(rescue, regress),
        "scene_net_success_deltas": dict(sorted(
            (str(k), sum(v)) for k, v in clusters.items())),
        "families": dict(by_family),
        "all_scenes_exact_bootstrap": exact_cluster_bootstrap(clusters),
        "five_scene_sensitivity_exclude_732000":
            exact_cluster_bootstrap({k: v for k, v in clusters.items() if k != 732000}),
    }


def main() -> int:
    data = load(SOURCE)
    evaluations = {
        "task_sac_vs_strong_search": compare(data["strong_search"], data["sac_task"]),
        "task_sac_vs_full_sac": compare(data["sac_full"], data["sac_task"]),
        "task_sac_vs_y_only_sac": compare(data["sac_y_only"], data["sac_task"]),
        "task_sac_2048_vs_search": compare(data["strong_search"], data["sac_task_2048"]),
    }
    # Protect the review against accidental changes to source semantics.
    assert evaluations["task_sac_vs_strong_search"]["rescue"] == 4
    assert evaluations["task_sac_vs_strong_search"]["regression"] == 1
    assert evaluations["task_sac_vs_strong_search"]["baseline_success"] == 32
    assert evaluations["task_sac_vs_strong_search"]["candidate_success"] == 35
    assert evaluations["task_sac_vs_full_sac"]["baseline_success"] == 31
    assert evaluations["task_sac_vs_y_only_sac"]["baseline_success"] == 33
    print(json.dumps({"source": str(SOURCE), "scope":
        "6 initial scenes x 6 predeclared task variants; one used in stop engineering; development not sealed; single training seed",
        "comparisons": evaluations}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
