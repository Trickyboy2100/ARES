#!/usr/bin/env python3
"""Analyze committed physical results only; this script performs no simulation."""
from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/EXP-003/2026-11-02"

def load(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))

def tool_panel():
    contexts = defaultdict(list)
    with (BASE / "skillresult/EPISODES.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["group"] == "train96":
                contexts[(int(r["seed"]), float(r["mass"]), float(r["target"]))].append(r)
    assert len(contexts) == 16 and sum(len(x) for x in contexts.values()) == 96
    hist = Counter()
    mixed = []
    spreads = []
    for (seed, mass, target), rows in sorted(contexts.items()):
        assert len(rows) == 6 and len({(r["grasp"], r["hook"]) for r in rows}) == 6
        wins = sum(int(r["success"]) for r in rows)
        dist = [float(r["distance_m"]) for r in rows]
        mm = 1000 * (max(dist) - min(dist))
        hist[wins] += 1
        spreads.append(mm)
        if 0 < wins < 6:
            mixed.append(dict(seed=seed, mass=mass, target=target,
                              success=wins, distance_spread_mm=round(mm, 2)))
    assert sorted(hist.items()) == [(0, 7), (3, 1), (4, 1), (6, 7)]
    return dict(episodes=96, scene_clusters=8, target_conditions=16,
                success=49, success_hist=dict(sorted(hist.items())),
                mixed=mixed, largest_distance_spread_mm=round(max(spreads), 2),
                best_measured_fixed=8, finite_action_oracle=9)

def historical_pyramid():
    x = load("pyramid_contract/HISTORICAL_OPPORTUNITY.json")
    cases = x["cases"]
    assert len(cases) == 64 and x["episodes"] == 576
    prev = {r["seed"] for r in cases if r["verified_A_different_suffix_outcomes"]}
    suffix = {r["seed"] for r in cases if r["successor_action_sensitive_prefixes"]}
    joint = [r["seed"] for r in cases if r["joint_oracle"] and
             not r["oracle_A_Cnom"] and not r["oracle_C_Anom"]]
    assert (len(prev), len(suffix), len(prev & suffix), len(joint)) == (9, 13, 6, 2)
    assert x["nominal"] == 52 and x["best_fixed"] == 59 and x["joint_oracle"] == 62
    return dict(episodes=576, scenes=64, nominal=52, best_fixed=59,
                measured_action_oracle=62, predecessor_sensitive=9,
                successor_sensitive=13, both_sensitive=sorted(prev & suffix),
                both_parameters_needed_vs_nominal=joint)

def latest_pyramid():
    s = load("pyramid_contract/SUMMARY.json")
    pairs = load("pyramid_contract/MATCHED_CONTROL.json")
    out = {}
    for name, n, wins in (("pilot", 18, 0), ("symmetry", 36, 29), ("feasible", 36, 36)):
        rows = s[name]["rows"]
        assert len(rows) == n and sum(bool(r["contract_success"]) for r in rows) == wins
        contexts = sorted({(r["seed"], r["yaw"]) for r in rows})
        out[name] = dict(episodes=n, success=wins,
                         scene_clusters=len({r["seed"] for r in rows}),
                         geometry_queries=sum(len(r.get("orientation_queries", [])) for r in rows),
                         success_by_context={
                             f"{seed}/{yaw}":sum(int(r["contract_success"]) for r in rows
                                              if r["seed"] == seed and r["yaw"] == yaw)
                             for seed, yaw in contexts})
    assert len(pairs) == 36 and all(r["prefix_exact"] for r in pairs)
    assert sum(r["rescue"] for r in pairs) == 7 and sum(r["regression"] for r in pairs) == 0
    assert out["feasible"]["geometry_queries"] == 144
    return dict(variants=out, exact_prefix_pairs=36,
                paired_rescues=7, paired_regressions=0,
                total_episodes=90, optimizer_updates=0)

def main():
    print(json.dumps(dict(
        tool=tool_panel(), old_pyramid=historical_pyramid(),
        new_pyramid=latest_pyramid(), new_simulation_steps=0,
        claim="UNVERIFIED_INDEPENDENT_ALGORITHM_GAIN"),
        ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
