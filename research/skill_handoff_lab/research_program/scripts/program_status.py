#!/usr/bin/env python3
"""ARES SCI research-program index checker; stdlib only, read-only.

Usage (from repository root):
    python research/skill_handoff_lab/research_program/scripts/program_status.py status
    python research/skill_handoff_lab/research_program/scripts/program_status.py check
    python research/skill_handoff_lab/research_program/scripts/program_status.py next
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

PROGRAM = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "STATE.json",
    "PAPER_CONTRACT.md",
    "STAGE_ROADMAP.md",
    "EXPERIMENT_BACKLOG.md",
    "METHOD_STRATEGY.md",
    "EVIDENCE_LEDGER.md",
    "PRIOR_ART.md",
    "DECISIONS.md",
    "OPERATING_PROTOCOL.md",
    "CODEX_START.md",
    "paper/CLAIM_MATRIX.md",
    "experiments/README.md",
    "experiments/_TEMPLATE.md",
]

ALLOWED_STATUS = {
    "ready", "running", "queued", "done", "pivoted", "abandoned"
}


def read_state() -> dict:
    with (PROGRAM / "STATE.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def verify(state: dict) -> list[str]:
    problems = []
    for name in REQUIRED:
        if not (PROGRAM / name).is_file():
            problems.append(f"missing required file: {name}")
    if state.get("schema_version") != 1:
        problems.append("unsupported STATE schema_version")
    experiments = state.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return problems + ["STATE experiments missing or empty"]
    seen = set()
    by_id = {}
    for experiment in experiments:
        name = experiment.get("id")
        if not name or name in seen:
            problems.append(f"empty or duplicate experiment ID: {name!r}")
        seen.add(name)
        by_id[name] = experiment
        if experiment.get("status") not in ALLOWED_STATUS:
            problems.append(f"{name}: invalid status {experiment.get('status')!r}")
        card = experiment.get("card")
        if not isinstance(card, str) or not (PROGRAM / card).is_file():
            problems.append(f"{name}: experiment card missing: {card!r}")
        if experiment.get("stage") not in (state.get("stage_status") or {}):
            problems.append(f"{name}: unknown stage: {experiment.get('stage')!r}")

    for name, experiment in by_id.items():
        for dep in experiment.get("depends_on", []):
            if dep not in by_id:
                problems.append(f"{name}: nonexistent dependency {dep!r}")

    active = state.get("active_experiment")
    if active not in by_id:
        problems.append(f"active_experiment not in experiments: {active!r}")
    else:
        if by_id[active]["status"] not in {"ready", "running"}:
            problems.append(f"active experiment must be ready or running: {active}")
        for dep in by_id[active].get("depends_on", []):
            if by_id[dep]["status"] not in {"done", "pivoted"}:
                problems.append(f"active {active} depends on unfinished {dep}")

    # Check local Markdown links so index moves do not strand readers.
    # External URLs and fragment-only links are intentionally excluded.
    for doc_name in REQUIRED:
        if not doc_name.endswith(".md"):
            continue
        doc = PROGRAM / doc_name
        if not doc.exists():
            continue
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0].strip()
            if not target or target.startswith(("https://", "http://", "mailto:")):
                continue
            if not (doc.parent / target).exists():
                problems.append(f"{doc_name}: broken local link {target}")

    if not state.get("next_action"):
        problems.append("STATE next_action must be nonblank")
    newest = state.get("latest_evidence")
    if newest and not (PROGRAM / newest).exists():
        problems.append(f"latest_evidence path missing: {newest}")
    if "UNVERIFIED" in state.get("method_claim_status", ""):
        # Document marker is a guard against accidental unsupported promotion.
        claim_text = (PROGRAM / "paper/CLAIM_MATRIX.md").read_text(encoding="utf-8") \
            if (PROGRAM / "paper/CLAIM_MATRIX.md").exists() else ""
        if "UNVERIFIED" not in claim_text:
            problems.append("method claim unverified in STATE but not in claim matrix")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", choices=("status", "next", "check"),
                    default="status")
    args = ap.parse_args()
    try:
        state = read_state()
        problems = verify(state)
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"PROGRAM ERROR: {err}", file=sys.stderr)
        return 2

    if args.command == "check":
        if problems:
            print("PROGRAM CHECK FAIL")
            for problem in problems:
                print(" -", problem)
            return 1
        print(f"PROGRAM CHECK PASS | {len(state['experiments'])} tracked experiments "
              f"| active {state['active_experiment']}")
        return 0

    print(f"Program: {state['program']}")
    print(f"Current scientific claim: {state['method_claim_status']}")
    print(f"Stage: {state['current_stage']}")
    print(f"Active experiment: {state['active_experiment']}")
    exp = next((e for e in state["experiments"]
                if e["id"] == state["active_experiment"]), None)
    if exp:
        print(f"Experiment card: {PROGRAM / exp['card']}")
        print(f"Experiment status: {exp['status']}")
        print(f"Question: {exp['primary_claim']}")
    print(f"Next action: {state['next_action']}")
    if args.command == "status":
        print(f"Evidence: {PROGRAM / state['latest_evidence']}")
        print(f"Historical baseline commit: {state['baseline_commit']}")
        print(f"Unresolved questions: {len(state.get('unresolved_questions', []))}")
    if problems:
        print(f"WARNING: {len(problems)} index issues; run 'check'", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
