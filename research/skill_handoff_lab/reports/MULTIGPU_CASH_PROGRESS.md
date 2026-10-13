# Isaac 5.1 / multi-GPU CASH progress, 2026-10-13

This round continues `codex/ares-skill-handoff-p2-1-20261013@2d4a97b`, preserving the P1/P2 and P2.1 negative results. The user explicitly selects Isaac Sim 5.1. ARES-R remains unchanged.

## Executed work

- Created `envs/cash` with system-site packages to reuse existing torch; installed Gymnasium 1.4.0, SB3 2.9.0 and SB3-contrib 2.9.0 without replacing torch/CUDA.
- Added a Gymnasium environment with fixed observation/action spaces, D0/D1/D2 stages, mask support and fixed terminal shapes.
- Added an immutable planning-tree cache that verifies request/result/validation file SHA-256, rejects train/test scene-group leakage and research training on artificial fixtures, and checks validation reports against recorded flags.
- Enabled planning-only rewards explicitly for independently validated cuRobo paths. Execution evidence remains distinct. Timeout and infrastructure failure are not negative task labels.
- Added supervised terminal-candidate prediction and CPU MaskablePPO train/test entry points. The PPO entry is a standard baseline, not the auxiliary-head CASH algorithm.
- Ran Gymnasium API checks and a 16-step MaskablePPO fixture update. This is software validation, not a research training run.
- Read the existing 5.1 USD through pxr and rendered 181 instance-proxy visual meshes on CPU, using all faces and excluding collision-only meshes. The image shows the actual USD geometry with simplified colors, not Isaac RTX output or simulated task success.
- Prepared an Isaac 5.1-only headless Replicator capture and isolated process-group timeout manager. GPU execution is pending.
- Prepared a bounded read-only server inventory script. No server connection attempted because its identity is not confirmed.

## Actual findings

The host `nvidia-smi` still fails with driver/library mismatch. Two wrist-camera USD references use `https:/...` and do not resolve in the pxr read. The canonical USD has not been overwritten.

USD arm bases are left `[0.05,1.2,1.2]` and right `[-0.35,1.2,1.2]`. Positions alone admit a BODY-to-WORLD rotation/translation, but comparing complete USD base transforms with the pinned historical controller-model base transforms gives inconsistent inferred WORLD-to-BODY transforms: about 1.498 m translation disagreement and 2.449 Frobenius rotation-matrix disagreement. This is a frame-contract mismatch to investigate, not permission to silently change real model constants. The source probe used q=0 URDF against saved articulated USD poses; link6 disagreement from that probe alone is not an asset failure.

## Artifacts outside Git

`/home/andyee/robot-research/ARES-Skilllab/runs/mentor_preview/` contains all seven requested progress documents, plus `usd_geometry_v3/ares_usd_geometry.png` and its source manifest. `/runs/alignment/usd_stage_probe_20261013.json` records the source stage transforms. Earlier incomplete preview versions remain separate.

## Pending resource-dependent work

Server audit/throughput, real Isaac render/physics, persistent cuRobo adapter, new Cartesian/folded candidates, paired trajectory comparison, real Ranker training, CASH auxiliary-head PPO integration, learning curves and strong-baseline improvements are not completed. The existing 4 GPU server is not identified by the local SSH aliases alone. The requested connection details are pending.

References: [STAP](https://arxiv.org/abs/2210.12250) supplies the whole-plan value perspective; [ViSkill (IROS 2023)](https://arxiv.org/abs/2307.16503) motivates downstream-value-informed skill termination. No improvement over these methods is claimed.
