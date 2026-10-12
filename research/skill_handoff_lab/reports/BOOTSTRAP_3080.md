# RTX 3080 bootstrap and reconnaissance report

Date: 2026-10-13 (Asia/Shanghai)

Scope: preparation and read-only reconnaissance through P0 plus a narrow P1 backend check. No robot I/O, Docker container launch, Isaac Sim launch, system package change, driver change, or production-branch modification was performed.

## Workspace and source revisions

- Lab root: `/home/andyee/robot-research/ARES-Skilllab`
- ARES writable research checkout: `research/skill-handoff-simbase-20261013` at `7b5f52802792600d1c33321de9a1a0e3ff188d82`
- ARES-R read-only reference checkout: `research/skill-handoff-smoke-20261012` at `9de17d2abe2775fbcbcdecc04db21c2b9caf1aca`
- The requested `envs`, `data`, `runs`, `logs`, and `reports` workspace directories were created.
- L0 virtual environment: Python 3.13.5, NumPy 2.5.4, SciPy 1.18.1.

## Executed checks

### Host audit

Evidence: `/home/andyee/robot-research/ARES-Skilllab/runs/host/host_audit.json`

- Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic, Intel i5-13400F, approximately 31 GiB RAM.
- RTX 3080 PCI device and NVIDIA kernel modules are present.
- Docker 29.1.3 and the NVIDIA container runtime are present.
- Existing images include `nvcr.io/nvidia/isaac-sim:4.5.0` and `nvcr.io/nvidia/isaac-sim:5.0.0`.
- A local Isaac Sim tree exists at `/home/andyee/isaacsim`; its `VERSION` reports `5.1.0-rc.19+release.26219.9c81211b.gl`.
- The legacy simulation directory exists and contains the reported MuJoCo and Isaac scripts.
- Home filesystem has about 98 GiB available but is already 89% used.

### Static assets and contracts

Evidence: `/home/andyee/robot-research/ARES-Skilllab/runs/alignment/static_alignment.json`

- Four offline alignment unit tests passed in the isolated L0 environment.
- All 16 pinned URDF/mesh/config SHA-256 entries match ARES-R's manifest.
- All six ARES URDF joint intervals have non-empty intersections with the ARES-R site soft limits after the 0.15 rad margin.
- BODY arm-base signs match the static contract: left Y = +0.20 m, right Y = -0.20 m.
- The canonical cuRobo YAML still contains the old absolute `/home/andyee/Developer/PG-JY/...` paths. It was not changed.

### FK parity

Evidence: `/home/andyee/robot-research/ARES-Skilllab/runs/alignment/fk_parity.json`

- `FK_IMPLEMENTATION_PARITY_PASS` over 48 arm/pose samples.
- Maximum flange position error: `2.5438405243138006e-16 m`.
- Maximum flange orientation error: `2.4148365394514667e-06 deg`.
- This validates the two pinned URDF flange FK implementations only; it does not validate the current physical TCP.

### Existing backend reconnaissance

- The legacy MuJoCo environment (`jnl263_mujoco`, MuJoCo 3.6.0) successfully compiled its generated model and ran 20 headless physics steps: `nq=26`, `nv=24`, `ngeom=18`, simulated time 0.04 s.
- That legacy model is a simplified primitive-geometry model with generic `[-2.8, 2.8]` joint ranges, not the pinned ARES URDF/mesh/collision model. It is evidence that MuJoCo runs, not evidence of robot-model alignment.
- The current base Python can import MuJoCo 3.9.0 and cuRobo, but `torch.cuda.is_available()` is false while the NVIDIA driver is unhealthy. No cuRobo GPU planning result was attempted or claimed.
- SimForge path checking succeeds and the bundled `main.usd` is present as a 16 MiB USD crate. Its WORLD-to-BODY transform was not inspected because the GPU/Isaac runtime gate is not currently healthy.
- Five ARES-R research-smoke unit tests passed. These validate evidence-gating behavior only; no Smoke-1 planner outcomes exist yet.

## Blocking and unresolved issues

1. **NVIDIA driver/library mismatch.** `nvidia-smi` returns code 18: kernel module `580.173.02` is loaded while the user-space NVML library is `580.178`. Isaac and cuRobo GPU execution must remain blocked until the host is returned to a consistent driver state and `nvidia-smi` succeeds. No driver action was taken.
2. **Isaac backend not launched.** Because of the driver mismatch, neither the existing 4.5 Docker image nor local 5.1 tree was started. GPU compatibility, VRAM usage, USD loading, and headless stability remain NOT_TESTED.
3. **USD/BODY alignment not verified.** The ARES USD WORLD convention has not yet been related to the ARES-R BODY contract.
4. **Live TCP and current site state absent.** The current controller `T_link6_tcp`, October soft limits/config delta, camera transform revision, and fresh scene snapshot require a new read-only site export. Static Git values must not be called current live alignment.
5. **Legacy models are not equivalent.** The old MuJoCo/Isaac scripts use simplified shoulder placement and geometry; they cannot be used as the scientific collision oracle without rebuilding against the pinned assets and BODY contract.
6. **Scientific fallbacks are unsafe as labels.** The ARES tray demo can use a single-step joint jump after cuRobo failure and FixedJoint/no-contact grasp fallbacks. These must be disabled or explicitly rejected for Smoke-1 labels.
7. **Storage and memory headroom are limited.** The filesystem is 89% used. During the audit, only about 8.5 GiB RAM was available and swap was almost full. This reinforces the low-memory, no-GUI L0/L1-first plan.

## Current acceptance matrix

```text
HOST_ENV_AUDIT                 FAIL        NVIDIA NVML driver/library mismatch; inventory otherwise captured
ASSET_PARITY                   PASS        16/16 pinned files match; site/URDF intersections valid
FK_IMPLEMENTATION_PARITY       PASS        48 samples; flange FK only
SIM_USD_BODY_ALIGNMENT         NOT_TESTED  Isaac stage audit blocked by GPU driver state
CONTROLLER_TCP_CONTRACT        NOT_TESTED  current read-only controller export absent
SCENARIO_GENERATION            NOT_TESTED
CUROBO_ORACLE                  NOT_TESTED  CUDA unavailable in current host state
SMOKE1_CANDIDATE_SENSITIVITY   NOT_TESTED
SMOKE1_HEURISTIC_HEADROOM      NOT_TESTED
READY_TO_TRY_RL                FAIL        required planning-only smoke has not been run
```

## Recommended next safe step

After the user restores a consistent NVIDIA driver state (often a reboot after a completed driver update, but the actual host state should be checked rather than assumed), rerun `nvidia-smi` and a minimal CUDA availability probe. Only then run an inspect-only Isaac 4.5 headless smoke. In parallel, L0 work can continue by defining the BODY-frame scene contract and a pinned-URDF MuJoCo/static fixture without using the legacy simplified model as ground truth.
