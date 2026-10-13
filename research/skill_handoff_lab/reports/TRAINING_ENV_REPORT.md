# Training environment report

> 2026-10-13 update: see `MULTIGPU_CASH_PROGRESS.md`. Gymnasium/SB3-contrib are
> now installed in a separate CASH environment, the real Gymnasium adapter and
> cache training entry points are implemented, and a 16-step MaskablePPO software
> fixture test passes. The historical report below describes the earlier round.

## Implemented

- Revisioned multi-stage handoff contracts.
- Seeded backend interface and Gymnasium-shaped `reset`/`step` API without a
  hard Gymnasium dependency.
- Explicit action masks and masked actor-critic update.
- State/remaining-goal/successor-constraint inputs.
- Strict evidence gate for reward and supervised labels.
- Deterministic checkpoint save/load format and gradient-update unit test.
- cuRobo job adapter bound to scene and attached-object revisions.
- Canonical runtime direct-preplace contract; no single-frame fallback.

All 10 unit tests pass. The actor-critic forward/update smoke changes real
parameters on CPU.

## Not executed

- No random-policy Isaac episode.
- No cuRobo plan in this round.
- No Ranker fit on physical outcomes.
- No MaskablePPO or CASH training curve.
- No real-robot execution.

Reason: the host loads NVIDIA kernel module 580.173.02 while installed
DKMS/userspace are 580.178.04. `nvidia-smi` fails; PyTorch reports CUDA error
804. Running Isaac/cuRobo training under this state would not be a valid
experiment. Gymnasium, Stable-Baselines3 and SB3-contrib are also absent; no
dependency installation was authorized or performed.

## GPU repair/qualification

Kernel 6.8.0-138 has a 580.178.04 DKMS build installed, while the old 580.173.02
module remains loaded. The minimum likely repair is a controlled reboot into
the already built module, followed by version/NVML/CUDA checks. This report does
not authorize or perform that reboot. Driver install/uninstall or module reload
is not recommended as an experiment-side workaround.

After recovery: run one cuRobo-only request, one no-fallback Isaac planned
segment, then one three-stage random CASH episode. Only then start the bounded
oracle batch and learning comparison.
