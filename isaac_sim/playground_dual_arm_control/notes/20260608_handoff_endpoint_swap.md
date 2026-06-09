# Handoff Endpoint Swap

Date: 2026-06-08 CST

Change:

- Kept the handoff spacing at `0.155 m`.
- Kept the handoff separation axis on world X.
- Swapped only the chest handoff endpoints:
  - left handoff pad: `[0.0775, 0.65, 1.2]`
  - right handoff pad: `[-0.0775, 0.65, 1.2]`
  - `right - left = [-0.155, 0, 0]`
- Left pick/lift contact side remains unchanged so the left-pad-to-tray calibration does not jump before handoff.

Accepted errors:

- `right_forward_ready`: position error `0.001332850 m`, forward-axis error `4.674538 deg`.
- `right_forward_extend`: position error `0.000187671 m`, forward-axis error `0.678772 deg`.
- Expected both-attachment tray agreement at handoff: `~7.5e-10 m`.

Artifacts:

- `runtime/tray_handoff_curobo_trajectory.json`
- `reports/tray_handoff_curobo_plan.json`
- Probe source promoted to official output:
  - `runtime/tray_handoff_swap_probe_trajectory.json`
  - `reports/tray_handoff_swap_probe_plan.json`
