# Gemini2 Wrist Camera Simulation

Created: 2026-06-09 CST

## Source Policy

The old USD scene:

```text
/home/andyee/Documents/isc/202606060821.usd
```

is used as a read-only reference only for the manually placed Gemini2 transform
relative to each arm's `Link_6`. No other geometry, object placement, physics
state, or scene content is imported from that file.

The referenced old Gemini2 asset path points to a remote Isaac asset and is not
portable in this workspace. The current clean task scene therefore uses local
lightweight USD `Camera` prims instead.

## Installed Paths

```text
/World/robot/jaka_minicobo_left/Link_6/Gemini2Sim
/World/robot/jaka_minicobo_left/Link_6/Gemini2Sim/RgbCamera
/World/robot/jaka_minicobo_left/Link_6/Gemini2Sim/DepthCamera

/World/robot/jaka_minicobo_right/Link_6/Gemini2Sim
/World/robot/jaka_minicobo_right/Link_6/Gemini2Sim/RgbCamera
/World/robot/jaka_minicobo_right/Link_6/Gemini2Sim/DepthCamera
```

The old remote-reference prims under `Link_6/CAM_Mount/Gemini2` are left
inactive.

## Mount Transform

Both arms use the same recovered transform relative to `Link_6`:

```text
translation ~= [-0.104999247, -0.000000001, 0.021] m
rotation ~= Link_6-relative matrix from the old manual Gemini2 placement
```

The full matrix is recorded in:

```text
reports/gemini2_wrist_camera_install_report.json
reports/curobo_task_scene_fix_report.json
```

## Simulated Gemini2 Parameters

RGB camera:

```text
resolution = 1280 x 720
HFOV = 86 deg
VFOV = 55 deg
near/far = 0.15 m / 10.0 m
Replicator annotator = rgb
```

Depth camera:

```text
resolution = 640 x 400
HFOV = 91 deg
VFOV = 66 deg
near/far = 0.15 m / 10.0 m
Replicator annotator = distance_to_camera
```

Depth noise metadata is stored on the depth camera prim:

```text
sigma_m = 0.003
proportional_coeff = 0.0005
invalid_ratio = 0.002
```

The metadata records the intended real-camera noise model. The actual RGB-D
capture script should apply this noise model after reading Replicator depth.

## Corrections Applied (2026-06-09)

Two bugs were found and fixed in `install_gemini2_wrist_cameras.py`:

1. **Optical axis reversed** — `DEFAULT_FORWARD_FLIP_DEG` was `0.0`; the reference
   mount transform had the camera's USD −Z pointing into the cam mount body instead
   of toward the workspace. Fixed to `180.0` (rotates around camera local Y).

2. **Depth-of-field blur** — `FStopAttr = 2.8` with `FocusDistanceAttr = 1.0 m`
   enabled DoF simulation; the wrist camera view spans 0.15–5 m, leaving almost
   nothing in focus. Fixed to `FStopAttr = 0.0` (pinhole model, no blur).

Diagnosis artifact: `scripts/gui_camera_direction_check.py` — adds runtime
`OpticalAxisArrow` prims at each `RgbCamera` along its local −Z for GUI inspection.
See `milestones/20260609_gemini2_camera_orientation_fix/` for full before/after
evidence and the fixed script snapshots.

## Rebuild

Install or refresh only the cameras:

```bash
python3 scripts/install_gemini2_wrist_cameras.py
```

Rebuild the clean task scene and reinstall the cameras automatically:

```bash
python3 scripts/build_curobo_task_scene.py
```

Disable camera installation during scene rebuild:

```bash
python3 scripts/build_curobo_task_scene.py --no-gemini2-cameras
```

## Downstream Use

For ArUco and RGB-D pick/place modules:

1. Use `RgbCamera` paths for Replicator `rgb`.
2. Use `DepthCamera` paths for Replicator `distance_to_camera`.
3. Convert USD camera coordinates to OpenCV coordinates explicitly.
   USD cameras look along local `-Z`; OpenCV depth convention uses `+Z`.
4. Use the exported `relative_to_link6_matrix4x4` plus current FK to compute
   wrist-camera extrinsics for marker pose and grasp-frame updates.
