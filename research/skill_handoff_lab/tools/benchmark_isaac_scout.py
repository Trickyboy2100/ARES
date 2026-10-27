"""Bounded unmodified Isaac Lab task runner; controllers are not new tasks.

Lift has no official success termination: report native reward/termination and
separate measured lift/goal diagnostics, never invent an official success label.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--task', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--num-envs', type=int, default=1)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--steps', type=int, default=200)
p.add_argument('--seed', type=int, default=27031)
p.add_argument('--controller', choices=['zero', 'random', 'lift_feedback', 'official_drawer'], default='zero')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
(a.output / 'source.py').write_bytes(Path(__file__).read_bytes())
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device=f'cuda:{a.gpu}', enable_cameras=True,
                  kit_args=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false')
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import imageio.v2 as imageio
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import subtract_frame_transforms

    torch.manual_seed(a.seed)
    cfg = parse_env_cfg(a.task, device=f'cuda:{a.gpu}', num_envs=a.num_envs)
    cfg.seed = a.seed
    cfg.viewer.resolution = (960, 720)
    cfg.viewer.eye = (1.4, 1.1, 0.9)
    cfg.viewer.lookat = (0.4, 0.0, 0.15)
    if 'Drawer' in a.task:
        cfg.viewer.eye = (-0.8, 1.2, 0.9)
        cfg.viewer.lookat = (0.4, 0.0, 0.35)
    cfg.viewer.origin_type = 'env'
    cfg.viewer.env_index = 0
    (a.output / 'native_config.json').write_text(json.dumps(cfg.to_dict(), default=str, indent=2))
    start = time.monotonic()
    env = gym.make(a.task, cfg=cfg, render_mode='rgb_array')
    u = env.unwrapped
    obs, info = env.reset(seed=a.seed)
    creation = time.monotonic() - start
    traces, returns = [], torch.zeros(a.num_envs, device=u.device)
    writer = imageio.get_writer(str(a.output / 'actual_execution.mp4'), fps=15, codec='libx264')
    initial = u.scene['robot'].data.joint_pos.detach().cpu().numpy().copy()
    state_machine = None
    if a.controller == 'official_drawer':
        import importlib.util
        import sys
        official = Path(sys.prefix).parent / 'IsaacLab-v2.3.0/scripts/environments/state_machine/open_cabinet_sm.py'
        source = official.read_text()
        (a.output / 'official_state_machine_source.py').write_text(source)
        # Preserve upstream Warp kernels/classes verbatim, omit only its GUI/CLI loop.
        core = a.output / 'official_state_machine_core.py'
        core.write_text(source[:source.index('"""')] + '\nimport torch\nimport warp as wp\nfrom collections.abc import Sequence\nwp.init()\n' +
                        source[source.index('class GripperState:'):source.index('\ndef main():')])
        spec = importlib.util.spec_from_file_location('official_drawer_core', core)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        state_machine = module.OpenDrawerSm(u.step_dt, a.num_envs, u.device)
    control_start = time.monotonic()
    for step in range(a.steps):
        with torch.inference_mode():
            action = torch.zeros(env.action_space.shape, device=u.device)
            if a.controller == 'random':
                action.uniform_(-0.2, 0.2)
            elif a.controller == 'official_drawer':
                ee, cabinet = u.scene['ee_frame'], u.scene['cabinet_frame']
                action = state_machine.compute(
                    torch.cat((ee.data.target_pos_w[:, 0] - u.scene.env_origins, ee.data.target_quat_w[:, 0]), dim=-1),
                    torch.cat((cabinet.data.target_pos_w[:, 0] - u.scene.env_origins, cabinet.data.target_quat_w[:, 0]), dim=-1))
            elif a.controller == 'lift_feedback':
                assert 'Lift' in a.task and 'IK-Abs' in a.task
                robot, obj = u.scene['robot'], u.scene['object']
                pos, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)
                elapsed = (step % int(cfg.episode_length_s / u.step_dt)) * u.step_dt
                target = pos.clone()
                # State feedback, then close, then lift toward the native goal.
                if elapsed < 0.8:
                    target[:, 2] += 0.12
                elif elapsed < 2.2:
                    target[:, 2] += 0.008
                else:
                    target = u.command_manager.get_command('object_pose')[:, :3].clone()
                action[:, :3] = target
                action[:, 3:7] = torch.tensor([0., 1., 0., 0.], device=u.device)
                action[:, 7] = 1. if elapsed < 1.6 else -1.
            obs, reward, terminated, truncated, info = env.step(action)
            returns += reward
            row = {'step': step + 1, 'reward': reward.cpu().tolist(),
                   'terminated': terminated.cpu().tolist(), 'truncated': truncated.cpu().tolist(),
                   'action': action.cpu().tolist(),
                   'joint_pos': u.scene['robot'].data.joint_pos.cpu().tolist()}
            if 'object' in u.scene.rigid_objects:
                row['object_state'] = u.scene['object'].data.root_state_w.cpu().tolist()
            row['other_articulations_qpos'] = {name: obj.data.joint_pos.cpu().tolist()
                                               for name, obj in u.scene.articulations.items() if name != 'robot'}
            row['native_termination_terms'] = {
                name: u.termination_manager.get_term(name).cpu().tolist()
                for name in u.termination_manager.active_terms}
            traces.append(row)
            if state_machine is not None and (terminated | truncated).any():
                state_machine.reset_idx((terminated | truncated).nonzero(as_tuple=False).squeeze(-1))
            if step % max(1, round(1 / (15 * u.step_dt))) == 0:
                frame = env.render()
                if frame is not None:
                    writer.append_data(frame)
                    if step == 0:
                        imageio.imwrite(a.output / 'initial.png', frame)
    writer.close()
    elapsed = time.monotonic() - control_start
    np.savez_compressed(a.output / 'initial_joint_state.npz', q=initial)
    (a.output / 'trace.json').write_text(json.dumps(traces))
    manifest = {'status': 'COMPLETE', 'track': 'OFFICIAL_ISAAC_LAB_TASK_UNMODIFIED',
                'task': a.task, 'num_envs': a.num_envs, 'seed': a.seed,
                'controller': a.controller, 'control_steps_per_env': a.steps,
                'environment_interactions': a.steps * a.num_envs,
                'physics_dt': cfg.sim.dt, 'decimation': cfg.decimation,
                'creation_seconds': creation, 'step_and_video_seconds': elapsed,
                'mean_return': returns.mean().item(), 'action_shape': list(env.action_space.shape),
                'native_termination_terms': u.termination_manager.active_terms,
                'state_timing': 'after ManagerBasedRLEnv.step; terminal environments may already be reset; native termination flags authoritative, not independent terminal audit',
                'task_success_claim': 'ONLY_NATIVE_TERMINATION_IF_PRESENT; lift has no success predicate',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    env.close()
except BaseException:
    (a.output / 'exception.txt').write_text(traceback.format_exc())
    traceback.print_exc()
    import os
    os._exit(1)
app.app.close(wait_for_replicator=False, skip_cleanup=True)
