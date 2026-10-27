"""Run official ManiSkill tasks with bounded random or official planning control.

No task/robot/dynamics/success modifications. Planning's internal motion can
exceed official horizons: stop exactly at the native budget and keep failure.
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
p.add_argument('--episodes', type=int, default=4)
p.add_argument('--seed', type=int, default=27031)
p.add_argument('--controller', choices=['random', 'zero', 'official_planner', 'planner_gpu_pose', 'ppo', 'stack_feedback', 'cartesian_ppo', 'residual_ppo'], default='random')
p.add_argument('--checkpoint', type=Path)
p.add_argument('--ppo-source', type=Path)
p.add_argument('--force-zero-policy', action='store_true', help='Causal residual ablation; keep nominal controller unchanged')
p.add_argument('--upstream-sha', required=True)
p.add_argument('--diagnostic-budget', type=int, default=0,
               help='Continue native physical dynamics after time limit ONLY as a separately labeled planner diagnostic')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
(a.output / 'source.py').write_bytes(Path(__file__).read_bytes())

try:
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import mani_skill.envs
    from mani_skill.utils import gym_utils

    def serial(x):
        if isinstance(x, dict):
            return {k: serial(v) for k, v in x.items()}
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return x.tolist() if isinstance(x, np.ndarray) else x.detach().cpu().tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return x

    class BudgetEnd(Exception):
        pass

    class Audited(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.horizon = gym_utils.find_max_episode_steps_value(env)
            self.rows = []
            self.writer = None
            self.initial = None

        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            self.rows = []
            self.initial = serial(self.unwrapped.get_state_dict())
            return result

        def step(self, action):
            if len(self.rows) >= (a.diagnostic_budget or self.horizon):
                raise BudgetEnd()
            result = self.env.step(action)
            self.rows.append({'step': len(self.rows) + 1, 'action': serial(action),
                              'reward': serial(result[1]), 'info': serial(result[-1]),
                              'terminated': serial(result[2]), 'truncated': serial(result[3]),
                              'state': serial(self.unwrapped.get_state_dict())})
            if self.writer:
                frame = self.env.render().detach().cpu().numpy()
                if frame.ndim == 4:
                    frame = np.concatenate(list(frame), axis=1)
                self.writer.append_data(frame)
            return result

    planning = a.controller in ['official_planner', 'planner_gpu_pose']
    assert not planning or a.num_envs == 1
    control = 'pd_joint_pos' if planning else 'pd_joint_delta_pos'
    if a.controller in ['cartesian_ppo', 'residual_ppo']:
        from benchmark_stack_residual import install_wrapper
        install_wrapper('residual' if a.controller == 'residual_ppo' else 'direct')
        control = 'pd_ee_delta_pos'
    if a.controller == 'stack_feedback':
        assert a.task == 'StackCube-v1'
        control = 'pd_ee_delta_pos'
    torch.manual_seed(a.seed)
    start = time.monotonic()
    env = Audited(gym.make(a.task, num_envs=a.num_envs, obs_mode='state',
                          control_mode=control, render_mode='rgb_array',
                          sim_backend='physx_cuda', render_backend='gpu',
                          reconfiguration_freq=1))
    rows = []
    created = time.monotonic() - start
    policy = None
    if a.controller in ['ppo', 'cartesian_ppo', 'residual_ppo']:
        import importlib.util
        import sys
        assert a.checkpoint and a.ppo_source
        spec = importlib.util.spec_from_file_location('upstream_ppo_eval', a.ppo_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        policy = module.Agent(env.env if a.controller != 'ppo' else env.unwrapped).to(env.unwrapped.device)
        policy.load_state_dict(torch.load(a.checkpoint, map_location=env.unwrapped.device, weights_only=True))
        policy.eval()
    obb_audit = []
    if planning:
        from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
        solve = MP_SOLUTIONS[a.task]
        if a.controller == 'planner_gpu_pose':
            import sys
            from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb
            def gpu_pose_obb(actor, to_world_frame=True, vis=False):
                old = get_actor_obb(actor, to_world_frame=to_world_frame, vis=False)
                box = get_actor_obb(actor, to_world_frame=False, vis=False)
                if to_world_frame:
                    import trimesh
                    transform = actor.pose.to_transformation_matrix()[0].detach().cpu().numpy().astype(float)
                    box = trimesh.primitives.Box(extents=np.array(box.primitive.extents, copy=True),
                                                 transform=transform @ np.array(box.primitive.transform))
                obb_audit.append({'native_cpu_component_obb_center': old.primitive.transform[:3, 3].tolist(),
                                  'current_gpu_pose': actor.pose.raw_pose.detach().cpu().tolist(),
                                  'gpu_corrected_obb_center': box.primitive.transform[:3, 3].tolist()})
                return box
            # Fix only the planner's geometric observation adapter, never task state.
            setattr(sys.modules[solve.__module__], 'get_actor_obb', gpu_pose_obb)
    for episode in range(a.episodes):
        seed = a.seed + episode * a.num_envs
        start = time.monotonic()
        error = None
        env.writer = imageio.get_writer(str(a.output / f'episode_{episode:03d}.mp4'),
                                        fps=env.unwrapped.control_freq, codec='libx264')
        try:
            if planning:
                result = solve(env, seed=seed, debug=False, vis=False)
                if isinstance(result, int) and result == -1:
                    error = 'PLANNER_REPORTED_FAILURE'
            else:
                obs, _ = env.reset(seed=seed)
                phase = torch.zeros(a.num_envs, dtype=torch.long, device=env.unwrapped.device)
                dwell = torch.zeros_like(phase)
                pickup = env.unwrapped.cubeA.pose.p.clone() if a.controller == 'stack_feedback' else None
                env.action_space.seed(seed)
                for _ in range(env.horizon):
                    action = env.action_space.sample()
                    if a.controller == 'zero':
                        action *= 0
                    elif policy is not None:
                        with torch.inference_mode():
                            action = policy.get_action(obs, deterministic=True)
                        if a.force_zero_policy:
                            assert a.controller == 'residual_ppo'
                            action = torch.zeros_like(action)
                        action = torch.maximum(torch.minimum(action,
                            torch.as_tensor(env.action_space.high, device=env.unwrapped.device)),
                            torch.as_tensor(env.action_space.low, device=env.unwrapped.device))
                    elif a.controller == 'stack_feedback':
                        u = env.unwrapped
                        tcp, cube, base = u.agent.tcp.pose.p, u.cubeA.pose.p, u.cubeB.pose.p
                        target = cube.clone()
                        target[:, 2] += .08
                        mask = (phase == 1) | (phase == 2)
                        target[mask] = pickup[mask]
                        target[mask, 2] += .003
                        mask = phase == 3
                        target[mask] = pickup[mask]
                        target[mask, 2] = .14
                        mask = phase >= 4
                        target[mask] = base[mask]
                        target[mask, 2] = .14
                        mask = (phase == 5) | (phase == 6)
                        target[mask, 2] = base[mask, 2] + .043
                        pose_error = target - tcp
                        action = torch.zeros((a.num_envs, 4), device=u.device)
                        action[:, :3] = (pose_error / .1).clamp(-.6, .6)
                        action[:, 3] = torch.where((phase >= 2) & (phase <= 5), -1., 1.)
                        # These are current observable geometry/dwell rules, not future labels.
                        near = torch.linalg.vector_norm(pose_error, dim=1) < .012
                        advance = ((phase == 0) & near) | ((phase == 1) & near) | ((phase == 2) & (dwell >= 6))
                        advance |= ((phase == 3) & near & (cube[:, 2] > .07)) | ((phase == 4) & near)
                        advance |= ((phase == 5) & near) | ((phase == 6) & (dwell >= 6))
                        dwell += 1
                        phase[advance] = (phase[advance] + 1).clamp_max(7)
                        dwell[advance] = 0
                    obs, _, _, _, _ = env.step(action)
                    if a.controller == 'stack_feedback':
                        env.rows[-1]['feedback_phase'] = phase.cpu().tolist()
        except BudgetEnd:
            error = 'NATIVE_HORIZON_EXHAUSTED'
        except Exception:
            error = traceback.format_exc()
        env.writer.close()
        env.writer = None
        native = serial(env.unwrapped.evaluate())
        row = {'episode': episode, 'seed': seed, 'steps': len(env.rows),
               'wall_seconds': time.monotonic() - start, 'native_final': native,
               'native_success_ever': [any(bool(r['info']['success'][i]) for r in env.rows[:env.horizon])
                                       for i in range(a.num_envs)],
               'diagnostic_success_ever': [any(bool(r['info']['success'][i]) for r in env.rows)
                                          for i in range(a.num_envs)], 'error': error}
        (a.output / f'episode_{episode:03d}.json').write_text(json.dumps(
            {'summary': row, 'initial_state': env.initial, 'transitions': env.rows}))
        rows.append(row)
        print(json.dumps(row), flush=True)
    software_errors = sum(e['steps'] == 0 or e['error'] not in (None, 'NATIVE_HORIZON_EXHAUSTED', 'PLANNER_REPORTED_FAILURE') for e in rows)
    manifest = {'status': 'COMPLETE' if not software_errors else 'SOFTWARE_ERROR_PRESENT',
                'software_error_episodes': software_errors, 'task': a.task, 'controller': a.controller,
                'num_envs': a.num_envs, 'native_horizon': env.horizon,
                'action_shape': list(env.action_space.shape),
                'observation_shape': list(env.observation_space.shape),
                'control_mode': control, 'creation_seconds': created, 'episodes': rows,
                'track': 'OFFICIAL_MANISKILL_TASK_DEFAULT_ROBOT_UNMODIFIED',
                'diagnostic_budget': a.diagnostic_budget,
                'evaluation_scope': 'OUT_OF_HORIZON_PLANNER_DIAGNOSTIC_NOT_OFFICIAL_SCORE' if a.diagnostic_budget else 'NATIVE_HORIZON',
                'simulator': 'ManiSkill3 / SAPIEN (physx_cuda backend), NOT Isaac Lab',
                'upstream_commit': a.upstream_sha,
                'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,
                'force_zero_policy': a.force_zero_policy,
                'planner_obb_pose_audit': obb_audit,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'license_note': 'Apache-2.0 code; upstream asset CC-BY-NC notice requires separate commercial-use review; no asset redistribution'}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    env.close()
except BaseException:
    (a.output / 'exception.txt').write_text(traceback.format_exc())
    raise
