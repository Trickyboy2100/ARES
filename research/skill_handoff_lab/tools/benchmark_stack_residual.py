"""Matched Cartesian/residual PPO pilots on unchanged native StackCube physics.

This is an ordinary residual baseline, NOT a claimed new algorithm. Both modes
observe the same state, phase and nominal command. The native reward, 50-step
horizon and success predicate are untouched. Only the controller interface is
wrapped. Phase is persistent controller state, never a future success label.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def install_wrapper(mode):
    import gymnasium as gym
    import numpy as np
    import torch
    import mani_skill.envs

    original_make = gym.make

    class StackController(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            u = self.unwrapped
            self.phase = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
            self.dwell = torch.zeros_like(self.phase)
            self.pickup = u.cubeA.pose.p.clone()
            n = env.observation_space.shape[-1] + 12
            self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
            self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, u.num_envs)
            self.single_action_space = u.single_action_space

        def nominal(self):
            u = self.unwrapped
            tcp, cube, base = u.agent.tcp.pose.p, u.cubeA.pose.p, u.cubeB.pose.p
            target = cube.clone()
            target[:, 2] += .08
            mask = (self.phase == 1) | (self.phase == 2)
            target[mask] = self.pickup[mask]
            target[mask, 2] += .003
            mask = self.phase == 3
            target[mask] = self.pickup[mask]
            target[mask, 2] = .14
            mask = self.phase >= 4
            target[mask] = base[mask]
            target[mask, 2] = .14
            mask = (self.phase == 5) | (self.phase == 6)
            target[mask, 2] = base[mask, 2] + .043
            error = target - tcp
            command = torch.zeros((u.num_envs, 4), device=u.device)
            command[:, :3] = (error / .1).clamp(-.6, .6)
            command[:, 3] = torch.where((self.phase >= 2) & (self.phase <= 5), -1., 1.)
            return command, error

        def observation(self, obs):
            command, _ = self.nominal()
            return torch.cat((obs, torch.nn.functional.one_hot(self.phase, 8).float(), command), dim=-1)

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            idx = (kwargs.get('options') or {}).get('env_idx', slice(None))
            self.phase[idx] = 0
            self.dwell[idx] = 0
            self.pickup[idx] = self.unwrapped.cubeA.pose.p[idx]
            return self.observation(obs), info

        def step(self, action):
            command, error = self.nominal()
            # Same legal Cartesian/gripper bounds; residual scaling is explicit.
            raw = command + .15 * action if mode == 'residual' else action
            executed = raw.clamp(-1., 1.)
            near = torch.linalg.vector_norm(error, dim=1) < .012
            p, d = self.phase, self.dwell
            advance = ((p == 0) & near) | ((p == 1) & near) | ((p == 2) & (d >= 6))
            advance |= ((p == 3) & near & (self.unwrapped.cubeA.pose.p[:, 2] > .07)) | ((p == 4) & near)
            advance |= ((p == 5) & near) | ((p == 6) & (d >= 6))
            self.dwell += 1
            self.phase[advance] = (self.phase[advance] + 1).clamp_max(7)
            self.dwell[advance] = 0
            obs, reward, terminated, truncated, info = self.env.step(executed)
            info['controller_action_audit'] = {'policy': action.detach().clone(),
                'nominal': command.detach().clone(), 'preclip': raw.detach().clone(),
                'executed': executed.detach().clone(), 'phase': self.phase.clone()}
            return self.observation(obs), reward, terminated, truncated, info

    def make(task, **kwargs):
        assert task == 'StackCube-v1'
        kwargs['control_mode'] = 'pd_ee_delta_pos'
        return StackController(original_make(task, **kwargs))
    gym.make = make
    return StackController


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--mode', choices=['direct', 'residual'], required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--steps', type=int, default=256000)
    parser.add_argument('--seed', type=int, default=31)
    args = parser.parse_args()
    assert 0 < args.steps <= 1024000 and args.steps % 3200 == 0
    out = Path('runs') / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out / 'controller_source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'upstream_ppo_source.py').write_bytes(args.upstream.read_bytes())
    (out / 'pilot_protocol.json').write_text(json.dumps({
        'mode': args.mode, 'seed': args.seed, 'steps': args.steps,
        'scope': 'NATIVE_TASK_CUSTOM_CONTROLLER_ORDINARY_PPO_BASELINE_NOT_NOVEL_METHOD',
        'upstream_sha256': hashlib.sha256(args.upstream.read_bytes()).hexdigest(),
        'reward_physics_horizon_success': 'UNCHANGED',
        'action': 'pd_ee_delta_pos; residual=clip(nominal + 0.15*policy,-1,1)',
        'observation': 'native state + controller phase onehot + nominal command; shared by both modes'
    }, indent=2))
    install_wrapper(args.mode)
    sys.argv = [str(args.upstream), '--env-id', 'StackCube-v1', '--exp-name', args.name,
                '--seed', str(args.seed), '--num-envs', '64', '--num-eval-envs', '4',
                '--total-timesteps', str(args.steps), '--num-steps', '50', '--num-eval-steps', '50',
                '--num-minibatches', '8', '--eval-freq', '10', '--capture-video', '--save-model',
                '--control-mode', 'pd_ee_delta_pos']
    runpy.run_path(str(args.upstream), run_name='__main__')
