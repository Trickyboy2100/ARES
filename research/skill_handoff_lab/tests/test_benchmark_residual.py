"""Interface-only fixtures, never training data or success evidence."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

path = Path(__file__).resolve().parents[1] / 'tools/benchmark_stack_residual.py'
spec = importlib.util.spec_from_file_location('benchmark_residual_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Fixture(gym.Env):
    def __init__(self):
        self.num_envs, self.device = 2, 'cpu'
        self.single_action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, 2)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2, 6), np.float32)
        self.cubeA = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0.,0.,.02]]).repeat(2,1)))
        self.cubeB = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[.1,0.,.02]]).repeat(2,1)))
        self.agent = SimpleNamespace(tcp=SimpleNamespace(pose=SimpleNamespace(p=torch.zeros(2,3))))
    def reset(self, **kwargs): return torch.zeros(2,6), {}
    def step(self, action):
        self.executed = action.clone()
        return torch.zeros(2,6), torch.zeros(2), torch.zeros(2,dtype=torch.bool), torch.zeros(2,dtype=torch.bool), {}


class ResidualTests(unittest.TestCase):
    def wrapped(self, mode):
        env = Fixture()
        package = ModuleType('mani_skill'); package.envs = ModuleType('mani_skill.envs')
        with patch.dict(sys.modules, {'mani_skill':package, 'mani_skill.envs':package.envs}), patch.object(gym,'make',return_value=env):
            cls=module.install_wrapper(mode)
            return cls(env)
    def test_zero_residual_is_nominal_and_does_not_write_state(self):
        env=self.wrapped('residual'); before=env.unwrapped.cubeA.pose.p.clone()
        obs,_=env.reset(); self.assertEqual(obs.shape,(2,18))
        expected,_=env.nominal()
        result=env.step(torch.zeros(2,4))
        torch.testing.assert_close(env.unwrapped.executed,expected)
        torch.testing.assert_close(env.unwrapped.cubeA.pose.p,before)
        self.assertIn('controller_action_audit',result[-1])
    def test_partial_reset_does_not_reset_other_environment(self):
        env=self.wrapped('residual'); env.phase[:]=5; env.dwell[:]=4
        env.reset(options={'env_idx':torch.tensor([0])})
        self.assertEqual(env.phase.tolist(),[0,5]); self.assertEqual(env.dwell.tolist(),[0,4])
    def test_bounds_and_matching_observation(self):
        direct=self.wrapped('direct'); residual=self.wrapped('residual')
        torch.testing.assert_close(direct.reset()[0],residual.reset()[0])
        for env in (direct,residual):
            env.step(torch.tensor([[100.,-100.,100.,-100.]]).repeat(2,1))
            self.assertTrue((env.unwrapped.executed.abs()<=1).all())


if __name__ == '__main__': unittest.main()
