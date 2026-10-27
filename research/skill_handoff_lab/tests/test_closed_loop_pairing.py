import importlib.util
from pathlib import Path
import unittest

source = Path(__file__).parents[1] / 'tools' / 'compare_closed_loop_cases.py'
spec = importlib.util.spec_from_file_location('closed_loop_pairing', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PairingTests(unittest.TestCase):
    def test_asynchronous_reset_does_not_inflate_episode_cost(self):
        episode = {'steps': 480, 'post_step_state': {'time_s': 200.0}}
        self.assertEqual(module.duration(episode), 8.0)

    def test_uses_explicit_episode_clock_when_available(self):
        episode = {'steps': 480, 'post_step_state': {'time_s': 200.0, 'episode_time_s': 9.0}}
        self.assertEqual(module.duration(episode), 9.0)


if __name__ == '__main__':
    unittest.main()
