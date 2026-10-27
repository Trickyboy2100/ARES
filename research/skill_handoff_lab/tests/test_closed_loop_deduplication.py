"""Synthetic filesystem-only tests; never used as research results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

source = Path(__file__).parents[1] / 'tools' / 'deduplicate_closed_loop_evidence.py'
spec = importlib.util.spec_from_file_location('closed_loop_deduplication', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeduplicationTests(unittest.TestCase):
    def test_preserves_unique_and_all_identical_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs' / 'client_sci_closed_loop_20261027'
            for name in ['a', 'b', 'c']:
                directory = root / name
                directory.mkdir(parents=True)
                (directory / 'RUN_COMPLETE.json').write_text('{}')
                (directory / 'policy.zip').write_bytes((b'x' if name != 'c' else b'y') * 100001)
            report = root / 'report.json'
            module.main(root, report, True)
            self.assertEqual((root / 'a/policy.zip').stat().st_ino, (root / 'b/policy.zip').stat().st_ino)
            self.assertNotEqual((root / 'a/policy.zip').stat().st_ino, (root / 'c/policy.zip').stat().st_ino)
            self.assertEqual((root / 'c/policy.zip').read_bytes(), b'y' * 100001)
            self.assertEqual(json.loads(report.read_text())['paths_removed'], 0)

    def test_rejects_broad_root(self):
        with self.assertRaises(AssertionError):
            module.main(Path('/tmp'), Path('/tmp/not-created.json'), True)


if __name__ == '__main__':
    unittest.main()
