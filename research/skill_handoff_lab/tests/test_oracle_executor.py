"""Synthetic worker tests only; never usable for research training."""
import json,tempfile,unittest
from pathlib import Path
from research.skill_handoff_lab.bench.oracle_executor import OracleExecutor,failure_class

class ExecutorTests(unittest.TestCase):
    def test_failure_classes(self):
        self.assertEqual(failure_class('CUDA out of memory'),'OOM')
        self.assertEqual(failure_class('unsupported PTX driver'),'DRIVER_COMPATIBILITY')
    def test_success_without_validation_is_unknown(self):
        class FixtureWorker:
            def main(self):
                import sys
                Path(sys.argv[2]).write_text(json.dumps({'observed_result':'SUCCESS','scene_digest':'fixture'}))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'in.json';path.write_text(json.dumps(
                {'planning_only':True,'execution_allowed':False,'scene_digest':'fixture'}))
            executor=OracleExecutor.__new__(OracleExecutor);executor.worker=FixtureWorker()
            report=executor.execute(path,root/'job')
            self.assertIsNone(report['label']);self.assertIn('self_collision',report['missing_qualification'])
            self.assertEqual(report['status'],'PLANNER_PATH_REQUIRES_QUALIFICATION')
    def test_forbid_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'in.json';path.write_text('{"execution_allowed":true}')
            executor=OracleExecutor.__new__(OracleExecutor)
            with self.assertRaises(ValueError):executor.execute(path,root/'job')

if __name__=='__main__':unittest.main()
