"""Check current delivery hashes/counts after physical and document reviewers."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(workspace):
    package = workspace / 'repos/ARES/research/skill_handoff_lab/deliverables/client_sci_20261027'
    new = package / '08_ITERATIVE_RESEARCH'
    index = json.loads((new / 'metrics/COMPLETED_RUNS_INDEX.json').read_text())
    for row in index['runs']:
        run = workspace / 'runs/client_sci_closed_loop_20261027' / row['run']
        assert row['audit_status'] == 'COMPLETE', row['run']
        assert sha(run / 'manifest.json') == row['manifest_sha256']
        assert sha(run / 'task_source.py') == row['source_sha256']
        assert sha(run / 'run_config.json') == row['run_config_sha256']
    checks = {}
    for filename in ('VIDEO_QA.json', 'TRAINING_VIDEO_QA.json', 'metrics/MODEL_LOAD_QA.json'):
        data = json.loads((new / filename).read_text())
        assert data['status'] == 'PASS', filename
        checks[filename] = sha(new / filename)
    models = json.loads((new / 'metrics/MODEL_LOAD_QA.json').read_text())
    for model in models['models']:
        path = Path(model['run'])
        if not path.is_absolute():
            path = workspace / path
        assert sha(path / 'policy.zip') == model['checkpoint_sha256']
    docs = json.loads((package / 'DOCUMENT_QA.json').read_text())
    assert docs['status'] == 'PASS'
    for doc in docs['documents']:
        assert sha(package / doc['path']) == doc['sha256'], doc['path']
    for source in docs['source_copies']:
        assert sha(package / source['source']) == sha(package / source['paper_copy']) == source['sha256']
    result = {'status': 'PASS', 'completed_runs_checked': len(index['runs']),
              'model_files_checked': len(models['models']), 'pdf_files_checked': len(docs['documents']),
              'known_online_interactions_including_failed_recording': index['known_charged_totals_including_failures']['online_training_interactions'],
              'incomplete_recordings_retained': len(index['incomplete_or_software_failures']),
              'evidence_check_hashes': checks,
              'limitations': ['Artifact integrity does not establish scientific superiority.',
                             'Physical limitations remain those stated in per-run sampled validation.',
                             'Unknown failed-run physics costs remain explicitly unknown.']}
    (new / 'FINAL_DELIVERY_QA.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    verify(parser.parse_args().workspace.resolve())
