"""Prevent controlled field outputs entering an enclosing Git worktree."""
from pathlib import Path
import subprocess


def require_private_location(path):
    path=Path(path).resolve()
    probe=path if path.is_dir() else path.parent
    while not probe.exists():probe=probe.parent
    result=subprocess.run(['git','-C',str(probe),'rev-parse','--show-toplevel'],capture_output=True)
    if result.returncode==0:
        raise ValueError('Controlled field data/figures must remain outside Git')
    return path
