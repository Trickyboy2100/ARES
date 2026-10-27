"""Read-only version and isolation receipt. No device network commands."""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import torch,numpy,gymnasium,sapien,mani_skill

out=Path(sys.argv[1]);workspace=Path(sys.argv[2]);upstream=workspace/'external/benchmark_scout_20261027/ManiSkill'
routes=Path('/proc/net/route').read_text().splitlines()[1:];serial=[str(p) for glob in ['ttyUSB*','ttyACM*','hidraw*'] for p in Path('/dev').glob(glob)]
assert not routes and not serial and torch.cuda.device_count()==1
readonly=[str(workspace/'repos/ARES')]+[line[9:] for line in subprocess.check_output(['git','-C',str(workspace/'repos/ARES-R'),'worktree','list','--porcelain'],text=True).splitlines() if line.startswith('worktree ')]
assert all(not os.access(p,os.W_OK) for p in readonly)
record=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),visible_gpus=torch.cuda.device_count(),numpy=numpy.__version__,gymnasium=gymnasium.__version__,sapien=sapien.__version__,effective_mani_skill_source=str(Path(mani_skill.__file__).resolve()),upstream_commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),task_source_sha256={n:hashlib.sha256((upstream/'mani_skill/envs/tasks/tabletop'/n).read_bytes()).hexdigest() for n in ['stack_cube.py','peg_insertion_side.py','stack_pyramid.py']},network_routes=routes,serial_devices=serial,ARES_R_mount_writable=False,readonly_repo_roots=readonly,CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),concurrency='Some independent physical runs shared GPU0; wall time is not an algorithm speed comparison',result='PASS')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
