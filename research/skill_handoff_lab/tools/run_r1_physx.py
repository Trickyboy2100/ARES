"""190 personal-directory bounded PhysX launcher, own process groups only."""
import argparse,json,os,subprocess,signal,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--script',type=Path,required=True)
p.add_argument('--asset',type=Path,required=True);p.add_argument('--name',required=True)
p.add_argument('--num-envs',type=int,default=1);p.add_argument('--steps',type=int,default=256)
p.add_argument('--replay',type=Path);p.add_argument('--timeout',type=int,default=360)
p.add_argument('--skip-capture',action='store_true');p.add_argument('--hold-correction',action='store_true')
p.add_argument('--industrial',action='store_true');p.add_argument('--controller',default='ppo');p.add_argument('--eval-episodes',type=int,default=0)
p.add_argument('--load-policy',type=Path);p.add_argument('--eval-case-file',type=Path);p.add_argument('--eval-case-ids',default='');a=p.parse_args()
out=a.root/'runs/r1_20261014';out.mkdir(exist_ok=True,parents=True)
if (out/(a.name+'.log')).exists():raise FileExistsError('never overwrite a previous experiment')
env=os.environ.copy()
for key in ['CUDA_VISIBLE_DEVICES','ALL_PROXY','all_proxy','HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy']:env.pop(key,None)
env.update(PYTHONUNBUFFERED='1',OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='2')
argv=[str(a.root/'envs/isaac51-check/bin/python'),'-u',str(a.script),'--asset',str(a.asset),'--output',str(out/a.name),
    '--num-envs',str(a.num_envs),'--steps',str(a.steps),'--gpu','1']
if a.replay:argv+=['--replay',str(a.replay)]
if a.skip_capture:argv+=['--skip-capture']
if a.hold_correction:argv+=['--hold-correction']
if a.industrial:argv+=['--industrial','--controller',a.controller]
if a.eval_episodes:argv+=['--eval-episodes',str(a.eval_episodes)]
if a.load_policy:argv+=['--load-policy',str(a.load_policy)]
if a.eval_case_file:argv+=['--eval-case-file',str(a.eval_case_file),'--eval-case-ids',a.eval_case_ids]
started=time.monotonic();timed_out=False
with (out/(a.name+'.log')).open('w') as log:
    child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=child.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
manifest=out/a.name/'manifest.json'
result={'argv':argv,'pid':child.pid,'returncode':code,'timeout':timed_out,'wall_s':time.monotonic()-started,
    'valid_manifest_exists':manifest.is_file(),'program_sha256':hashlib.sha256(a.script.read_bytes()).hexdigest()}
(out/(a.name+'_process.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
# Kit may fast-exit with code zero on a Python error: require real artifacts.
raise SystemExit(0 if code==0 and manifest.is_file() and not timed_out else 1)
