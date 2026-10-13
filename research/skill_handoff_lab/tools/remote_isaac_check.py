"""Bounded personal-directory Isaac5.1 minimal checker, no driver maintenance."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path

def run(command,path,env,timeout):
    started=time.monotonic();timed_out=False
    with path.open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    return {'command':command,'pid':child.pid,'returncode':code,'timeout':timed_out,'wall_s':time.monotonic()-started}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--render-core',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);env=os.environ.copy()
    for key in ['ALL_PROXY','all_proxy','HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy']:env.pop(key,None)
    env.update(UV_CACHE_DIR=str(a.root/'envs/uv-cache'),UV_HTTP_TIMEOUT='30',OMNI_KIT_ACCEPT_EULA='YES',
        VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='4')
    # Vulkan enumerates physical GPUs separately; select activeGpu, not CUDA remapping.
    env.pop('CUDA_VISIBLE_DEVICES',None)
    uv=str(a.root/'envs/isaac-installer/bin/uv');python=str(a.root/'envs/isaac51-check/bin/python')
    command=[uv,'pip','install','--python',python,'--index','https://pypi.nvidia.com',
        '--default-index','https://pypi.org/simple','isaacsim[compatibility-check]==5.1.0','packaging','setuptools']
    if a.render_core:
        env['UV_HTTP_TIMEOUT']='180'
        command.extend(name+'==5.1.0' for name in ['isaacsim-core','isaacsim-replicator','isaacsim-asset','isaacsim-storage','isaacsim-utils'])
    install=run(command,a.output/'install.log',env,300 if a.render_core else 240)
    (a.output/'install_process.json').write_text(json.dumps(install,indent=2));print(json.dumps({'install':install}),flush=True)
    if install['returncode'] or install['timeout']:return
    command=[str(a.root/'envs/isaac51-check/bin/isaacsim'),'isaacsim.exp.compatibility_check','--no-window',
        '--/app/quitAfter=20','--/renderer/activeGpu='+str(a.gpu),'--/renderer/multiGpu/enabled=false']
    check=run(command,a.output/'compatibility.log',env,120)
    (a.output/'check_process.json').write_text(json.dumps(check,indent=2));print(json.dumps({'check':check}),flush=True)
    # Exit code alone is not renderer/physics success: inspect full Kit log.

if __name__=='__main__':main()
