"""Own-process timeout/log manager; never kills another Isaac process."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--isaac-root",type=Path)
    p.add_argument('--isaac-python',type=Path);p.add_argument('--vulkan-icd',type=Path)
    p.add_argument("--scene",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=0);p.add_argument("--timeout",type=int,default=240)
    a=p.parse_args()
    if a.isaac_root:
        version=(a.isaac_root/'VERSION').read_text().strip();python=a.isaac_root/'python.sh'
    elif a.isaac_python:
        python=a.isaac_python
        version=subprocess.check_output([str(python),'-c','import importlib.metadata; print(importlib.metadata.version("isaacsim"))'],text=True).strip()
    else:raise ValueError('Isaac root or pip environment Python required')
    if not version.startswith("5.1."):raise ValueError("Isaac Sim 5.1 required")
    subprocess.run(["nvidia-smi"],check=True,capture_output=True,timeout=15)
    a.output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env['OMNI_KIT_ACCEPT_EULA']='YES'
    for key in ['ALL_PROXY','all_proxy','HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy']:env.pop(key,None)
    if a.vulkan_icd:env['VK_ICD_FILENAMES']=str(a.vulkan_icd)
    argv=[str(python),str(Path(__file__).with_name("isaac51_capture.py")),
        "--scene",str(a.scene.resolve()),"--output",str(a.output.resolve()),'--gpu',str(a.gpu)]
    started=time.monotonic();timed_out=False
    with (a.output/"isaac.log").open("w") as log:
        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    manifest={"argv":argv,"isaac_version":version,"gpu":a.gpu,"pid":child.pid,
        "returncode":code,"timeout":timed_out,"wall_s":time.monotonic()-started}
    (a.output/"process_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if code or timed_out:raise RuntimeError("capture failed; inspect isolated isaac.log")
if __name__=="__main__":main()
