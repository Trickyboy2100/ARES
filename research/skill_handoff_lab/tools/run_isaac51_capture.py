"""Own-process timeout/log manager; never kills another Isaac process."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--isaac-root",type=Path,required=True)
    p.add_argument("--scene",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=0);p.add_argument("--timeout",type=int,default=240)
    a=p.parse_args();version=(a.isaac_root/"VERSION").read_text().strip()
    if not version.startswith("5.1."):raise ValueError("Isaac Sim 5.1 required")
    subprocess.run(["nvidia-smi"],check=True,capture_output=True,timeout=15)
    a.output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    argv=[str(a.isaac_root/"python.sh"),str(Path(__file__).with_name("isaac51_capture.py")),
        "--scene",str(a.scene.resolve()),"--output",str(a.output.resolve())]
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
