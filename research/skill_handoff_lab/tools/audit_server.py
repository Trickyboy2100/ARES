"""Bounded read-only inventory, to run after the server identity is confirmed."""
import argparse,json,platform,subprocess,shutil,time
from pathlib import Path

COMMANDS={
 "gpu":["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version","--format=csv"],
 "gpu_jobs":["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory","--format=csv"],
 "gpu_topology":["nvidia-smi","topo","-m"],
 "cpu":["lscpu"],"ram":["free","-h"],
 "tasks":["ps","-eo","user,pid,comm,pcpu,pmem","--sort=-pcpu"],
 "cuda":["nvcc","--version"],"docker":["docker","--version"],
 "docker_images":["docker","images","--format","{{.Repository}}:{{.Tag}} {{.Size}}"],
 "conda":["conda","env","list"],
 "python_packages":["python3","-c","import importlib.metadata as m,json; names=['torch','numpy','mujoco','nvidia-curobo','curobo','isaacsim']; print(json.dumps({n:next((d.version for d in m.distributions() if d.metadata['Name'].lower()==n),None) for n in names}))"]}
def collect():
    result={"hostname":platform.node(),"kernel":platform.release(),"created_unix":time.time(),
        "robot_io":False,"containers_started":False,"commands":{}}
    for name,argv in COMMANDS.items():
        try:
            p=subprocess.run(argv,capture_output=True,text=True,timeout=15)
            result["commands"][name]={"argv":argv,"returncode":p.returncode,
                "stdout":p.stdout[:18000],"stderr":p.stderr[:2000]}
        except (OSError,subprocess.TimeoutExpired) as exc:
            result["commands"][name]={"error":str(exc)}
    disk=shutil.disk_usage(Path.cwd());result["disk_bytes"]={"total":disk.total,"free":disk.free}
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(collect(),indent=2)+"\n")
