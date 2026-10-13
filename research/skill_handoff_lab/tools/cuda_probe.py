"""Real, bounded CUDA smoke test, not a planner throughput benchmark."""
import argparse, hashlib, json, platform, subprocess, time
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    samples=[]
    for _ in range(3):
        start=time.perf_counter()
        x=torch.arange(1048576,device='cuda',dtype=torch.float32).reshape(1024,1024)/1048576
        y=x@x.T;torch.cuda.synchronize()
        samples.append({'wall_s':time.perf_counter()-start,'sum':float(y.sum()),
            'finite':bool(torch.isfinite(y).all())})
    data={'host':platform.node(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'samples':samples,'status':'PASS',
        'evidence':'REAL_CUDA_MATRIX_MULTIPLICATION_NOT_CUROBO',
        'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.used,memory.total',
            '--format=csv'],text=True),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'cuda_probe.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))

if __name__=='__main__':main()
