"""Four native100 episodes: two unchanged controls + two TRANSFER-only slowdowns."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for gx in [.035,.070]:
        for stride in [2,1]:
            label=f'740001_g{int(gx*1000):03d}_transfer_stride{stride}'
            cmd=[sys.executable,str(Path(__file__).with_name('program_multiskill_contact_probe.py')),'--seed','740001','--grasp-x',str(gx),'--transfer-stride',str(stride),'--label',label,'--out',str(a.out/label)]
            print('RUN',label,flush=True);start=time.monotonic()
            with (a.out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            rows.append(dict(label=label,command=cmd,exit_code=r.returncode,wall_s=time.monotonic()-start,resource_context=os.environ.get('ARES_RESOURCE_CONTEXT','UNSPECIFIED')));(a.out/'QUEUE_ACCOUNTING.json').write_text(json.dumps(rows,indent=2))
            if r.returncode:raise RuntimeError('Preserved failure; stop queue: '+label)
if __name__=='__main__':main()
