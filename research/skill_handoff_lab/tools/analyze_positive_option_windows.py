"""Does a successful intervention leave unfinished work, or only success maintenance?"""
import argparse,gzip,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=[]
for folder in ['design_a','design_b','disagreement_design']:
 for pair in json.loads((a.root/folder/'PAIRS.json').read_text()):
  if pair['delta']<=0:continue
  path=Path(pair['branch_path']);raw=json.load(gzip.open(path,'rt'));success_steps=[z['step']for z in raw['steps']if bool(z['evaluation']['success'][0])];release=pair['step']+pair['duration'];first=min(success_steps)if success_steps else None
  out.append(dict(collection=folder,seed=pair['seed'],entry=pair['step'],selector=pair['action'],duration=pair['duration'],release=release,remaining=100-release,first_native_success_step=first,success_already_reached_at_release=first is not None and first<=release,final_success=raw['summary']['success'],grasp_false_after_success=sum(not z['grasped']for z in raw['steps']if first is not None and z['step']>=first),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Per-step official success and final native success both retained. Remaining budget after success is not unfinished multi-intervention work; no changed success criterion.'))
(a.root/'POSITIVE_OPTION_WINDOWS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
