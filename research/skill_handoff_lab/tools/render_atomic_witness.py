"""Render a complete learned physical rescue with exact source SHA and frames."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();paths=[]
for name in ['queue_bc64','macro_bc64']:
 h=json.loads((a.root/name/'HISTORY.json').read_text());paths.append(next(v['row']['path']for v in h if v['job']==dict(kind='eval',checkpoint=32,seed=761202)))
subprocess.run([sys.executable,str(Path(__file__).with_name('render_peg_skill_pair.py')),'--left',paths[0],'--right',paths[1],'--out',str(a.root/'video'),'--left-title','Queue SAC: correction at step59 | FAILURE','--right-title','Macro-trained SAC: correction at step63 | SUCCESS','--comparison-title','761202: real learned intervention timing; seen development'],check=True)
