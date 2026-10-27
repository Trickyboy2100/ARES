"""Re-read existing public Stack64 evidence; no new training or test labels."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    names=['ms_stack_residual_zero_eval64','ms_stack_residual_256k_eval64','ms_stack_residual_seed37_eval64','ms_stack_residual_seed43_eval64'];labels=['nominal (zero residual)','residual31','residual37','residual43'];records=[];rows=[];source=[]
    for name in names:
        group=[]
        for i in range(16):
            f=a.data/name/f'episode_{i:03d}.json';d=json.loads(f.read_text());group.append(d);source.append(dict(path=str(f),sha256=sha(f)))
        records.append(group)
    for i in range(16):
        hashes=[hashlib.sha256(json.dumps(rr[i]['initial_state'],sort_keys=True).encode()).hexdigest() for rr in records];assert len(set(hashes))==1
        for k,rr in enumerate(records):
            for env,y in enumerate(rr[i]['summary']['native_final']['success']):
                n=records[0][i]['summary']['native_final']['success'][env]
                rows.append(dict(source_kind='HISTORICAL_PUBLIC_PHYSICS_NOT_NEW_TRAINING',policy=labels[k],batch_seed=rr[i]['summary']['seed'],env_index=env,case_index=i*4+env,success=int(y),nominal_success=int(n),pair='RESCUE' if y>n else 'REGRESSION' if y<n else 'BOTH_SUCCESS' if y else 'BOTH_FAIL',initial_sha256=hashes[k]))
    with (a.output/'HISTORICAL_STACK64.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    matrix=np.array([[r['success'] for r in rows if r['policy']==label] for label in labels]);assert matrix.sum(1).tolist()==[58,61,62,60]
    fig,ax=plt.subplots(figsize=(14,4.2));fig.subplots_adjust(left=.20,bottom=.22,top=.70);ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap='RdYlGn',vmin=0,vmax=1);ax.set_yticks(range(4),labels);ax.set_xticks(np.arange(0,64,4));ax.set_xlabel('Historical case index (16 reset groups x4 environments)');fig.suptitle('HISTORICAL public Stack64: nominal58, residual61/62/60',fontsize=17);fig.text(.2,.80,'Read-only reuse of 2026-10-27 SAPIEN evidence; never used to fit the new selector.',fontsize=10);fig.text(.2,.04,'Green=native terminal success; red=failure. This is not the new64 development set.',fontsize=10);target=a.output/'figures/Fig0_historical_stack64.png';fig.savefig(target,dpi=180);plt.close(fig)
    # All five shared historical rescues, plus every seed43 regression.
    selected=[r for r in rows if (r['policy']=='residual37' and r['pair']=='RESCUE') or (r['policy']=='residual43' and r['pair']=='REGRESSION')];videos=[]
    for r in selected:
        i=r['case_index']//4;idx=r['env_index'];k=labels.index(r['policy']);left=a.data/names[0]/f'episode_{i:03d}.mp4';right=a.data/names[k]/f'episode_{i:03d}.mp4';target=a.output/'media'/f'HISTORICAL_{r["pair"]}_case{r["case_index"]}_{r["policy"]}.mp4'
        # Original recorder concatenated four512px environment views along X.
        text=f'HISTORICAL SAPIEN PhysX | seed {r["batch_seed"]} env {idx} | nominal vs {r["policy"]} | {r["pair"]}'
        filt=f'[0:v]crop=512:512:{idx*512}:0[l];[1:v]crop=512:512:{idx*512}:0[r];[l][r]hstack,pad=iw:ih+36:0:36:white,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text=\'{text}\':fontsize=16:fontcolor=black:x=8:y=8[v]'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(left),'-i',str(right),'-filter_complex',filt,'-map','[v]','-c:v','libx264','-crf','20',str(target)],check=True);videos.append(dict(file='media/'+target.name,source=[dict(path=str(x),sha256=sha(x)) for x in [left,right]],sha256=sha(target),case=r,render='Actual recorded4-env horizontal image, exact selected environment crop; original20Hz timing'))
    manifest=dict(source_kind='HISTORICAL_PUBLIC_PHYSICS_NOT_NEW_TRAINING',sources=source,videos=videos,figure=dict(file='figures/Fig0_historical_stack64.png',sha256=sha(a.output/'figures/Fig0_historical_stack64.png')),no_new_interactions=True)
    (a.output/'HISTORICAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    html='<h2>历史64案例复核（不计为本轮新训练）</h2><p>58 vs61/62/60；仅用于确认旧问题，未用来训练新gate。全部5个共同救回与seed43的3个退化均保留。</p><img src="figures/Fig0_historical_stack64.png">'
    for r in videos:html+=f'<p>{r["file"]}</p><video controls preload="none" src="{r["file"]}"></video>'
    with (a.output/'index.html').open('a') as f:f.write(html)
    print(json.dumps(dict(historical_counts=matrix.sum(1).tolist(),videos=len(videos))))
if __name__=='__main__':main()
