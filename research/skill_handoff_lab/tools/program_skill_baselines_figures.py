"""Plot completed development comparisons; never infer missing physics results."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();root=a.evidence
    scores=json.loads((root/'EVAL_SCORES.json').read_text());pairs=json.loads((root/'PAIRS.json').read_text())
    rows={x['group']:x for x in scores if x['family']=='ALL' and x['completion']=='dwell' and x['episodes']==36}
    authorities=['full','y_only','task'];labels=['Full YZ','Y-only projection','Task-conditioned projection']
    methods=[('Original search','#8c9ca7',lambda x:('controls_full' if x=='full' else 'controls_'+x)+'::search'),('Deduplicated + 4mm search','#e09d47',lambda x:'controls_fast_'+x+'::search_unique_fast'),('Ordinary LN-SAC 2048','#7ab3bf',lambda x:'eval_'+x+'_2048::model'),('Ordinary LN-SAC 4096','#245f80',lambda x:'eval_'+x+'::model')]
    fig,ax=plt.subplots(figsize=(10.4,4.6));width=.19
    for j,(label,color,key) in enumerate(methods):
        for i,authority in enumerate(authorities):
            x=i+(j-1.5)*width;r=rows.get(key(authority))
            if r is None:
                ax.text(x,1.5,'Pending',rotation=90,ha='center',fontsize=7);continue
            ax.bar(x,r['success'],width=.18,color=color,label=label if i==0 else None)
            ax.text(x,r['success']+.3,str(r['success']),ha='center',fontsize=9)
    ax.set(xticks=range(3),xticklabels=labels,ylabel='Native success at step 200 (out of 36)',ylim=(0,39),title='Shared observable completion / identical development conditions')
    ax.axhline(36,color='#526579',lw=.7,ls=':');ax.legend(ncol=2,loc='lower left',fontsize=8,framealpha=.95)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.01,'6 scene clusters × 6 conditions; one optimizer seed; projected 3D SAC. Development evidence, not sealed generalization.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(root/'CORE_RESULTS.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.1));out=[]
    for i,authority in enumerate(authorities):
        control='controls_fast_'+authority+'::search_unique_fast';model='eval_'+authority+'::model'
        pair=next((x for x in pairs if x['left']==control and x['right']==model),None)
        if pair is None:ax.text(i,0,'Pending',ha='center');continue
        if pair['matched_conditions']!=36 or pair['recorded_prefix_equal']!=36:raise ValueError('Refusing incompletely paired figure')
        rescue=pair['right_rescues'];regress=pair['right_regressions'];out.append(dict(authority=authority,search=control,model=model,rescues=rescue,regressions=regress,matched_conditions=36,recorded_prefix_equal=36,scene_cluster_bootstrap=pair['scene_cluster_bootstrap']))
        ax.bar(i-.16,rescue,.3,color='#238879',label='SAC rescues search failures' if i==0 else None)
        ax.bar(i+.16,-regress,.3,color='#b94a40',label='SAC loses search successes' if i==0 else None)
        ax.text(i-.16,rescue+.2,str(rescue),ha='center');ax.text(i+.16,-regress-.7,str(regress),ha='center')
    ax.axhline(0,color='#526579',lw=.7);ax.set(xticks=range(3),xticklabels=labels,ylabel='Paired conditions (rescue / regression)',title='4096-decision SAC vs same-authority fast search')
    if out:
        ax.set_ylim(-max(x['regressions'] for x in out)-2,max(x['rescues'] for x in out)+2)
        ax.legend(fontsize=8,loc='best')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(root/'PAIRED_TRADEOFF.png',dpi=180);plt.close(fig)
    (root/'CORE_FIGURE_SOURCES.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['EVAL_SCORES.json','PAIRS.json']},paired=out,physics_steps_added=0,limitation='Raw development counts; six scene clusters, not 36 independent initial scenes. Error bars omitted; exploratory cluster intervals are in PAIRS.json. No novelty or convergence claim.'),indent=2)+'\n')

if __name__=='__main__':main()
