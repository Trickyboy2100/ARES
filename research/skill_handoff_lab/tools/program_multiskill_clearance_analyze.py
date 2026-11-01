"""CPU paired transfer-height diagnosis; does not fit a policy on tiny panels."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

def load(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def comparison(a,b):
    # Same initial state and all physical commands through real first-skill end.
    end=a['decisions'][1]['boundary_step'] if len(a['decisions'])>1 else 0
    other=b['decisions'][1]['boundary_step'] if len(b['decisions'])>1 else 0
    keys=['action','q','qvel','tcp','tool','cube','tcp_T_tool']
    deltas={k:max([float(np.max(np.abs(np.asarray(x[k])-np.asarray(y[k])))) for x,y in zip(a['steps'][:end],b['steps'][:other])],default=0.) for k in keys}
    return dict(steps=end,other_steps=other,initial_exact=a['summary']['initial_sha256']==b['summary']['initial_sha256'],max_abs_deltas=deltas,exact=end==other and all(v==0 for v in deltas.values()))
def summarize(data):
    phases={}
    for step in data['steps']:
        phase=step['phase'];v=phases.setdefault(phase,dict(steps=0,tool_contact_peak_N=0.,robot_contact_peak_N=0.,samples=0))
        v['steps']+=1
        for sub in step.get('physics_substep_contacts',[]):
            force=sub['force_norms_N'];v['samples']+=1
            v['tool_contact_peak_N']=max(v['tool_contact_peak_N'],force.get('held_tool',0.))
            v['robot_contact_peak_N']=max(v['robot_contact_peak_N'],max([x for k,x in force.items() if k!='held_tool'],default=0.))
    search=[p for p in data['plans'] if p['phase']=='SEARCH']
    return dict(success=data['summary']['success'],error=data['summary']['error'],steps=len(data['steps']),phases=phases,final_distance_m=data['steps'][-1]['cube_base_distance_m'],selected_parameters=data['decisions'][-1]['parameters'] if data['decisions'] else {},search=search)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prior-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    base={(740001,.035):'740001_g035',(740001,.07):'740001_lookahead',(740000,.07):'740000_lookahead'}
    rows=[]
    for p in sorted(a.root.glob('*/TRAJECTORY.json.gz')):
        d=load(p);m=json.loads((p.parent/'MANIFEST.json').read_text());params=m['requested_parameters'];key=(m['seed'],params['grasp_x']);prior=a.prior_root/base[key]/'TRAJECTORY.json.gz';old=load(prior)
        row=dict(case=p.parent.name,seed=m['seed'],grasp_x=params['grasp_x'],transfer_height_m=params['transfer_height'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),prior_path=str(prior),prior_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),prefix=comparison(old,d),prior=summarize(old),current=summarize(d))
        row['rescued']=not row['prior']['success'] and row['current']['success'];row['degraded']=row['prior']['success'] and not row['current']['success'];rows.append(row)
    output=dict(episodes=len(rows),actual_control_steps=sum(x['current']['steps'] for x in rows),actual_substep_samples=sum(sum(y['samples'] for y in x['current']['phases'].values()) for x in rows),optimizer_updates=0,successes=sum(x['current']['success'] for x in rows),rescued=sum(x['rescued'] for x in rows),degraded=sum(x['degraded'] for x in rows),all_prefixes_exact=all(x['prefix']['exact'] and x['prefix']['initial_exact'] for x in rows),scope='controlled physical transfer-height diagnostic, not RL or independent held-out evaluation',rows=rows)
    (a.out/'CLEARANCE_AUDIT.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2))
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,4.8));values=[x['current']['phases'].get('TRANSFER',{}).get('tool_contact_peak_N',0.) for x in rows]
        ax.bar(range(len(rows)),values,color=['#247a62' if x['current']['success'] else '#b85c43' for x in rows]);ax.set_xticks(range(len(rows)),[f"{x['seed']} / g{x['grasp_x']*1000:.0f}\nz={x['transfer_height_m']*100:.0f}cm / {'PASS' if x['current']['success'] else 'FAIL'}" for x in rows]);ax.set_ylabel('Actual tool–cube peak force during TRANSFER (N)');ax.set_title('Higher transfer endpoints alone did not rescue failed pulls\n6 native100 episodes · actual 100Hz contacts · identical first-skill prefixes')
        for i,v in enumerate(values):ax.text(i,v+.08,f'{v:.2f}',ha='center')
        ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(values)*1.2+.1);fig.tight_layout();fig.savefig(a.out/'TRANSFER_ENDPOINT_CONTACT.png',dpi=180);fig.savefig(a.out/'TRANSFER_ENDPOINT_CONTACT.svg');plt.close(fig)
if __name__=='__main__':main()
