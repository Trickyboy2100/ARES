"""Measured tool grasp, suffix outcome and finite-action opportunity summaries."""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def angle(q,r):return float(np.degrees(2*np.arccos(np.clip(abs(np.dot(q,r))/(np.linalg.norm(q)*np.linalg.norm(r)),0,1))))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];groups={};prefix_checks={}
    for result in sorted(a.run.glob('*/RESULTS.json')):
        rr=json.loads(result.read_text());group=result.parent.name;groups[group]=dict(episodes=len(rr),steps=sum(r['steps'] for r in rr),methods={})
        for r in rr:
            name=f"{r['seed']}_{r['mode']}";path=result.parent/f'{name}.json.gz'
            with gzip.open(path,'rt') as f:t=json.load(f)
            ss=t['steps'];phases={ph:[z for z in ss if z['phase']==ph] for ph in {z['phase'] for z in ss}};initial=t['initial_measure'];lift=phases.get('LIFT',[None])[-1];pull=phases.get('PULL',[None])[-1];hook=phases.get('HOOK',[None])[-1];end=ss[-1]
            x=r.get('parameters',{}).get('grasp_x',float('nan'));row=dict(group=group,seed=r['seed'],method=r['mode'],success=int(r['success']),error=r['error'],grasp_x=x,yaw=r.get('parameters',{}).get('yaw',float('nan')),lift_grasped=int(lift['grasped']) if lift else 0,final_grasped=int(end['grasped']),cube_dx_m=end['cube'][0]-initial['cube'][0],tool_relative_rotation_drift_deg=angle(lift['tcp_T_tool'][3:],end['tcp_T_tool'][3:]) if lift else float('nan'),tool_relative_translation_drift_m=float(np.linalg.norm(np.array(lift['tcp_T_tool'][:3])-end['tcp_T_tool'][:3])) if lift else float('nan'),peak_tool_cube_force_N=max(z['tool_cube_force_N'] for z in ss),peak_arm_qvel=max(max(abs(np.array(z['qvel'][:7]))) for z in ss),wall_s=r['wall_s'],planning_wall_s=r['planning_wall_s'],steps=r['steps']);rows.append(row)
            mm=groups[group]['methods'].setdefault(r['mode'],dict(n=0,success=0,errors={},wall_s=[],planning_wall_s=[]));mm['n']+=1;mm['success']+=int(r['success']);mm['errors'][str(r['error'])]=mm['errors'].get(str(r['error']),0)+1;mm['wall_s'].append(r['wall_s']);mm['planning_wall_s'].append(r['planning_wall_s'])
            if group=='design8_v4_retry' and lift and r['mode'].startswith('g'):
                key=(r['seed'],x);state=np.r_[lift['q'],lift['qvel'],lift['tool'],lift['cube'],lift['tool_linear_velocity'],lift['tool_angular_velocity'],lift['cube_linear_velocity']];prefix_checks.setdefault(key,[]).append(state)
        byseed={s:[r for r in rr if r['seed']==s] for s in {r['seed'] for r in rr}}
        groups[group]['finite_observed_union']=sum(any(r['success'] for r in rs) for rs in byseed.values());groups[group]['case_count']=len(byseed)
    with (a.out/'EPISODES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    checks=[max(np.max(abs(x-v[0])) for x in v) for v in prefix_checks.values() if len(v)==3]
    output=dict(groups=groups,physical_steps=sum(g['steps'] for g in groups.values()),recorded_prefix_groups=len(checks),recorded_prefix_max_difference=float(max(checks)) if checks else None)
    (a.out/'SUMMARY.json').write_text(json.dumps(output,indent=2))
    selected=[r for r in rows if r['group']=='design8_v4_retry' and r['method'].startswith('g')]
    if selected:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for success,color in [(0,'#b91c1c'),(1,'#0f766e')]:
            rr=[r for r in selected if r['success']==success];ax.scatter([r['grasp_x']*1000 for r in rr],[-r['cube_dx_m']*1000 for r in rr],c=color,label='Native success' if success else 'Native failure',alpha=.65,s=30)
        ax.set_xlabel('Commanded grasp position along tool handle [mm]');ax.set_ylabel('Measured cube displacement toward robot [mm]');ax.set_title('Verified grasp does not guarantee useful tool contact');ax.legend();fig.tight_layout();fig.savefig(a.out/'grasp_effect.png',dpi=180);plt.close(fig)
        methods=sorted({r['method'] for r in selected});seeds=sorted({r['seed'] for r in selected});lookup={(r['seed'],r['method']):r for r in selected};mat=[[lookup[s,m]['success'] for m in methods] for s in seeds];fig,ax=plt.subplots(figsize=(9,4.5));ax.imshow(mat,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(methods)),methods,rotation=35,ha='right');ax.set_yticks(range(len(seeds)),[str(s) for s in seeds]);ax.set_title('Actual native100 outcomes; 1=success, 0=failure')
        for i,row in enumerate(mat):
            for j,v in enumerate(row):ax.text(j,i,str(v),ha='center',va='center')
        fig.tight_layout();fig.savefig(a.out/'paired_parameter_outcomes.png',dpi=180);plt.close(fig)
    print(json.dumps({k:dict(episodes=v['episodes'],case_count=v['case_count'],union=v['finite_observed_union'],methods={n:(m['success'],m['n']) for n,m in v['methods'].items()}) for k,v in groups.items()},indent=2))
if __name__=='__main__':main()
