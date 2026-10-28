"""Publish measured physics / seed-clustered statistics, never a test oracle policy."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE=['nominal','rule','fixed','absolute_q','paired_ranker']
COLORS={'nominal':'#667085','ordinary':'#3182bd','candidate':'#b9562c'}
def writecsv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def boundary(r,i,stage):return next((b for b in r['boundaries'][i] if b['stage']==stage),{})
def verified(r,i):return bool(len(r['options'][i])==2 and r['errors'][i] is None)
def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pairedbatchci(values):
    rng=np.random.default_rng(9002);v=np.asarray(values);v=v[None,:] if v.ndim==1 else v;assert v.shape[1]==64
    draws=[]
    for _ in range(10000):
        selected=v[rng.integers(0,v.shape[0],v.shape[0])].reshape(v.shape[0],4,16);draws.append(selected[:,rng.integers(0,4,4),:].mean())
    return [float(x) for x in np.percentile(draws,[2.5,97.5])]
def pairedci(values):
    rng=np.random.default_rng(9001);v=np.asarray(values);v=v[None,:] if v.ndim==1 else v
    draws=np.array([v[rng.integers(0,v.shape[0],v.shape[0])][:,rng.integers(0,v.shape[1],v.shape[1])].mean() for _ in range(10000)])
    return [float(x) for x in np.percentile(draws,[2.5,97.5])]
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root,out=a.root,a.output;out.mkdir(exist_ok=True,parents=True);figs=out/'FIGURES';figs.mkdir(exist_ok=True);plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':14,'axes.titleweight':'bold','savefig.facecolor':'white','font.family':'DejaVu Sans'})
    data=json.loads((root/'opportunity64'/'RESULTS.json').read_text());rs=data['rows'];grid={};op=[]
    for r in rs:
        aa,cc=int(r['policy'][1]),int(r['policy'][4])
        for i,seed in enumerate(r['seeds']):
            grid[seed,aa,cc]=(r,i);b=boundary(r,i,0);c=boundary(r,i,1);op.append(dict(seed=seed,reset_group=(seed-420000)//16,policy=r['policy'],A_action=aa,C_action=cc,A_verified=b.get('valid',False),C_verified=c.get('valid',False),native_success=r['success'][i],verified_program_success=verified(r,i),verified_full_success=bool(verified(r,i) and r['success'][i]),error=r['errors'][i],A_boundary_step=b.get('step'),C_boundary_step=c.get('step'),AB_margin_after_A_m=b.get('native_AB_margin_m'),centered_overlap_after_A_m2=b.get('predicted_centered_min_overlap_m2'),actual_overlap_after_C_m2=c.get('actual_min_overlap_m2'),physical_steps=250,initial_batch_sha256=r['initial_sha256'],trajectory_sha256=r['trajectory_sha256']))
    writecsv(out/'OPPORTUNITY_STUDY.csv',op);seeds=sorted({x[0] for x in grid});opsummary=[];first_cases=[];second_cases=[];prefixdiff=[]
    for aa in range(3):
        for cc in range(3):
            subset=[v for v in op if v['A_action']==aa and v['C_action']==cc];local=sum(v['A_verified'] for v in subset);success=sum(v['native_success'] for v in subset);opsummary.append(dict(A_action=aa,C_action=cc,n=64,A_verified=local,native_success=success,verified_program_success=sum(v['verified_program_success'] for v in subset),conditional_success=success/local if local else None))
    for seed in seeds:
        legal=[]
        for aa in range(3):
            r,i=grid[seed,aa,1];b=boundary(r,i,0)
            if b.get('valid'):legal.append((aa,bool(r['success'][i])))
            variants=[grid[seed,aa,cc] for cc in range(3)];ds=[r['decisions'][i] for r,i in variants]
            if all(len(d)==2 for d in ds):
                diff=float(np.max(np.ptp([d[1]['observation'] for d in ds],axis=0)));prefixdiff.append(diff)
                if diff<2e-6 and len({bool(r['success'][i]) for r,i in variants})>1:second_cases.append([seed,aa])
        if len({s for aa,s in legal})>1:first_cases.append(seed)
    summary=dict(n=64,nominal_success=next(x['native_success'] for x in opsummary if x['A_action']==1 and x['C_action']==1),best_fixed=max(x['native_success'] for x in opsummary),hindsight_union=sum(any(grid[s,aa,cc][0]['success'][grid[s,aa,cc][1]] for aa in range(3) for cc in range(3)) for s in seeds),current_verified_downstream_different_cases=first_cases,identical_posterior_C_choice_different_outcome=second_cases,prefix_comparison_n=len(prefixdiff),prefix_max_abs_difference=max(prefixdiff),grid=opsummary,physical_interactions=data['physical_interactions'])
    (out/'OPPORTUNITY_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6.7,5));z=np.array([x['native_success'] for x in opsummary]).reshape(3,3);im=ax.imshow(z,vmin=48,vmax=64,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{z[i,j]}/64',ha='center',va='center',fontsize=18,color='white' if z[i,j]>57 else '#142438')
    ax.set(xticks=range(3),xticklabels=['-0.35','0','+0.35'],yticks=range(3),yticklabels=['-90','0','+90'],xlabel='C grasp-yaw offset (rad)',ylabel='A placement-direction offset (deg)',title='New training-design scenes: physical 3 x 3 grid');fig.subplots_adjust(bottom=.25);fig.text(.5,.015,f'Native success at250 steps | best fixed {summary["best_fixed"]}/64 | hindsight union {summary["hindsight_union"]}/64\nHindsight union is an opportunity ceiling, not a deployable policy.',ha='center',fontsize=10);save(fig,figs,'Fig3a_physical_opportunity_grid')
    # Same initial intervention with identical nominal suffix. Show actual
    # support changes rather than a requested displacement as physical result.
    case=420010;fig,axes=plt.subplots(1,3,figsize=(13.5,3.8))
    for aa,col in [(0,'#9b3a36'),(2,'#167d75')]:
        rr,idx=grid[case,aa,1];batch=(case-420000)//16;trace=json.loads((root/'opportunity64'/f'{batch:03d}_A{aa}_C1.json').read_text());rows=trace['transitions'];poses=np.array([[t['state']['actors'][x][idx][:3] for x in ['cubeA','cubeB','cubeC']] for t in rows]);t=np.arange(1,251)*.05;AB=np.linalg.norm(poses[:,0,:2]-poses[:,1,:2],axis=1);label=f'A{aa}, native={int(rr["success"][idx])}'
        axes[0].plot(t,(np.sqrt(.04**2+.04**2)+.005-AB)*1000,color=col,label=label);axes[1].plot(t,poses[:,2,2]*1000,color=col,label=label);q=np.array([t['q'][idx][:7] for t in rows]);axes[2].plot(t,np.linalg.norm(q-q[0],axis=1),color=col,label=label)
        b=boundary(rr,idx,0);axes[0].axvline(b['step']*.05,color=col,ls='--',alpha=.5)
    axes[0].axhline(0,color='#444',lw=1);axes[0].set(ylabel='Native AB distance margin (mm)',title='Actual support separation');axes[1].set(ylabel='C center height (mm)',title='Actual downstream object state');axes[2].set(ylabel='Joint displacement norm (rad)',title='Measured joint motion')
    for ax in axes:ax.set_xlabel('Physical simulation time (s)');ax.grid(alpha=.16)
    axes[0].set(xlim=(2.6,7.5),ylim=(-5,30));axes[0].legend(fontsize=9);fig.tight_layout();fig.suptitle('Case420010: both A releases verified; same C parameter, different full-task outcome',y=1.06,fontsize=15);save(fig,figs,'Fig3b_intervention_physical_state')
    fig,ax=plt.subplots(figsize=(9,4.4));x=[];y=[];co=[]
    for v in op:
        if v['A_verified'] and v['C_action']==1:x.append(v['AB_margin_after_A_m']*1000);y.append(v['centered_overlap_after_A_m2']*1e6);co.append('#287d6d' if v['native_success'] else '#c24c43')
    ax.scatter(x,y,c=co,alpha=.65,s=32,edgecolor='white',lw=.4);ax.set(xlabel='Measured AB margin after verified A (mm)',ylabel='Hypothetical centered-C projected overlap (mm²)',title='Verified predecessor states: downstream successes and failures');ax.text(.02,.97,'Green: native success | red: failure\nSame64 scenes x3 A interventions; nominal C suffix\nProjection is a geometric proxy, not contact area or isolated mediation.',transform=ax.transAxes,va='top',fontsize=10);save(fig,figs,'Fig3c_measured_handoff_geometry')
    if second_cases:
        from matplotlib.colors import ListedColormap
        z=np.array([[int(grid[seed,aa,cc][0]['success'][grid[seed,aa,cc][1]]) for cc in range(3)] for seed,aa in second_cases]);fig,ax=plt.subplots(figsize=(6.2,8.8));ax.imshow(z,cmap=ListedColormap(['#d8a19a','#69ad9b']),vmin=0,vmax=1,aspect='auto')
        for i in range(len(second_cases)):
            for j in range(3):ax.text(j,i,str(z[i,j]),ha='center',va='center',fontsize=13)
        ax.set(xticks=range(3),xticklabels=['-0.35','0','+0.35'],yticks=range(len(second_cases)),yticklabels=[f'{seed} / A{aa}' for seed,aa in second_cases],xlabel='Physically executed C grasp-yaw offset (rad)',ylabel='Same-case / same A action prefix',title='Same measured handoff state, different C result');fig.subplots_adjust(bottom=.14,left=.3);fig.text(.5,.025,'18 differing-outcome contexts among181 verified prefixes\nNative outcome at250 steps; no reset between A and C.',ha='center',fontsize=10);save(fig,figs,'Fig3d_second_boundary_interventions')
    histories=[]
    for f in sorted(root.glob('ppo_*/HISTORY.json')):histories.extend(json.loads(f.read_text()))
    writecsv(out/'ONLINE_RL_RESULTS.csv',histories)
    if histories:
        fig,ax=plt.subplots(figsize=(8,4.4))
        for beta,col,label in [(0,COLORS['ordinary'],'Matched ordinary PPO'),(2,COLORS['candidate'],'Paired-risk PPO candidate')]:
            allseed=[]
            for sd in [31,37,43]:
                h=[x for x in histories if x['mode']=='full' and x['seed']==sd and x['beta']==beta];h.sort(key=lambda x:x['update'])
                if len(h)==12:allseed.append(np.array([x['success']/16 for x in h]));ax.plot(np.arange(1,13)*4000,np.array([x['success']/16 for x in h]),c=col,lw=.8,alpha=.25)
            if len(allseed)==3:
                v=np.array(allseed);ax.plot(np.arange(1,13)*4000,v.mean(0),c=col,lw=2.5,label=label);ax.fill_between(np.arange(1,13)*4000,v.min(0),v.max(0),color=col,alpha=.1)
        ax.set(xlabel='Online physical interactions per model',ylabel='Native success /16 rollout episodes',ylim=(.4,1.04),title='Two-boundary online training: all three seeds retained');ax.legend(loc='lower right',fontsize=9);ax.grid(alpha=.15);fig.subplots_adjust(bottom=.24);fig.text(.5,.02,'Shared144k physical design collection is additional. Training curves are not generalization scores.',ha='center',fontsize=10);save(fig,figs,'Fig4_online_learning')
    dev=root/'development64'/'RESULTS.json'
    if not dev.exists():return
    dr=json.loads(dev.read_text())['rows'];methods={};cases=[]
    for r in dr:
        for i,seed in enumerate(r['seeds']):
            methods.setdefault(r['method'],{})[seed]=(r,i);cases.append(dict(method=r['method'],seed=seed,native_success=r['success'][i],verified_program_success=verified(r,i),verified_full_success=bool(verified(r,i) and r['success'][i]),A_verified=boundary(r,i,0).get('valid',False),C_verified=boundary(r,i,1).get('valid',False),error=r['errors'][i],A_choice=r['decisions'][i][0]['action'],C_choice=r['decisions'][i][1]['action'] if len(r['decisions'][i])>1 else '',A_tau=boundary(r,i,0).get('physical_steps'),native_tau=250,logical_finish=r['logical_finish_steps'][i],trajectory_sha256=r['trajectory_sha256'],model_sha256=r.get('model_sha256')))
    writecsv(out/'CASEWISE_EFFECTS.csv',cases);ss=list(range(520000,520064));nom=np.array([methods['nominal'][s][0]['success'][methods['nominal'][s][1]] for s in ss],int);stats=[]
    for name,mp in methods.items():
        if len(mp)!=64:continue
        vs=np.array([mp[s][0]['success'][mp[s][1]] for s in ss],int);local=sum(boundary(*mp[s],0).get('valid',False) for s in ss);complete=sum(verified(*mp[s]) for s in ss);good=sum(vs.astype(bool)&~nom.astype(bool));bad=sum(~vs.astype(bool)&nom.astype(bool));ci=pairedci(vs-nom);stats.append(dict(method=name,n=64,native_success=int(vs.sum()),verified_program_success=complete,verified_full_success=sum(verified(*mp[s]) and mp[s][0]['success'][mp[s][1]] for s in ss),A_verified=local,conditional_success=float(sum(bool(vs[j]) and boundary(*mp[s],0).get('valid',False) for j,s in enumerate(ss))/local) if local else None,rescues=int(good),regressions=int(bad),net_pp=float((vs-nom).mean()*100),paired_ci_low_pp=ci[0]*100,paired_ci_high_pp=ci[1]*100,reset_batch_ci_low_pp=pairedbatchci(vs-nom)[0]*100,reset_batch_ci_high_pp=pairedbatchci(vs-nom)[1]*100,C_verified=sum(boundary(*mp[s],1).get('valid',False) for s in ss),physical_evaluation_interactions=16000))
    writecsv(out/'DEVELOPMENT_RESULTS.csv',stats)
    mainstats={x['method']:x for x in stats};groups=[]
    for be in [0,2]:
        names=[f'ppo_s{s}_b{be}_full' for s in [31,37,43]]
        if all(n in methods and len(methods[n])==64 for n in names):
            v=np.array([[methods[n][s][0]['success'][methods[n][s][1]] for s in ss] for n in names],int);ci=pairedci(v-nom);groups.append(dict(beta=be,seeds=[31,37,43],success_counts=v.sum(1).tolist(),mean_success=float(v.mean()),mean_net_pp=float((v-nom).mean()*100),case_and_seed_bootstrap_ci_pp=[x*100 for x in ci],reset_batch_and_seed_bootstrap_ci_pp=[x*100 for x in pairedbatchci(v-nom)]))
    pair=None
    if len(groups)==2:
        v=[]
        for sd in [31,37,43]:v.append([int(methods[f'ppo_s{sd}_b2_full'][s][0]['success'][methods[f'ppo_s{sd}_b2_full'][s][1]])-int(methods[f'ppo_s{sd}_b0_full'][s][0]['success'][methods[f'ppo_s{sd}_b0_full'][s][1]]) for s in ss])
        pair=dict(mean_difference_pp=float(np.mean(v)*100),seed_differences=[int(sum(x)) for x in v],case_and_seed_bootstrap_ci_pp=[x*100 for x in pairedci(v)],reset_batch_and_seed_bootstrap_ci_pp=[x*100 for x in pairedbatchci(v)])
    (out/'STATISTICAL_SUMMARY.json').write_text(json.dumps(dict(methods=stats,seed_groups=groups,candidate_minus_ordinary=pair,unit='64 scenes in4 reset batches crossed with3 training seeds; bootstrap both scenes and reset-batches, never treat192 as independent task cases',sealed_test='UNOPENED'),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,6));names=[n for n in BASE if n in mainstats]+[f'ppo_s{s}_b{be}_full' for be in [0,2] for s in [31,37,43] if f'ppo_s{s}_b{be}_full' in mainstats];vals=[mainstats[n]['native_success'] for n in names];labels=[n.replace('ppo_s','seed').replace('_b0_full',' ordinary').replace('_b2_full',' candidate') for n in names];cols=[COLORS['ordinary'] if '_b0_' in n else COLORS['candidate'] if '_b2_' in n else COLORS['nominal'] for n in names];ax.barh(np.arange(len(names)),vals,color=cols);ax.set(yticks=np.arange(len(names)),yticklabels=labels,xlim=(0,68),xlabel='Native successes out of64 UNSEEN development scenes',title='Matched interfaces: strong search and supervised baselines included');ax.invert_yaxis()
    for i,v in enumerate(vals):ax.text(v+.4,i,str(v),va='center',fontsize=10)
    save(fig,figs,'Fig5a_development_all_seeds')
    fig,ax=plt.subplots(figsize=(8.3,4.5));names=[n for n in names if n!='nominal'];xx=[mainstats[n]['net_pp'] for n in names];lo=[mainstats[n]['paired_ci_low_pp'] for n in names];hi=[mainstats[n]['paired_ci_high_pp'] for n in names];yy=np.arange(len(names));ax.errorbar(xx,yy,xerr=[np.maximum(np.array(xx)-lo,0),np.maximum(np.array(hi)-xx,0)],fmt='o',color='#225b78',capsize=3);ax.axvline(0,c='#777',ls='--');ax.set(yticks=yy,yticklabels=[n.replace('ppo_s','seed').replace('_b0_full',' ordinary').replace('_b2_full',' candidate') for n in names],xlabel='Paired native success difference from nominal (percentage points)',title='Paired task-case bootstrap intervals (descriptive95%)');ax.invert_yaxis();save(fig,figs,'Fig5b_paired_intervals')
    abl=['ppo_s31_b2_full_04','ppo_s31_b2_no_next','ppo_s31_b2_no_posterior','ppo_s31_b2_no_second']
    if all(n in mainstats for n in abl):
        fig,ax=plt.subplots(figsize=(7.8,4));vv=[mainstats[n]['native_success'] for n in abl];ax.bar(range(4),vv,color=['#b9562c','#647e98','#647e98','#647e98']);ax.set(xticks=range(4),xticklabels=['Full\nfeedback','No next\ngeometry','No object\nposterior','Commit both\ninitially'],ylabel='Native successes /64',ylim=(0,68),title='Matched16k online-interaction ablations (seed31 only)')
        for i,v in enumerate(vv):ax.text(i,v+.6,str(v),ha='center')
        fig.subplots_adjust(bottom=.26);fig.text(.5,.02,'Information ablations are exploratory; a single training seed is not a robust mechanism claim.',ha='center',fontsize=10);save(fig,figs,'Fig6_information_and_redecision_ablations')
    tex=out/'LATEX';tex.mkdir(exist_ok=True)
    text='\\begin{tabular}{lrrrrr}\n\\toprule\nMethod & Native/64 & Verified/64 & A verified & Rescue & Regress\\\\\n\\midrule\n'
    for s in stats:text+=s['method'].replace('_','\\_')+f" & {s['native_success']} & {s['verified_program_success']} & {s['A_verified']} & {s['rescues']} & {s['regressions']} "+r'\\'+'\n'
    text+='\\bottomrule\n\\end{tabular}\n';(tex/'development_results.tex').write_text(text)
    print(json.dumps(dict(opportunity=summary,candidate_comparison=pair),indent=2))
if __name__=='__main__':main()
