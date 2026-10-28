"""Compact, reproducible native-outcome and paired-regression evidence."""
import argparse,csv,gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);(a.out/'figures').mkdir(exist_ok=True)
    sets={k:json.loads((a.run/k/'RESULTS.json').read_text()) for k in ['physical32','physical32_contact','tracking_probe','physical32_contact_paired']};rr=sets['physical32']['rows']+sets['physical32_contact']['rows'];cases={};digests={};manifest=[]
    for group,ds in sets.items():
        for r in ds['rows']:
            assert r['native_horizon']==250 and r['physical_interactions']==4000
            key=tuple(r['seeds']);old=digests.setdefault(key,r['initial_sha256']);assert old==r['initial_sha256']
            path=a.run/group/r['trajectory_file'];digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==r['trajectory_sha256'];manifest.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size))
        with gzip.open(a.out/f'{group}_RESULTS.json.gz','wt') as f:json.dump(ds,f,separators=(',',':'))
    for r in rr:
        for i,seed in enumerate(r['seeds']):cases.setdefault(r['method'],{})[seed]=dict(y=int(r['success'][i]),error=r['errors'][i],actions=[d['action'] for d in r['decisions'][i]],A_verified=any(b['stage']==0 and b['valid'] for b in r['boundaries'][i]))
    base=cases['nominal'];table=[]
    for name,dd in cases.items():
        assert len(dd)==32 and dd.keys()==base.keys()
        table.append(dict(method=name,cases=32,success=sum(z['y'] for z in dd.values()),rescues=sum(z['y'] and not base[s]['y'] for s,z in dd.items()),regressions=sum(base[s]['y'] and not z['y'] for s,z in dd.items()),A_verified=sum(z['A_verified'] for z in dd.values()),suffix_failures=sum(z['A_verified'] and not z['y'] for z in dd.values())))
    with (a.out/'PHYSICAL_RESULTS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
    (a.out/'PAIRED_CASES.json').write_text(json.dumps(cases,indent=2));(a.out/'RAW_TRAJECTORY_INDEX.json').write_text(json.dumps(manifest,indent=2))
    total=sum(ds['physical_interactions'] for ds in sets.values());assert total==104000
    probe=[]
    for batch in [0,3]:
        x,y=[r for r in sets['tracking_probe']['rows'] if r['batch']==batch]
        for dx,dy in zip(x['decisions'],y['decisions']):
            assert len(dx)==len(dy)
            if len(dx)==2:assert dx[1]['observation']==dy[1]['observation']
        probe.append(dict(batch=batch,baseline=sum(x['success']),full_samples=sum(y['success']),rescues=sum(not i and j for i,j in zip(x['success'],y['success'])),regressions=sum(i and not j for i,j in zip(x['success'],y['success'])),identical_observed_handoffs=True))
    paired={}
    for r in sets['physical32_contact_paired']['rows']:
        for seed,y in zip(r['seeds'],r['success']):paired.setdefault(r['method'],{})[seed]=int(y)
    b=paired['fitted_value'];c=paired['contract_contact'];paired_result=dict(fitted_value=sum(b.values()),contract_contact=sum(c.values()),rescues=sum(c[s] and not b[s] for s in b),regressions=sum(b[s] and not c[s] for s in b),cases=32,scope='Same-process paired comparison, shared strict-state plan cache')
    (a.out/'CONTACT_PAIRED_RESULT.json').write_text(json.dumps(dict(summary=paired_result,cases=paired),indent=2))
    (a.out/'EXPERIMENT_SUMMARY.json').write_text(json.dumps(dict(total_new_physics_steps=total,new_physics_episodes=total//250,offline_models_trained=5,optimizer_steps=5*2*300,tracking_probe=probe,contact_same_process=paired_result,table=table,train_seeds=[420000,420047],calibration_seeds=[420048,420063],development_seeds=[530000,530031],contact_variant='Adapted after first results; same reused development cases, not unseen evaluation',sealed_test_opened=False),indent=2))
    for n in ['OPPORTUNITY.json','CONTACT_DIAGNOSIS.csv','CONTACT_DIAGNOSIS.json','source_v1_handoff_contract_value.py','source_v2_handoff_contract_value.py']:shutil.copy2(a.audit/n,a.out/n)
    for kind in ['models','models_contact']:shutil.copytree(a.run/kind,a.out/kind,dirs_exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    op=json.loads((a.audit/'OPPORTUNITY.json').read_text());fig,ax=plt.subplots(figsize=(9,4.6));vals=[op[k] for k in ['nominal','best_fixed','oracle_A_Cnom','oracle_C_Anom','joint_oracle']];bars=ax.bar(['Nominal','Best fixed','A oracle\nC nominal','C oracle\nA nominal','Joint oracle'],vals,color=['#64748b','#2563eb','#b7791f','#b7791f','#b7791f']);ax.bar_label(bars,padding=4);ax.set_ylim(0,68);ax.set_ylabel('Native success / 64 initial states');ax.set_title('Finite tested-action opportunity — DESIGN data');fig.text(.02,.015,'Orange: retrospective ceilings, not deployable policy scores',fontsize=10);fig.tight_layout(rect=(0,.045,1,1));fig.savefig(a.out/'figures/opportunity.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));plot_rows=[r for r in table if r['method']!='contract_contact'];names=[r['method'] for r in plot_rows];y=np.arange(len(names));ax.barh(y,np.array([r['rescues'] for r in plot_rows]),color='#0f766e',label='Rescues vs nominal');ax.barh(y,-np.array([r['regressions'] for r in plot_rows]),color='#b91c1c',label='Regressions vs nominal');ax.set_yticks(y,names);ax.axvline(0,color='black',lw=.8);ax.set_xlim(-3,3);ax.set_xlabel('Paired cases; native250, 32 shared initial states');ax.set_title('Different parameters, no net gain over strong nominal control');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(a.out/'figures/paired_effects.png',dpi=180);plt.close(fig)
    contact=list(csv.DictReader((a.audit/'CONTACT_DIAGNOSIS.csv').open()));fig,ax=plt.subplots(figsize=(7,4.7))
    for fail,color,label in [('0','#64748b','Grasp detected (515)'),('1','#b91c1c','Grasp not detected (28)')]:
        rows=[r for r in contact if r['grasp_failed']==fail];ax.scatter([float(r['tcp_target_error_mm']) for r in rows],[float(r['cube_moved_since_handoff_mm']) for r in rows],s=14,alpha=.55,c=color,label=label)
    ax.set_xlabel('Measured TCP target error at close [mm]');ax.set_ylabel('Cube displacement since handoff\n[mm; symlog scale]');ax.set_yscale('symlog',linthresh=1);ax.set_title('Planning success does not guarantee executable grasp');ax.legend();fig.tight_layout();fig.savefig(a.out/'figures/contact_diagnostics.png',dpi=180);plt.close(fig)
    # Selected actual episodes retain every native step, including all actors and joints.
    selected=[]
    selections={'physical32_contact_paired':{'fitted_value':{530028},'contract_contact':{530028}},'physical32':{'nominal':{530005,530022,530028},'fitted_value':{530028},'contract':{530028}},'physical32_contact':{'contract_contact':{530028}},'tracking_probe':{'original':{420002,420052},'full_C_approach_samples':{420002,420052}}}
    for group,methods in selections.items():
        for r in sets[group]['rows']:
            wanted=methods.get(r['method'],set())&set(r['seeds'])
            if not wanted:continue
            with gzip.open(a.run/group/r['trajectory_file'],'rt') as f:t=json.load(f)
            for seed in sorted(wanted):
                i=r['seeds'].index(seed);frames=[]
                for z in t['transitions']:
                    frames.append(dict(step=z['step'],segment=z['segments'][i],q=z['q'][i],qvel=z['qvel'][i],tcp=z['tcp'][i],action=z['action'][i],actors={k:v[i] for k,v in z['state']['actors'].items()}))
                selected.append(dict(seed=seed,method=r['method'],source_sha256=r['trajectory_sha256'],native_success=r['success'][i],error=r['errors'][i],decisions=r['decisions'][i],boundaries=r['boundaries'][i],frames=frames,plans=[z for z in t['plans'] if z['env']==i]))
    with gzip.open(a.out/'SELECTED_MEASURED_TRAJECTORIES.json.gz','wt') as f:json.dump(selected,f,separators=(',',':'))
    print(json.dumps(dict(total_steps=total,models=5,table=table,probe=probe),indent=2))
if __name__=='__main__':main()
