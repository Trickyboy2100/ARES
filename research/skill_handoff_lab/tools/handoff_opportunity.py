"""Audit paired native physics; error location is not a causal attribution."""
import argparse,collections,json
from pathlib import Path
import numpy as np

def load(root):
    episodes={}
    for r in json.loads((root/'RESULTS.json').read_text())['rows']:
        a,c=int(r['policy'][1]),int(r['policy'][4])
        for i,s in enumerate(r['seeds']):
            episodes[s,a,c]=dict(seed=s,a=a,c=c,y=int(r['success'][i]),boundaries=r['boundaries'][i],decisions=r['decisions'][i],error=r['errors'][i])
    return episodes

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();e=load(args.source);seeds=sorted({s for s,a,c in e});args.out.mkdir(parents=True,exist_ok=True)
    def valid(r,stage):return any(b['stage']==stage and b['valid'] for b in r['boundaries'])
    y=np.array([[[e[s,a,c]['y'] for c in range(3)] for a in range(3)] for s in seeds]);av=np.array([[[valid(e[s,a,c],0) for c in range(3)] for a in range(3)] for s in seeds])
    assert np.all(av==av[:,:,:1]),'Prefix validity changed with suffix'
    evidence=[];errors=collections.Counter();counts=collections.Counter();mismatches=[]
    for k,r in e.items():
        A=valid(r,0);C=valid(r,1)
        category='native_success' if r['y'] else 'predecessor_not_verified' if not A else 'both_local_contracts_pass_native_fails' if C else 'successor_did_not_complete'
        counts[category]+=1
        if not r['y']:errors[str(r['error'])]+=1
        if C and not r['y']:mismatches.append(dict(seed=k[0],a=k[1],c=k[2],boundaries=r['boundaries']))
    for j,s in enumerate(seeds):
        good_a=[a for a in range(3) if av[j,a,1]]
        evidence.append(dict(seed=s,nominal=int(y[j,1,1]),oracle_A_Cnom=int(y[j,:,1].max()),oracle_C_Anom=int(y[j,1,:].max()),joint_oracle=int(y[j].max()),verified_A_different_suffix_outcomes=len({int(y[j,a,1]) for a in good_a})>1,successor_action_sensitive_prefixes=[a for a in good_a if len(set(y[j,a,:]))>1]))
    risks=[]
    for (s,a,c),r in e.items():
        for d in r['decisions']:
            st=d['stage']
            if st==0 and c!=1:continue
            baseline=e[s,1,1]['y'] if st==0 else e[s,a,1]['y']
            risks.append(dict(seed=s,stage=st,action=d['action'],risk=int(baseline and not r['y'])))
    riskstats={}
    for name,rr in [('all',risks),('non_reference',[r for r in risks if r['action']!=1])]:
        riskstats[name]=dict(tokens=len(rr),positive=sum(r['risk'] for r in rr),unique_positive_cases=len({r['seed'] for r in rr if r['risk']}),by_stage={st:dict(tokens=sum(r['stage']==st for r in rr),positive=sum(r['risk'] for r in rr if r['stage']==st)) for st in [0,1]})
    same=[]
    for s in seeds:
        for a in range(3):
            ds=[e[s,a,c]['decisions'] for c in range(3)]
            if all(len(d)==2 for d in ds):same.append(max(np.max(abs(np.array(ds[0][1]['observation'])-np.array(d[1]['observation']))) for d in ds))
    result=dict(scope='DESIGN data; finite tested-action hindsight ceiling, not policy performance',episodes=len(e),case_count=len(seeds),matrix=y.sum(0).tolist(),outcome_location=dict(counts),failure_errors=dict(errors),nominal=int(y[:,1,1].sum()),best_fixed=int(y.sum(0).max()),oracle_A_Cnom=int(y[:,:,1].max(1).sum()),oracle_C_Anom=int(y[:,1,:].max(1).sum()),joint_oracle=int(y.max((1,2)).sum()),oracle_C_by_fixed_A=y.max(2).sum(0).tolist(),risk=riskstats,identical_suffix_start_contexts=len(same),max_suffix_observation_difference=float(max(same)),cases=evidence,local_native_mismatches=mismatches)
    (args.out/'OPPORTUNITY.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['cases','local_native_mismatches']},indent=2))
if __name__=='__main__':main()
