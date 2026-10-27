#!/usr/bin/env python3
"""Typeset only completed formal development evaluations from actual run index.

Camera re-evaluations and training-geometry checks are deliberately excluded.
No independent status means no populated success cell. Training is linked by
checkpoint SHA and is not charged again for repeated inference.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent

def tex(value):
    return str(value).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%')

def main():
    source=ROOT/'metrics/COMPLETED_RUNS_INDEX.json'
    index=json.loads(source.read_text())
    allruns=index['runs']
    rows=[]
    for run in allruns:
        if run.get('round') not in ['r1','r2','r3']: continue
        if run.get('split')!='development' or run.get('evaluation_cases')!=32: continue
        if run.get('audit_status')!='COMPLETE': continue
        name=run['run'].split('/',1)[1]
        # Formal trained-policy evaluations or explicit same-version feedback.
        if not name.endswith('_development') and not name.startswith('feedback32_'): continue
        policy=run.get('policy_sha256')
        training=[r for r in allruns if policy and r.get('policy_sha256')==policy
                  and r.get('online_training_interactions',0)>0]
        budgets={r['online_training_interactions'] for r in training}
        if policy and len(budgets)!=1:
            raise ValueError(f'Ambiguous training budget: {run["run"]}: {budgets}')
        budget=next(iter(budgets)) if budgets else 0
        label=re.sub(r'_seed\d+_development$','',name).removesuffix('_development')
        labels={'feedback32_zero_steering':'Feedback', 'feedback32_yaw_zero':'Feedback',
                'param_ppo':'Parameter PPO', 'effect_param_ppo':'Effect-history PPO',
                'yaw_param_ppo':'Yaw parameter PPO', 'yaw_effect_param_ppo':'Yaw + history PPO',
                'feedback32_recontact':'Re-contact feedback',
                'feedback32_old_best_no_recontact':'Feedback / nominal',
                'event_ordinary':'Event / ordinary PPO', 'event_active':'Event / active PPO',
                'event_qualified':'Event / active PPO',
                'fixed_ordinary':'Fixed / ordinary PPO', 'fixed_active':'Fixed / active PPO',
                'fixed_qualified':'Fixed / active PPO'}
        label=labels.get(label,label.replace('_',' '))
        rows.append({'round':run['round'].upper(),'method':label,'seed':run['seed'] if policy else '--',
                     'state_action':f'{run["observation_dim"]}/{run["action_dim"]}',
                     'training_interactions':budget,
                     'success':f'{run["independent_successes"]}/32',
                     'source_run':run['run'],'manifest_sha256':run['manifest_sha256'],
                     'checkpoint_sha256':policy})
    order={'Feedback':0,'Re-contact feedback':0,'Feedback / nominal':0,
           'Parameter PPO':1,'Effect-history PPO':2,
           'Yaw parameter PPO':1,'Yaw + history PPO':2,'Event / ordinary PPO':1,
           'Event / active PPO':2,'Fixed / ordinary PPO':3,'Fixed / active PPO':4}
    rows.sort(key=lambda row:(row['round'],order.get(row['method'],9),str(row['seed'])))
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',
           r'Round & Method & Seed & Obs./act. & Train interactions & Independent success \\',r'\midrule']
    for row in rows:
        lines.append(' & '.join(tex(row[k]) for k in ['round','method','seed','state_action','training_interactions','success'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    dest=ROOT/'tables/COMPLETED_ITERATIONS.tex'
    dest.write_text('\n'.join(lines)+'\n')
    meta={'input_path':str(source.relative_to(ROOT)),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'rows':rows,'table_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
          'limitations':['Single training seed per new method; not a multi-seed result',
                         'Rows across rounds use distinct physical task versions and duration contracts',
                         'Development scenes are not sealed final tests',
                         'Feedback zero training does not mean zero physical evaluation cost']}
    (ROOT/'tables/DOCUMENT_RESULTS_PROVENANCE.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'table':str(dest)},indent=2))

if __name__=='__main__':main()
