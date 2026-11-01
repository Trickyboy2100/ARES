"""Bounded8+4 native physical panel; other-axis selection never reads success.

Run via the repository's network-isolated launcher. Each episode is a fresh
process, native100, and all first8 must finish before second-action interventions.
"""
import argparse,gzip,json,os,subprocess,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',default='742010,742011,742012,742013');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    seeds=[int(x) for x in a.seeds.split(',')];script=Path(__file__).with_name('program_multiskill_two_robot.py');log=[]
    def run(seed,endpoint,label,axis='geometry'):
        path=a.out/label;cmd=[sys.executable,str(script),'--seed',str(seed),'--push-y',str(endpoint),'--stride','3','--controller','continuous-clear','--axis-objective','full-suffix','--grasp-axis',axis,'--out',str(path)]
        print('RUN',label,flush=True);start=time.monotonic()
        with (a.out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        row=dict(label=label,command=cmd,exit_code=r.returncode,wall_s=time.monotonic()-start,resource_context=os.environ.get('ARES_RESOURCE_CONTEXT','UNSPECIFIED'));log.append(row);(a.out/'QUEUE_ACCOUNTING.json').write_text(json.dumps(log,indent=2))
        if r.returncode:raise RuntimeError('Episode failed; preserve partial accounting and stop: '+label)
    for seed in seeds:
        for endpoint in [.02,.05]:run(seed,endpoint,f'{seed}_full_p{int(endpoint*1000):03d}')
    # Deliberately access only model decisions; no native success/failure labels.
    interventions=[]
    for seed in seeds:
        source=a.out/f'{seed}_full_p020'/'TRAJECTORY.json.gz'
        with gzip.open(source,'rt') as f:decisions=json.load(f)['decisions']
        second=next((d for d in decisions if d['skill']=='RIGHT_PICK_TO_GOAL'),None)
        if second is None:
            interventions.append(dict(seed=seed,source=str(source),status='NO_MODEL_SECOND_DECISION',action=None,reason='Cannot define other selected axis without an actual model decision; no extra physics'))
            continue
        selected=second['parameters']['grasp_axis'];other={'cube_x':'cube_y','cube_y':'cube_x'}[selected]
        interventions.append(dict(seed=seed,source=str(source),status='READY',model_selected_axis=selected,intervention_axis=other,boundary_step=second['boundary_step'],selection='opposite orthogonal axis chosen exclusively from model decision, not physical outcome'))
    (a.out/'INTERVENTION_SELECTION.json').write_text(json.dumps(interventions,indent=2))
    for r in interventions:
        if r['status']=='READY':run(r['seed'],.02,f"{r['seed']}_full_p020_other",r['intervention_axis'])
if __name__=='__main__':main()
