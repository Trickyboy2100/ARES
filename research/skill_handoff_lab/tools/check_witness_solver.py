import sys,torch,json,copy,argparse
from pathlib import Path
sys.path.insert(0,'research/skill_handoff_lab/tools')
from witness_learner import WitnessLearner
from witness_qp_learner import WitnessQPLearner
from program_rule_option import Assets
from train_option_online import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists(),'Preserve prior numerical checks'
torch.set_num_threads(2);r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/continuation_retention_20261108');saved=r/'SOLVER_DIAGNOSTIC_INPUT_V2.pt';d=torch.load(saved if saved.exists()else r/'witness_project_32301/CURRENT.pt',weights_only=False);atomic_torch(saved,d)if not saved.exists()else None;sh=torch.load(r/'SHARED.pt',weights_only=False);out=[]
for cls,arm in [(WitnessLearner,'witness_project'),(WitnessQPLearner,'witness_qp')]:
 m=cls(Assets(),32301,arm);st=copy.deepcopy(d['model']);st['arm']=arm;m.load(st);m.set_evidence(sh['pairs'],sh['anchors'],sh['reference_sha']);log=m.train(d['data'],40,0);out.append(dict(arm=arm,log=log));print(out[-1],flush=True)
state=copy.deepcopy(m.state());m.train(d['data'],2,0);n=WitnessQPLearner(Assets(),32301);n.load(state);n.set_evidence(sh['pairs'],sh['anchors'],sh['reference_sha']);n.train(d['data'],2,0);assert all(torch.equal(v,n.actor.state_dict()[k])for k,v in m.actor.state_dict().items());assert all(torch.equal(v,n.q.state_dict()[k])for k,v in m.q.state_dict().items());print('QP exact resume passed',flush=True)
atomic_json(a.output,dict(scope='CPU same checkpoint/data/RNG optimizer diagnosis; no new physical episodes',results=out))
