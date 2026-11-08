import sys,torch,copy,json
from pathlib import Path
sys.path.insert(0,'research/skill_handoff_lab/tools')
from witness_margin_learner import MarginWitnessLearner
from program_rule_option import Assets
from train_option_online import atomic_json
torch.set_num_threads(2);r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/continuation_retention_20261108');s=torch.load(r/'SHARED.pt',weights_only=False);m=MarginWitnessLearner(Assets(),32301);m.set_graph(torch.load(r/'WITNESS_GRAPH.pt',weights_only=False));m.set_evidence(s['pairs'],s['anchors'],s['reference_sha']);data=m.decorate(s['data'],-1000);log=m.train(data,40,0);print(log,flush=True);d=copy.deepcopy(m.state());m.train(data,2,0);n=MarginWitnessLearner(Assets(),32301);n.load(d);n.set_evidence(s['pairs'],s['anchors'],s['reference_sha']);n.train(data,2,0);assert all(torch.equal(v,n.actor.state_dict()[k])for k,v in m.actor.state_dict().items());atomic_json(r/'MARGIN_CPU_CHECK.json',dict(log=log,exact_resume=True,scope='CPU only, half reference margin predeclared; no physical result'));print('margin exact resume',flush=True)
