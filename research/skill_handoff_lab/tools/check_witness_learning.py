import sys,torch,json,time,copy
from pathlib import Path
sys.path.insert(0,'research/skill_handoff_lab/tools')
from witness_graph import build_graph
from witness_learner import WitnessLearner
from program_rule_option import Assets
from train_option_online import atomic_torch,atomic_json
torch.set_num_threads(2);r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/continuation_retention_20261108');tic=time.monotonic();g=build_graph([r/'design_a',r/'design_b']);assert torch.load(r/'WITNESS_GRAPH.pt',weights_only=False)==g if (r/'WITNESS_GRAPH.pt').exists()else True
if not (r/'WITNESS_GRAPH.pt').exists():atomic_torch(r/'WITNESS_GRAPH.pt',g);print('graph',len(g['nodes']),'seconds',time.monotonic()-tic,flush=True)
shared=torch.load(r/'SHARED.pt',weights_only=False);m=WitnessLearner(Assets(),32301,'witness_project');m.set_evidence(shared['pairs'],[],shared['reference_sha']);m.set_graph(g);data=m.decorate(shared['data'],-1000);log=m.train(data,40,0);print(log,flush=True);assert m._valid();save=copy.deepcopy(m.state());m.train(data,2,0);n=WitnessLearner(Assets(),32301,'witness_project');n.load(save);n.set_evidence(shared['pairs'],[],shared['reference_sha']);n.set_graph(g);n.train(data,2,0);assert all(torch.equal(v,n.actor.state_dict()[k])for k,v in m.actor.state_dict().items());atomic_json(r/'WITNESS_CPU_CHECK.json',dict(nodes=len(g['nodes']),log=log,exact_resume=True,scope='CPU checks on measured training data, no native deployment success claimed'));print('exact resume',flush=True)

m._before(None,None,None)
k=int(torch.where(m.gm[0]&~m.ga[0])[0][0])
with torch.no_grad():m.actor[-1].bias[3+k]+=20
assert not m._valid()
m._after(None,None,None)
assert m._valid() and (m.projections[-1]['cuts']>0 or m.projections[-1]['scale']<1)
atomic_json(r/'WITNESS_PROJECT_STRESS.json',dict(projection=m.projections[-1],scope='adversarial parameter displacement on recorded training states; no physical success claim'))
print('nonlinear projection branch verified',m.projections[-1],flush=True)
