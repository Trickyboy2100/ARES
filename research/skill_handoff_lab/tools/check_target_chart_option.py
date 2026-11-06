import sys,json,gzip,copy,numpy as np,torch
from pathlib import Path
from scipy.spatial.transform import Rotation
sys.path.insert(0,'research/skill_handoff_lab/tools')
from train_target_chart_option import Learner
from train_full_option import Learner as Flat
from program_rule_option import Assets
torch.set_num_threads(2);root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/action_frontier_20261107');assets=Assets();m=Learner(assets,32201);flat=Flat(assets,32201,'sac');static=Learner(assets,32201,'static');data=torch.load(root/'SHARED_FRONTIER.pt',weights_only=False)['data'];ss=[t['s']for t in data];mat=m.matrix(ss)
assert torch.all(torch.linalg.matrix_rank(mat)==13)
assert torch.equal(m.q[0][0].weight,flat.q[0][0].weight)
for s in ss[::10]:assert m.choose(s)==flat.choose(s)==static.choose(s)
errors=[];incomplete=0
for f in root.glob('design_*/physics/*.json.gz'):
 d=json.load(gzip.open(f,'rt'))
 for r in d['summary']['decisions']:
  k=r['id'];s=np.array(r['obs'])
  if r['stage']!=1 or not r.get('replanned') or k not in [1,2,3,4,5,8,9,10,11]:continue
  if "target" not in r:incomplete+=1;continue
  shift=np.zeros(3)
  if k in [2,3,8,9]:shift[1]=.003*(1 if k in [2,8] else -1)
  if k in [4,5,10,11]:shift[2]=.003*(1 if k in [4,10] else -1)
  if k in [1,8,9,10,11]:shift+=.05*s[77:80]
  q=s[31:35];actual=Rotation.from_quat(np.r_[q[1:],q[0]]).inv().apply(np.array(r['target'][:3])-r['active_target_before'][:3]);errors.append(float(np.max(np.abs(actual-shift))))
assert max(errors)<1e-6,max(errors)
log=m.train(m.decorate(copy.deepcopy(data[:50]),-1000),2,0)
assert np.isfinite(log['last']['actor'])
result=dict(states_rank_checked=len(ss),minimum_rank=13,common_initial_greedy=True,common_critic_initialization=True,target_geometry_replans_checked=len(errors),attempted_replans_without_returned_target=incomplete,max_target_coordinate_error_m=max(errors),optimizer_unit_updates=2,units=dict(translation_m=.003,rotation_deg=5),scope='same predecision observations; no new physics; not complete-Option outcome equivalence')
(root/'TARGET_CHART_CHECKS.json').write_text(json.dumps(result,indent=2));print(result)
