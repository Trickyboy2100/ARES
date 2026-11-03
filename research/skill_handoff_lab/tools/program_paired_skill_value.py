"""Matched-data skill-value experiment. Pair consistency is a hypothesis, not novelty.
Scene-group cross-validation on already observed DEVELOPMENT physics, never sealed.
Future outcomes are labels only. Every method sees exactly the same trajectories.
"""
import argparse,json,gzip,hashlib,time,copy,shutil
from pathlib import Path
import numpy as np,torch
from torch import nn
from program_contact_direction import transitions
class Net(nn.Module):
 def __init__(self,n=56,out=1):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,out))
 def forward(self,x):return self.net(x)
def load(root):
 groups=[]
 for folder in ['relative','near','design']:
  rows=json.loads((root/folder/'RESULTS.json').read_text());by={}
  for r in rows:
   assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(r['path'],'rt'));by.setdefault((r['seed'],r['grasp']),[]).append((r,d,transitions(d)))
  for key,items in by.items():
   if len(items)!=5:continue
   events=[x[1]['contact_event']for x in items]
   if events[0]:
    n=events[0]['step'];assert all(x[1]['steps'][:n]==items[0][1]['steps'][:n]for x in items),'NONMATCHED_PREFIX'
   groups.append(dict(seed=key[0],grasp=key[1],items=items))
 return groups

def peg_features(m,remaining,stage):
 from scipy.spatial.transform import Rotation
 tcp=np.array(m['tcp']);peg=np.array(m['peg']);goal=np.array(m['goal']);rot=lambda q:Rotation.from_quat(np.r_[q[1:],q[0]])
 attached=rot(tcp[3:]).inv().apply(peg[:3]-tcp[:3]);local=rot(goal[3:]).inv().apply(peg[:3]-goal[:3]);angle=(rot(goal[3:]).inv()*rot(peg[3:])).magnitude()
 return np.r_[m['q'][:7],m['qvel'][:7],tcp,peg,goal,m['half_sizes'],attached,remaining/100,float(m['grasped']),np.linalg.norm(local),stage,local,angle,np.zeros(5)].astype(np.float32)
def load_peg(root):
 names=['gpu_retime','gpu_refresh','grid_-0.01_nominal','grid_-0.01_refresh','grid_0.01_nominal','grid_0.01_refresh']+[f'new_{g}_{mode}'for g in ['-0.01','0','0.01']for mode in ['nominal','refresh']];by={}
 for name in names:
  folder=root/name;manifest=json.loads((folder/'MANIFEST.json').read_text());offset=manifest.get('grasp_offset',0.);refresh=manifest.get('refresh_terminal',name=='gpu_refresh');rows=json.loads((folder/'RESULTS.json').read_text())
  for r in rows:
   assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(r['path'],'rt'));boundary=next((c['start']for c in r['calls']if c['phase']=='INSERT'and c['start']<100),None);x=peg_features(d['initial'],100,0);z=peg_features(d['steps'][-1],0,1);a0=[offset/.01,0.];r=dict(r,action=[float(refresh),0.]);ts=[(x,z,a0,float(r['success']),True)]
   if boundary is not None:
    b=peg_features(d['steps'][boundary-1],100-boundary,1);ts=[(x,b,a0,0.,False),(b,z,r['action'],float(r['success']),True)]
   by.setdefault((r['seed'],offset),[]).append((r,d,ts,boundary))
 groups=[]
 for (seed,grasp),items in by.items():
  assert len(items)==2
  n=items[0][3]
  if n is not None:assert n==items[1][3]and items[0][1]['steps'][:n]==items[1][1]['steps'][:n]
  groups.append(dict(seed=seed,grasp=grasp,items=[x[:3]for x in items]))
 return groups

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['tool','peg'],default='tool');p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--methods',nargs='+',default=['fqi','paired','unpaired','supervised','foresight','stap_grid']);p.add_argument('--updates',type=int,default=600);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/'SOURCE.py');torch.set_num_threads(2);groups=(load_peg(a.root)if a.task=='peg'else load(a.root));scenes=sorted({g['seed']for g in groups});results=[];logs=[]
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task=a.task,question='Does within-identical-SkillResult return-difference supervision improve action ranking over absolute values, generic differences and downstream value?',methods=a.methods,folds=3,training_seeds=[31201,31202,31203],updates=a.updates,scope='historical development cross-validation, logged candidate execution outcomes, not new physical rollouts or original-method superiority',sources=[dict(path=r['path'],sha256=r['sha256'])for g in groups for r,_,_ in g['items']],split={str(s):i%3 for i,s in enumerate(scenes)}),indent=2))
 for fold in range(3):
  train=[g for g in groups if scenes.index(g['seed'])%3!=fold];test=[g for g in groups if scenes.index(g['seed'])%3==fold];ts=[t for g in train for _,_,tt in g['items']for t in tt];obs=np.array([t[0]for t in ts]);mu=obs.mean(0);sd=np.maximum(obs.std(0),.05)
  x1=[];y1=[];pairs=[];x0=[];reach=[];nexts=[];ends=[];group0=[]
  for g in train:
   ids=[]
   for r,d,tt in g['items']:
    x0.append(np.r_[(tt[0][0]-mu)/sd,tt[0][2]]);reach.append(len(tt)==2);ends.append(float(r['success']));group0.append((g['seed'],g['grasp']))
    nexts.append((tt[1][0]-mu)/sd if len(tt)==2 else np.zeros(54))
    if len(tt)==2:ids.append(len(x1));x1.append(np.r_[(tt[1][0]-mu)/sd,tt[1][2]]);y1.append(float(r['success']))
   pairs.extend((i,j)for i in ids for j in ids if i<j)
  X=torch.tensor(np.array(x1),dtype=torch.float32);Y=torch.tensor(y1,dtype=torch.float32)[:,None];Z=torch.tensor(np.array(x0),dtype=torch.float32);R=torch.tensor(reach,dtype=torch.float32)[:,None];E=torch.tensor(ends,dtype=torch.float32)[:,None];N=torch.tensor(np.array(nexts),dtype=torch.float32);pi=torch.tensor(pairs,dtype=torch.long);actions=torch.tensor([[0,0],[1,0]]if a.task=='peg'else [[0,0],[-1,0],[1,0],[0,-1],[0,1]],dtype=torch.float32)
  for seed in [31201,31202,31203]:
   for kind in a.methods:
    torch.manual_seed(seed);q1=Net();q0=Net();dyn=Net(out=54);opt=torch.optim.Adam(list(q1.parameters())+list(q0.parameters())+(list(dyn.parameters())if kind=='stap_grid'else []),lr=3e-4);start=time.monotonic();history=[];perm=torch.randperm(len(X));alt=pi.clone();alt[:,1]=perm[alt[:,1]]
    for it in range(a.updates):
     pred=q1(X).sigmoid();loss1=nn.functional.binary_cross_entropy_with_logits(q1(X),Y);loss=loss1
     if kind=='absolute_mixed':
      active=torch.zeros(len(X),dtype=torch.bool);discord=pi[(Y[pi[:,0],0]-Y[pi[:,1],0]).abs()>0];active[discord.flatten()]=True
      if active.any():loss=loss+nn.functional.binary_cross_entropy_with_logits(q1(X[active]),Y[active])
     if kind in ['paired','unpaired','paired_mixed']:
      ij=alt if kind=='unpaired'else pi
      if kind=='paired_mixed':ij=ij[(Y[ij[:,0],0]-Y[ij[:,1],0]).abs()>0]
      loss=loss+((pred[ij[:,0]]-pred[ij[:,1]]-(Y[ij[:,0]]-Y[ij[:,1]]))**2).mean()
     with torch.no_grad():
      qs=q1(torch.cat([N[:,None,:].expand(-1,len(actions),-1),actions[None].expand(len(N),-1,-1)],-1)).sigmoid().squeeze(-1)
      target=R*(qs.mean(1,keepdim=True)if kind=='foresight'else qs.max(1,keepdim=True).values)
      if kind=='supervised':target=E
      if kind=='stap_grid':target=R
     loss=loss+nn.functional.binary_cross_entropy_with_logits(q0(Z),target)
     if kind=='stap_grid':loss=loss+.1*((dyn(Z[R[:,0]>0])-N[R[:,0]>0])**2).mean()
     opt.zero_grad();loss.backward();opt.step()
     if (it+1)%200==0:
      history.append(dict(update=it+1,loss=float(loss),terminal_bce=float(loss1)));torch.save(dict(q0=q0.state_dict(),q1=q1.state_dict(),dyn=dyn.state_dict()if kind=='stap_grid'else None,mu=mu,sd=sd,fold=fold,seed=seed,kind=kind,updates=it+1),a.out/f'{kind}_f{fold}_s{seed}_{it+1}.pt')
    with torch.no_grad():
     for g in test:
      rr=g['items'];tt=rr[0][2];vals=[]
      if len(tt)==2:
       xx=torch.tensor(np.array([np.r_[(t[1][0]-mu)/sd,t[1][2]]for _,_,t in rr]),dtype=torch.float32);vals=q1(xx).sigmoid().ravel().tolist();idx=int(np.argmax(vals))
      else:vals=[0.]*len(rr);idx=0
      z=torch.tensor(np.r_[(tt[0][0]-mu)/sd,tt[0][2]],dtype=torch.float32)[None];before=float(q0(z).sigmoid()[0,0])
      if kind=='stap_grid':before*=float(q1(torch.cat([dyn(z).expand(len(actions),-1),actions],1)).sigmoid().max())
      yy=[float(r['success'])for r,_,_ in rr];nom=next(i for i,(r,_,_)in enumerate(rr)if r['action']==[0.,0.]);ranking=[int((vals[i]-vals[j])*(yy[i]-yy[j])>0)for i in range(len(rr))for j in range(i)if yy[i]!=yy[j]]
      results.append(dict(fold=fold,training_seed=seed,method=kind,scene=g['seed'],grasp=g['grasp'],reached=len(tt)==2,selected=idx,action=rr[idx][0]['action'],success=yy[idx],nominal=yy[nom],oracle=max(yy),q0=before,predictions=vals,outcomes=yy,rank_correct=sum(ranking),rank_total=len(ranking),selected_source=rr[idx][0]['path']))
    logs.append(dict(method=kind,fold=fold,seed=seed,updates=a.updates,physical_queries=0,training_episodes=sum(len(g['items'])for g in train),postcontact_rows=len(X),matched_pairs=len(pi),wall_s=time.monotonic()-start,history=history));(a.out/'TRAINING.json').write_text(json.dumps(logs,indent=2));(a.out/'PREDICTIONS.json').write_text(json.dumps(results,indent=2));print(fold,seed,kind,history[-1],flush=True)
 summary={}
 for kind in sorted({r['method']for r in results}):
  rr=[r for r in results if r['method']==kind];joint=[]
  for seed in [31201,31202,31203]:
   for scene in scenes:
    c=[r for r in rr if r['training_seed']==seed and r['scene']==scene];chosen=max(c,key=lambda x:x['q0']);joint.append(chosen['success'])
  summary[kind]=dict(contexts=len(rr),success=sum(r['success']for r in rr),nominal=sum(r['nominal']for r in rr),rescues=sum(r['success']>r['nominal']for r in rr),regressions=sum(r['success']<r['nominal']for r in rr),ranking_correct=sum(r['rank_correct']for r in rr),ranking_total=sum(r['rank_total']for r in rr),joint_scene_success=sum(joint),joint_scene_trials=len(joint))
 (a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
