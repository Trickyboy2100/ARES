"""Scene-held-out covariance falsification on actual same-prefix terminal outcomes.
Nonparametric controls, not new RL gain. Identical neighbors and marginals isolate
whether independent hypothetical-label branches invent useful contrasts.
"""
import argparse,json
from pathlib import Path
import numpy as np
from program_discrete_options import prior

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();_,groups=prior(a.prior);pairs=[]
 for g in groups:
  if len(g['items'][0][2])!=2:continue
  pairs.append(dict(scene=g['seed'],grasp=g['grasp'],x=g['items'][0][2][1][0],y=[int(r['success'])for r,d,t in g['items']]))
 result=[]
 for q in pairs:
  train=[r for r in pairs if r['scene']!=q['scene']];x=np.array([r['x']for r in train]);sd=np.maximum(x.std(0),.05);dist=np.mean(((x-q['x'])/sd)**2,axis=1);ids=np.argsort(dist)[:5];counts=np.ones((2,2))*.25
  for i in ids:counts[tuple(train[i]['y'])]+=1
  joint=counts/counts.sum();m0=joint.sum(1);m1=joint.sum(0);ind=m0[:,None]*m1[None,:];y0,y1=q['y'];one=np.zeros((2,2));one[y0,y1]=1
  row=dict(scene=q['scene'],grasp=q['grasp'],outcomes=q['y'],neighbor_scenes=[train[i]['scene']for i in ids],joint=joint.tolist(),independent=ind.tolist())
  for name,pp in [('joint',joint),('independent',ind)]:
   row[name+'_nll']=float(-np.log(pp[y0,y1]));row[name+'_brier']=float(np.sum((pp-one)**2));row[name+'_discordance_probability']=float(pp[0,1]+pp[1,0]);conditional=pp[y0,1]/pp[y0,:].sum();row[name+'_conditional_brier']=float((conditional-y1)**2)
  result.append(row)
 summary={name:{metric:float(np.mean([r[name+'_'+metric]for r in result]))for metric in ['nll','brier','discordance_probability','conditional_brier']}for name in ['joint','independent']};summary.update(actual_discordance_rate=float(np.mean([r['outcomes'][0]!=r['outcomes'][1]for r in result])),contexts=len(result),scenes=len({r['scene']for r in result}),scope='leave-one-SCENE-out diagnostic; no physical or policy gain; no dev/sealed labels',matched_information='same five neighbors, same marginal probabilities, same Dirichlet smoothing; covariance only')
 a.out.write_text(json.dumps(dict(summary=summary,rows=result),indent=2));print(summary)
if __name__=='__main__':main()
