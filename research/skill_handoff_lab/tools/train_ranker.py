"""Supervised terminal-candidate predictor from qualified real planning records.

Initial baseline fits PREPLACE outcomes only. Earlier-stage value propagation
requires complete branching observations; it is not manufactured here.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from research.skill_handoff_lab.cash.cache_backend import CachedPlanningBackend,observation
from research.skill_handoff_lab.cash.contracts import STAGES

def rows(backend):
    x=[];y=[];groups=[]
    for e in backend.episodes:
        for r in e["transitions"]:
            if r["downstream_success"] is None:continue
            o=observation(e["observations"][r["node"]])
            backend.episode=e; backend.node=r["node"]
            backend.execute(o,r["action_id"])
            action=next(a for a in o.actions if a.action_id==r["action_id"])
            x.append(o.robot_state+o.remaining_goal+o.successor_constraints+action.features)
            y.append(float(r["downstream_success"])); groups.append(e["scene_group"])
    if not x: raise ValueError("no qualified terminal outcome labels")
    return np.array(x,np.float32),np.array(y,np.float32),groups

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--epochs",type=int,default=80);a=p.parse_args()
    train=rows(CachedPlanningBackend(a.dataset,"train"));test=rows(CachedPlanningBackend(a.dataset,"test"))
    if len(set(train[1]))<2: raise ValueError("training labels lack success/failure variation")
    a.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed);torch.set_num_threads(2)
    mean=train[0].mean(0);scale=np.maximum(train[0].std(0),1e-6)
    x=torch.from_numpy((train[0]-mean)/scale);y=torch.from_numpy(train[1])
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.Tanh(),torch.nn.Linear(64,1))
    opt=torch.optim.Adam(model.parameters(),lr=3e-4);curve=[]
    for epoch in range(a.epochs):
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1),y)
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            tx=torch.from_numpy((test[0]-mean)/scale)
            probability=torch.sigmoid(model(tx).squeeze(-1)).numpy()
        curve.append({"epoch":epoch,"train_bce":float(loss.detach()),
            "test_brier":float(np.mean((probability-test[1])**2)),
            "test_accuracy":float(np.mean((probability>=.5)==test[1]))})
    torch.save({"model":model.state_dict(),"mean":mean,"scale":scale,"input_dim":x.shape[1]},a.output/"ranker.pt")
    (a.output/"curve.json").write_text(json.dumps(curve,indent=2)+"\n")
    (a.output/"manifest.json").write_text(json.dumps({"seed":a.seed,"train_rows":len(train[1]),
        "test_rows":len(test[1]),"dataset_sha256":hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
        "scope":"terminal candidate outcomes; earlier-stage propagation pending","evidence":"VALIDATED_PLANNING"},indent=2)+"\n")
if __name__=="__main__":main()
