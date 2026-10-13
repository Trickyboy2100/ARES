#!/usr/bin/env python3
"""Create a deterministic, explicitly UNTRAINED CASH V0 checkpoint."""
import argparse,json,hashlib
from pathlib import Path
import torch
from research.skill_handoff_lab.cash.model import CashActorCritic
from research.skill_handoff_lab.cash.trainer import save_checkpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261013); args=p.parse_args()
    torch.manual_seed(args.seed)
    model=CashActorCritic(state_dim=32,action_dim=16,hidden=96,effect_dim=8)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    metadata={"status":"UNTRAINED_ARCHITECTURE_INITIALIZATION_ONLY",
      "seed":args.seed,"state_dim":32,"action_dim":16,"hidden":96,
      "effect_dim":8,"scientific_result":False,
      "reason":"no executed cuRobo/Isaac outcome dataset while NVML is broken"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(args.output,model,optimizer,metadata)
    sha=hashlib.sha256(args.output.read_bytes()).hexdigest()
    (args.output.with_suffix(".json")).write_text(json.dumps(dict(metadata,sha256=sha),indent=2)+"\n")
    print(json.dumps(dict(metadata,sha256=sha),indent=2))
if __name__=="__main__": main()
