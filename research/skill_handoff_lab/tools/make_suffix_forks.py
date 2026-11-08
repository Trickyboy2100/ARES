"""Reconstruct immutable checkpoint32/replay forks; never overwrite evidence."""
import argparse,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
for mode in ['sac','trace']:
 dst=a.out/f'FORK32_{mode}.pt';assert not dst.exists()
 d=torch.load(a.root/f'{mode}_32401/CURRENT.pt',weights_only=False);model=torch.load(a.root/f'{mode}_32401/CHECKPOINT_32.pt',weights_only=False)
 data=[t for t in d['data']if 'mu'not in t or t['policy_episode']<32]
 for t in data:
  if 'successor_index'in t:assert t['successor_index']<len(data)
 torch.save(dict(model=model,data=data),dst)
