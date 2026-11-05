"""Independent neural initialization on the identical shared nominal warm data."""
import argparse,json,torch,hashlib
from pathlib import Path
from program_budget_feedback_sac import FeedbackSAC,normalization
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);d=torch.load(a.source/'WARM_DATA.pt',weights_only=False);data=d['data'];mu,sd=normalization(data);m=FeedbackSAC('ordinary',a.seed,mu,sd,nobs=83,nactions=12);m.save(a.out/'RANDOM.pt');x=m.x([t['s']for t in data]);actions=torch.tensor([t['a']for t in data]);idx=torch.arange(len(data))
for _ in range(512):
 loss=-m.probs(x)[idx,actions].clamp_min(1e-9).log().mean();m.aopt.zero_grad();loss.backward();m.aopt.step()
m.save(a.out/'INITIAL.pt');(a.out/'MANIFEST.json').write_text(json.dumps(dict(seed=a.seed,source=str(a.source),source_data_sha=hashlib.sha256((a.source/'WARM_DATA.pt').read_bytes()).hexdigest(),actor_supervised_updates=512,critic_updates=0,new_physics_steps=0,shared_warm_data='same16actual nominal episodes, not independently recollected'),indent=2))
