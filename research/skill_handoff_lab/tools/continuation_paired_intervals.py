"""Scene-cluster bootstrap conditional on the two actually trained networks."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();g=json.loads(a.summary.read_text())['groups'];rng=np.random.default_rng(3221108);out=[]
for left,right in [('witness_qp','witness_soft'),('witness_qp','sac'),('witness_soft','sac'),('retention','sac'),('ranking','sac')]:
 names=[f'confirm_{arm}_{seed}'for arm in [left,right]for seed in [32301,32302]]
 if not all(n in g and g[n]['episodes']==32 for n in names):continue
 cases=sorted(g[names[0]]['outcomes']);assert all(sorted(g[n]['outcomes'])==cases for n in names)
 diff=np.array([[int(g[f'confirm_{left}_{seed}']['outcomes'][s])-int(g[f'confirm_{right}_{seed}']['outcomes'][s])for s in cases]for seed in [32301,32302]])
 per_scene=diff.mean(0);boot=per_scene[rng.integers(len(cases),size=(20000,len(cases)))].mean(1);out.append(dict(left=left,right=right,mean_difference=float(per_scene.mean()),conditional_scene_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),per_network_differences=diff.mean(1).tolist(),scene_count=32,network_count=2,scope='Exploratory interval conditional on these two trained models; shared scenes clustered across seeds; not uncertainty over all training seeds or a confirmatory multiple-comparison claim'))
a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
