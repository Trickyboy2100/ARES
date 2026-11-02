"""Chosen stage-0 probability vs actual fresh development outcome; not calibration certification."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=[]
for method in ['supervised','stap_open','stap_replan','downstream']:
 rows=json.loads((a.root/'evaluation512'/method/'RESULTS.json').read_text())
 for r in rows:
  z=json.loads((a.root/'evaluation512'/method/f"{r['seed']}_POLICY.json").read_text())[0]
  pred=math.exp(max(map(max,z['log_values'])))if 'log_values'in z else 1/(1+math.exp(-max(z['logits'])))
  out.append(dict(method=method,seed=r['seed'],stage0_predicted_success=pred,actual_success=r['success'],error=r['error']))
(a.root/'analysis/PREDICTION_VS_PHYSICS.json').write_text(json.dumps(out,indent=2))
