import sys,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np,torch
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
sys.path.insert(0,str(Path(__file__).resolve().parent))
from program_effect_value import Net,Learned,features,GRASPS,MODES
m=Learned(root/'models/MODEL.pt');torch.manual_seed(31132);one=Net(46,1);torch.manual_seed(31132);dyn=Net(46,45);torch.save(dict(scalar=one.state_dict(),dynamics=dyn.state_dict(),note='Deterministically reconstructed initial weights from retained constructor and seed31132, not a contemporaneously saved checkpoint.'),root/'models/INITIAL_RECONSTRUCTED.pt')
rr=json.loads((root/'evaluation/RESULTS.json').read_text());out=[]
for r in rr:
 d=json.load(gzip.open(r['path'],'rt'));valid=len(d['decisions'])>1
 if not valid:continue
 dec=d['decisions'][1];s=features(dec['observation'],100-dec['boundary_step'],r['mass'],0,1);p=float(torch.sigmoid(m.call('q1',s,[MODES.index(r['mode'])-1])[0,0]).detach());out.append(dict(method=r['method'],seed=r['seed'],mass=r['mass'],grasp=r['grasp'],mode=r['mode'],q1_chosen_probability=p,actual=r['success'],remaining_steps=100-dec['boundary_step']))
(root/'analysis/PREDICTIONS.json').write_text(json.dumps(out,indent=2))
from scipy.spatial.transform import Rotation
facts=[]
for r in json.loads((root/'tool_all_contacts/RESULTS.json').read_text()):
 d=json.load(gzip.open(r['path'],'rt'));s=d['steps'];b=d['skill_results'][0]['boundary_step'];sub=[z for x in s[b:] for z in x['physics_substep_contacts']];rot=Rotation.from_quat(np.asarray([x['tcp_T_tool'][3:] for x in s])[:,[1,2,3,0]]);ang=(rot[b-1].inv()*rot).magnitude()*180/np.pi
 first=lambda field:next((z['control_step'] for z in sub if z[field[0]][field[1]]>.5),None)
 t=first(('additional_forces_N','tool_table'));c=first(('force_norms_N','held_tool'));before=min(x for x in [t,c,101] if x is not None)
 facts.append(dict(grasp=r['grasp'],mode=r['mode'],success=r['success'],first_tool_table_step=t,first_tool_cube_step=c,max_rotation_before_first_table_or_cube_deg=float(ang[b:before-1].max()),max_rotation_after_boundary_deg=float(ang[b:].max()),terminal_distance_mm=s[-1]['cube_base_distance_m']*1000))
(root/'analysis/CONTACT_FACTS.json').write_text(json.dumps(facts,indent=2))
# Failed software probe is retained and excluded, even though its terminal holds consumed physics.
invalid=root/'endpoint/RESULTS.json'
if invalid.exists():
 attempts=json.loads(invalid.read_text())
 assert all('AttributeError' in str(r['error']) for r in attempts)
 (root/'endpoint/INVALID_RUN.json').write_text(json.dumps(dict(valid_scientific_episodes=0,invalid_logger_attempts=len(attempts),terminal_hold_steps=sum(r['steps'] for r in attempts),cause='Installed mplib get_link_pose returns array, logger assumed Pose; before first planned path execution. Fixed in endpoint_fixed; not task failure.'),indent=2))
