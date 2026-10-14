"""Numerically validate actual PhysX held-object Jacobian against pinned URDF."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import arm_link_transforms
p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True)
p.add_argument('--audit',type=Path,required=True);p.add_argument('--request',type=Path,required=True);a=p.parse_args()
m=PinnedRobotModel(a.ares,a.ares_r);audit=json.loads(a.audit.read_text());r=json.loads(a.request.read_text())
q=np.array(audit['q']);h=np.array(r['attached_object_collision']['T_link6_object']);eps=1e-6
def fk(q):return m.body_base@arm_link_transforms(m.urdf,q)['link6']@h
t=fk(q);cols=[]
for i in range(6):
    shifted=q.copy();shifted[i]+=eps;pose=fk(shifted)
    cols.append(np.r_[(pose[:3,3]-t[:3,3])/eps,Rotation.from_matrix(pose[:3,:3]@t[:3,:3].T).as_rotvec()/eps])
error=float(np.abs(np.array(cols).T-np.array(audit['J_body_held'])).max())
result={'max_abs_error':error,'passed':error<1e-4,'finite_difference_epsilon_rad':eps,
    'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.audit.with_name('finite_difference_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
