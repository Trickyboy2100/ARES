"""Pinned-model FK/IK, site limits and conservative CPU collision checks."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ares_r.motion.tcp_orientation import level_rotation
from ares_r.perception.robot_collision import (CollisionBox, _box_from_local,
    _world_arm_transform, arm_link_transforms, obb_overlap)

class PinnedRobotModel:
    """BODY FK with site limits and a pinned-URDF tool proxy, not live TCP."""
    def __init__(self, ares_root: Path, ares_r_root: Path):
        self.ares_root=Path(ares_root); self.ares_r_root=Path(ares_r_root)
        self.urdf=self.ares_root/"isaac_sim/simforge/robot/jaka_minicobo_gripper.urdf"
        self.world=json.loads((self.ares_r_root/"config/robot_world.json").read_text())
        self.collision_model=json.loads((self.ares_r_root/"config/robot_collision_model.json").read_text())
        limits=json.loads((self.ares_r_root/"config/jaka_mini2_motion.site.json").read_text())
        margin=float(limits["soft_limit_margin_rad"])
        self.lower=np.asarray(limits["lower_rad"])+margin; self.upper=np.asarray(limits["upper_rad"])-margin
        audit_path=self.ares_r_root/"worklog/evidence/2026-09-21-p3-production-demo/block_final/right_fk_audit.json"
        audit=json.loads(audit_path.read_text())
        correction=np.asarray(audit["T_controller_model"],dtype=float)
        tool=audit["diagnostics"]["tool_data"]["pose_mm_rad"]
        from ares_r.perception.robot_collision import transform_xyz_rpy
        self.body_base=_world_arm_transform(self.world,"right")@correction
        self.link6_tool=transform_xyz_rpy(np.asarray(tool[:3],dtype=float)*.001,tool[3:])
        self.tool_contract=("PINNED_URDF_WITH_2026-09-21_READONLY_FK_CORRECTION_"
                            "AND_TOOL2_NOT_CURRENT_LIVE_TCP")
    def fk(self,q): return self.body_base@arm_link_transforms(self.urdf,q)["link6"]@self.link6_tool
    def solve(self,xyz,yaw,roll_branch,seeds,seed_value,attempts=28):
        xyz=np.asarray(xyz); target_r=level_rotation(yaw,roll_branch); rng=np.random.default_rng(seed_value)
        starts=[np.clip(np.asarray(q),self.lower,self.upper) for q in seeds]
        while len(starts)<attempts: starts.append(rng.uniform(self.lower,self.upper))
        answers=[]; keys=set()
        def residual(q):
            pose=self.fk(q); angular=Rotation.from_matrix(target_r.T@pose[:3,:3]).as_rotvec()
            return np.r_[pose[:3,3]-xyz,.18*angular]
        for initial in starts[:attempts]:
            ans=least_squares(residual,initial,bounds=(self.lower,self.upper),max_nfev=140,xtol=1e-9,ftol=1e-9,gtol=1e-9)
            pose=self.fk(ans.x); pe=float(np.linalg.norm(pose[:3,3]-xyz))
            oe=math.degrees(float(np.linalg.norm(Rotation.from_matrix(target_r.T@pose[:3,:3]).as_rotvec())))
            key=tuple(np.round(ans.x,4))
            if pe<=.002 and oe<=1.0 and key not in keys:
                keys.add(key); margin=float(np.min(np.r_[ans.x-self.lower,self.upper-ans.x])); answers.append((key,ans.x.copy(),pe,oe,margin))
        answers.sort(key=lambda row:(row[2]+math.radians(row[3]),-row[4])); return answers
    def arm_boxes(self,q):
        links=arm_link_transforms(self.urdf,q); result=[]
        for name,local in self.collision_model["arm_link_boxes"].items():
            result.append(_box_from_local("right/"+name,"right_arm","arm_link",local,self.body_base@links[name],"pinned mesh AABB"))
        result.append(_box_from_local("right/gripper","right_tool","gripper_envelope",self.collision_model["gripper_max_envelope_link6"],self.body_base@links["link6"],"pinned EG2 envelope")); return result
    @staticmethod
    def obstacle_box(value):
        return CollisionBox(value["id"],"scene",value["kind"],tuple(value["center_body_m"]),((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),tuple(np.asarray(value["dims_m"])/2.),False,value.get("source","seeded CPU scenario"))
    def collision_codes(self,q,obstacles):
        codes=[]
        for rbox in self.arm_boxes(q):
            for sbox in map(self.obstacle_box,obstacles):
                if obb_overlap(rbox,sbox): codes.append("COLLISION:%s:%s"%(rbox.geometry_id,sbox.geometry_id))
        return sorted(set(codes))
    def dense_path_check(self,start,goal,obstacles,samples=25):
        failures=[]
        for index,fraction in enumerate(np.linspace(0.,1.,samples)):
            q=np.asarray(start)+fraction*(np.asarray(goal)-np.asarray(start))
            if np.any(q<self.lower) or np.any(q>self.upper): failures.append("SOFT_LIMIT_AT_%d"%index)
            failures.extend("PATH_%d_%s"%(index,code) for code in self.collision_codes(q,obstacles))
        return not failures,sorted(set(failures)),samples
