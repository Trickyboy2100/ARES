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
        audit_root=self.ares_r_root/"worklog/evidence/2026-09-21-p3-production-demo/block_final"
        self.audits={side:json.loads((audit_root/(side+"_fk_audit.json")).read_text())
                     for side in ("left","right")}
        audit=self.audits["right"]
        correction=np.asarray(audit["T_controller_model"],dtype=float)
        tool=audit["diagnostics"]["tool_data"]["pose_mm_rad"]
        from ares_r.perception.robot_collision import transform_xyz_rpy
        self.body_base=_world_arm_transform(self.world,"right")@correction
        self.body_bases={side:_world_arm_transform(self.world,side)@
                         np.asarray(self.audits[side]["T_controller_model"],dtype=float)
                         for side in ("left","right")}
        self.link6_tool=transform_xyz_rpy(np.asarray(tool[:3],dtype=float)*.001,tool[3:])
        self.tool_contract=("PINNED_URDF_WITH_2026-09-21_READONLY_FK_CORRECTION_"
                            "AND_TOOL2_NOT_CURRENT_LIVE_TCP")
        poses=json.loads((self.ares_r_root/"config/named_poses.json").read_text())["poses"]
        self.parked_left_q=tuple(poses["center"]["arms"]["left"]["joint_rad"])
        self.commissioned_predecessor_q=tuple(poses["folded"]["arms"]["right"]["ik_joint_rad"])
        self.context_coverage={
          "scene_cuboids":"TESTED_CPU_OBB",
          "inactive_left_arm":"TESTED_CPU_OBB_AT_RECORDED_PARKED_Q",
          "attached_object":"TESTED_CPU_ORIENTED_CUBOID_RESEARCH_PROXY",
          "central_exclusion":"TESTED_TCP_MARGIN",
          "chassis":"NOT_TESTED_FILTER_OWNED_CONSERVATIVE_PROXY_PENDING_MEASURED_CAD",
          "right_arm_self_collision":"NOT_TESTED_CPU"}
    def fk(self,q,side="right"):
        return self.body_bases[side]@arm_link_transforms(self.urdf,q)["link6"]@self.link6_tool
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
    def arm_boxes(self,q,side="right"):
        links=arm_link_transforms(self.urdf,q); result=[]
        for name,local in self.collision_model["arm_link_boxes"].items():
            result.append(_box_from_local(side+"/"+name,side+"_arm","arm_link",local,self.body_bases[side]@links[name],"pinned mesh AABB"))
        result.append(_box_from_local(side+"/gripper",side+"_tool","gripper_envelope",self.collision_model["gripper_max_envelope_link6"],self.body_bases[side]@links["link6"],"pinned EG2 envelope")); return result
    @staticmethod
    def obstacle_box(value):
        return CollisionBox(value["id"],"scene",value["kind"],tuple(value["center_body_m"]),((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),tuple(np.asarray(value["dims_m"])/2.),False,value.get("source","seeded CPU scenario"))
    def collision_codes(self,q,obstacles):
        codes=[]
        for rbox in self.arm_boxes(q):
            for sbox in map(self.obstacle_box,obstacles):
                if obb_overlap(rbox,sbox): codes.append("COLLISION:%s:%s"%(rbox.geometry_id,sbox.geometry_id))
        return sorted(set(codes))
    def attached_box(self,q,object_contract):
        pose=self.fk(q); offset=np.asarray(object_contract["tcp_offset_m"],dtype=float)
        center=pose[:3,:3]@offset+pose[:3,3]
        return CollisionBox("attached/"+object_contract["object_id"],"attached_object",
            "held_cuboid",tuple(center),tuple(tuple(row) for row in pose[:3,:3]),
            tuple(np.asarray(object_contract["dimensions_m"],dtype=float)/2.),False,
            object_contract["revision"])
    def fixed_context_boxes(self):
        result=[]
        for item in self.collision_model["fixed_body_boxes"]:
            # These boxes are self-filter geometry, not independently qualified
            # robot-environment collision geometry.  In particular the chassis
            # proxy overlaps ordinary commissioned link1/link2 states.
            if item["kind"]=="central_exclusion" or item.get("filter_owned"): continue
            result.append(CollisionBox(item["geometry_id"],item["owner"],item["kind"],
                tuple(item["center_body_m"]),((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),
                tuple(np.asarray(item["dims_m"],dtype=float)/2.),False,item["source"]))
        result.extend(self.arm_boxes(self.parked_left_q,"left")); return result
    def context_collision_codes(self,q,obstacles,object_contract=None):
        moving=self.arm_boxes(q,"right")
        if object_contract is not None: moving.append(self.attached_box(q,object_contract))
        static=[self.obstacle_box(item) for item in obstacles]+self.fixed_context_boxes()
        codes=[]
        for a in moving:
            for b in static:
                if a.geometry_id=="right/base_link" and b.geometry_id=="body/chassis":
                    continue
                if obb_overlap(a,b): codes.append("COLLISION:%s:%s"%(a.geometry_id,b.geometry_id))
        tcp_y=float(self.fk(q)[1,3]); central_margin=-tcp_y-.070
        if central_margin<=0: codes.append("CENTRAL_TCP_EXCLUSION_MARGIN:%.6f"%central_margin)
        return sorted(set(codes)),central_margin
    def singularity(self,q):
        base=self.fk(q); columns=[]; epsilon=1e-5
        for index in range(6):
            shifted=np.asarray(q,dtype=float).copy(); shifted[index]+=epsilon; pose=self.fk(shifted)
            linear=(pose[:3,3]-base[:3,3])/epsilon
            angular=Rotation.from_matrix(base[:3,:3].T@pose[:3,:3]).as_rotvec()/epsilon
            columns.append(np.r_[linear,angular])
        return float(np.linalg.svd(np.column_stack(columns),compute_uv=False)[-1])
    def dense_context_path_check(self,start,goal,obstacles,object_contract=None,samples=41):
        failures=[]; margins=[]
        for index,fraction in enumerate(np.linspace(0.,1.,samples)):
            q=np.asarray(start)+fraction*(np.asarray(goal)-np.asarray(start))
            if np.any(q<self.lower) or np.any(q>self.upper): failures.append("SOFT_LIMIT_AT_%d"%index)
            codes,margin=self.context_collision_codes(q,obstacles,object_contract); margins.append(margin)
            failures.extend("PATH_%d_%s"%(index,code) for code in codes)
        return {"clear":not failures,"failure_codes":sorted(set(failures)),
                "samples":samples,"minimum_central_margin_m":float(min(margins))}
    def dense_path_check(self,start,goal,obstacles,samples=25):
        failures=[]
        for index,fraction in enumerate(np.linspace(0.,1.,samples)):
            q=np.asarray(start)+fraction*(np.asarray(goal)-np.asarray(start))
            if np.any(q<self.lower) or np.any(q>self.upper): failures.append("SOFT_LIMIT_AT_%d"%index)
            failures.extend("PATH_%d_%s"%(index,code) for code in self.collision_codes(q,obstacles))
        return not failures,sorted(set(failures)),samples
