"""Callback-driven discrete options on frozen official Peg100 physical controller."""
import gzip,hashlib,inspect,json,random,time,types
from pathlib import Path
import numpy as np,torch,gymnasium as gym
import sapien,trimesh
from scipy.spatial.transform import Rotation
import program_peg_native as old
from program_discrete_options import OFFSETS,CONTROLLER,episode_transitions
from program_paired_skill_value import peg_features

class Runtime:
    def __init__(self,out,successor_mode="single",feedback=None,feedback_start=6):
        self.feedback_start=feedback_start
        self.feedback=feedback
        assert successor_mode in ("single", "split_refresh")
        self.successor_mode=successor_mode
        self.controller_id=CONTROLLER if successor_mode=="single" else CONTROLLER+"_split_refresh_option_v1"
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.rows=[];self.counter=0;self.callback=None
        self.env=old.Measure(gym.make('PegInsertionSide-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1))
        source=Path(inspect.getfile(old.official));code=source.read_text();self.source_sha=hashlib.sha256(code.encode()).hexdigest()
        code=code.replace('-max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01)','(-max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01)+choose_grasp(env))')
        code=code.replace('insert_pose = env.goal_pose * peg_init_pose.inv() * grasp_pose', 'insert_pose = env.goal_pose * peg_init_pose.inv() * grasp_pose\n    planner.original_insert_pose = insert_pose')
        mod=types.ModuleType('frozen_peg_options');exec(compile(code,str(source),'exec'),mod.__dict__);base_obb=mod.get_actor_obb
        def obb(actor,*args,**kw):
            box=base_obb(actor,to_world_frame=False);p=np.array(old.val(actor.pose.raw_pose[0]));T=np.eye(4);T[:3,:3]=Rotation.from_quat(np.r_[p[4:],p[3]]).as_matrix();T[:3,3]=p[:3]
            return trimesh.primitives.Box(extents=np.array(box.primitive.extents).copy(),transform=T@np.array(box.primitive.transform))
        mod.get_actor_obb=obb;owner=self
        class Planner(old.Planner):
            speed=2.;refresh_terminal=False;model=None
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                original=self.planner.plan_screw
                def measured(*aa,**kk):
                    st=time.monotonic();ret=original(*aa,**kk);owner.queries.append(dict(kind='plan_screw',wall_s=time.monotonic()-st,status=ret.get('status')if isinstance(ret,dict)else str(type(ret))));return ret
                self.planner.plan_screw=measured
            def move_to_pose_with_screw(self,pose,*a,**kw):
                if owner.feedback is not None and len(self.env.calls)>=owner.feedback_start and len(self.env.records)<100:return owner.feedback(self,pose,*a,**kw)
                if len(self.env.calls)==6 and len(self.env.records)<100:
                    obs=peg_features(self.env.measure(),100-len(self.env.records),1);action=owner.decide(obs);self.refresh_terminal=bool(action)
                    if action==2:
                        # Experimental additional option, shared authority for all
                        # methods: one midpoint, then one measured-target update.
                        # Original option IDs0/1 and native100 remain unchanged.
                        self.refresh_terminal=False;u=self.env.unwrapped
                        target=u.goal_pose.sp*u.peg.pose.sp.inv()*u.agent.tcp.pose.sp*sapien.Pose([0.05,0,0])
                        midpoint=sapien.Pose((u.agent.tcp.pose.sp.p+target.p)/2,target.q)
                        first=super().move_to_pose_with_screw(midpoint,*a,**kw)
                        if isinstance(first,int)and first==-1:return first
                        target=u.goal_pose.sp*u.peg.pose.sp.inv()*u.agent.tcp.pose.sp*sapien.Pose([0.05,0,0])
                        return super().move_to_pose_with_screw(target,*a,**kw)
                return super().move_to_pose_with_screw(pose,*a,**kw)
        def grasp(_):return OFFSETS[self.decide(peg_features(self.env.initial,100,0))]
        mod.choose_grasp=grasp;mod.PandaArmMotionPlanningSolver=Planner;self.solve=mod.solve
        (self.out/'SHARED_SOLVER.py').write_text(code);(self.out/'CONTROLLER.json').write_text(json.dumps(dict(id=self.controller_id,successor_mode=successor_mode,official_sha=self.source_sha,horizon=100,native_success=True,actions=[list(OFFSETS),['original','actual_attachment']],speed=2.),indent=2))
    def decide(self,obs):
        start=time.monotonic();a=int(self.callback(obs));stage=int(obs[44]);assert 0<=a<(3 if stage==0 or self.successor_mode=='split_refresh' else 2)
        self.actions.append(dict(stage=stage,step=len(self.env.records),id=a,obs=obs.tolist(),wall_s=time.monotonic()-start));return a
    def run(self,seed,callback,tag):
        self.counter+=1;self.callback=callback;self.actions=[];self.queries=[];error=None;start=time.monotonic();outer=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        try:
            try:
                ret=self.solve(self.env,seed=seed,vis=False);error='PLAN_FAILED'if isinstance(ret,int)and ret==-1 else None
            except old.Horizon:error='HORIZON'
            except Exception as ex:error=type(ex).__name__+':'+str(ex)
            e=self.env;e.phase='TERMINAL_HOLD';q=np.array(old.val(e.unwrapped.agent.robot.get_qpos()[0,:7]));grip=-1 if any(r['phase']=='CLOSE'for r in e.records)else 1
            while len(e.records)<100:e.step(np.r_[q,grip])
            row=dict(seed=int(seed),success=bool(e.unwrapped.evaluate()['success'][0]),error=error,steps=len(e.records),calls=e.calls,queries=self.queries,decisions=self.actions,tag=tag,wall_s=time.monotonic()-start,controller=self.controller_id)
            f=self.out/f'{self.counter:05d}_{seed}_{tag}.json.gz';d=dict(summary=row,initial=e.initial,steps=e.records)
            with gzip.open(f,'wt')as h:json.dump(d,h)
            row=dict(row,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest());self.rows.append(row);(self.out/'RESULTS.json').write_text(json.dumps(self.rows,indent=2))
            if not self.actions:raise RuntimeError('No option decision; inspect saved invalid episode '+str(f))
            if self.feedback is not None:return row,self.feedback_transitions(d,str(f)),d
            ts=episode_transitions(d,OFFSETS[self.actions[0]['id']],self.actions[-1]['id']if len(self.actions)>1 else 0,str(f))
            if self.successor_mode=='split_refresh':
                for t in ts:t['controller']=self.controller_id
                if len(ts)>1:ts[1]['a']=self.actions[-1]['id']
            # The actual callback and physically reconstructed boundary must agree.
            assert len(ts)==len(self.actions)
            for t,a in zip(ts,self.actions):assert t['a']==a['id'] and np.max(np.abs(np.array(t['s'])-a['obs']))<1e-6
            print('PHYSICS',tag,seed,row['success'],[a['id']for a in self.actions],error,round(row['wall_s'],2),flush=True)
            return row,ts,d
        finally:random.setstate(outer[0]);np.random.set_state(outer[1]);torch.set_rng_state(outer[2])
    def close(self):self.env.close()
