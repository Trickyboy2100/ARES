"""Isaac5.1/Lab2.3 single-env physical replay, never training or hardware.

Articulation reset writes occur ONCE. Every subsequent state is PhysX-produced;
commands follow an independently qualified archived path with offline retiming.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
a=p.parse_args();os.umask(0o077);a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device=f'cuda:{a.gpu}',enable_cameras=True,
    kit_args=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false')
app=launcher.app
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation,ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    from isaaclab.utils.math import quat_apply
    import omni.replicator.core as rep
    from PIL import Image
    torch.set_num_threads(2)
    meta=json.loads((a.asset/'SITE_ASSET_MANIFEST.json').read_text())
    path=np.load(a.asset/'qualified_transport.npz');q=path['joints_rad'];low=path['lower_rad'];high=path['upper_rad']
    assert hashlib.sha256((a.asset/'right_robot.usda').read_bytes()).hexdigest()==meta['robot_sha256']
    base=np.array(meta['body_base']);xyzw=Rotation.from_matrix(base[:3,:3]).as_quat()
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120,device=f'cuda:{a.gpu}',render_interval=16))
    sim.set_camera_view([2.0,-2.0,2.05],[.45,0,1.05])
    site=sim_utils.UsdFileCfg(usd_path=str((a.asset/'site.usda').resolve()))
    site.func('/World/Site',site)
    cfg=ArticulationCfg(prim_path='/World/Robot',
        spawn=sim_utils.UsdFileCfg(usd_path=str((a.asset/'right_robot.usda').resolve()),activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=32,solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(base[:3,3]),rot=(float(xyzw[3]),*map(float,xyzw[:3])),
            joint_pos={f'joint_{i+1}':float(v) for i,v in enumerate(q[0])}),
        actuators={'arm':ImplicitActuatorCfg(joint_names_expr=['joint_.*'],stiffness=1500.,damping=80.,
            effort_limit_sim=100.,velocity_limit_sim=.1)})
    robot=Articulation(cfg);sensor=ContactSensor(ContactSensorCfg(prim_path='/World/Robot/.*',history_length=1))
    light=sim_utils.DomeLightCfg(intensity=350.);light.func('/World/Light',light)
    sim.reset();robot.reset();sensor.reset()
    ids,_=robot.find_joints([f'joint_{i+1}' for i in range(6)],preserve_order=True)
    link=robot.find_bodies('Link_6')[0][0]
    pos=robot.data.default_joint_pos.clone();vel=robot.data.default_joint_vel.clone()
    pos[:,ids]=torch.tensor(q[0],device=sim.device,dtype=pos.dtype);vel.zero_()
    robot.write_joint_state_to_sim(pos,vel)  # ONLY initialization, never again.
    tool=np.array(meta['tool']);offset=torch.tensor(tool[:3,3],device=sim.device,dtype=torch.float32).repeat(1,1)
    camera=rep.create.camera(position=(2.,-2.,2.05),look_at=(.45,0,1.05))
    product=rep.create.render_product(camera,(960,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
    dt=1/120;duration=meta['native_timing']['duration_s'];native_dt=meta['native_timing']['sample_period_s']
    times=np.arange(len(q))*native_dt
    samples=[];frames=[];start=time.perf_counter();abort=None;initial_sim_time=sim.current_time
    for step in range(int((duration+1.)/dt)+1):
        time_s=step*dt
        command=np.array([np.interp(min(time_s,duration),times,q[:,i]) for i in range(6)])
        robot.set_joint_position_target(torch.tensor(command,device=sim.device,dtype=torch.float32).unsqueeze(0),joint_ids=ids)
        robot.write_data_to_sim();sim.step(render=False);robot.update(dt);sensor.update(dt,force_recompute=True)
        actual=robot.data.joint_pos[0,ids].cpu().numpy().copy()
        state=robot.data.body_link_state_w[0,link].cpu().numpy().copy()
        quat=state[3:7];rotation=Rotation.from_quat([*quat[1:],quat[0]]).as_matrix()
        tcp=state[:3]+rotation@tool[:3,3];tcp_rot=rotation@tool[:3,:3]
        level=np.degrees(np.arcsin(np.clip(max(abs(tcp_rot[2,2]),abs(tcp_rot[2,0])),0,1)))
        contact=float(torch.linalg.vector_norm(sensor.data.net_forces_w,dim=-1).max().cpu())
        soft=float(np.minimum(actual-low,high-actual).min())
        samples.append([time_s,*actual,*command,*tcp,level,contact,soft])
        if step%40==0:
            sim.render();pixels=np.asarray(rgb.get_data())
            if pixels.size and pixels.ndim==3:frames.append(pixels[:,:,:3].copy())
        if step%240==0:print(f'PHYSX_PROGRESS step={step} sim_time={sim.current_time} contact={contact:.3f}',flush=True)
        if not np.isfinite(actual).all() or soft<-.005 or contact>500.:
            abort='NONFINITE_OR_REFERENCE_SOFT_LIMIT_OR_RESEARCH_CONTACT_ABORT';break
    rows=np.array(samples)
    np.savez_compressed(a.output/'PHYSX_STATES.npz',rows=rows,
        columns=np.array(['time_s',*[f'q{i+1}' for i in range(6)],*[f'command_q{i+1}' for i in range(6)],
                          'tcp_x','tcp_y','tcp_z','level_deg','contact_net_N','soft_margin_rad']))
    if frames:
        Image.fromarray(frames[-1]).save(a.output/'SITE_PHYSX_LAST.png')
    final_tcp=rows[-1,13:16];error=float(np.linalg.norm(final_tcp-meta['goal_tcp_body_m']))
    result={'publication':'CONTROLLED_NOT_PUBLIC','isaac_sim':'5.1','isaac_lab':'2.3.0','physics_steps':len(rows),
        'physics_dt_s':dt,'simulated_s':len(rows)*dt,'sim_context_elapsed_s':sim.current_time-initial_sim_time,
        'wall_s':time.perf_counter()-start,'state_writes_after_initialization':0,
        'trajectory_source':'archived cuRobo transport + qualified offline timing, NOT policy training',
        'scope':meta['scope'],'source_request_sha256':meta['source_request_sha256'],'robot_asset_sha256':meta['robot_sha256'],
        'site_asset_sha256':meta['site_sha256'],'final_tcp_error_m':error,'max_level_error_deg':float(rows[:,16].max()),
        'max_net_contact_N':float(rows[:,17].max()),'minimum_reference_soft_margin_rad':float(rows[:,18].min()),
        'max_joint_tracking_error_rad':float(np.abs(rows[:,1:7]-rows[:,7:13]).max()),'abort':abort,
        'transport_segment_diagnostic_pass':bool(abort is None and error<.005 and rows[:,16].max()<3.5 and rows[:,18].min()>=-.002),
        'preland_success':False,'placement_success':False,'new_rl_training':False,
        'limitations':['field attachment is conservative AABB, not actual tray CAD','payload mass/inertia not measured',
            'net contact force is diagnostic, not full penetration/contact certification','left-arm collisions use frozen proxies',
            'planner scene boxes remain conservative, not recovered cabinet CAD'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'PHYSX_REPLAY_RESULT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    # PIL avoids media plugin discovery/network downloads inside a live Kit
    # process. Export only frames captured during this run, never a loop.
    if frames:
        images=[Image.fromarray(f).resize((640,480)).quantize(colors=128) for f in frames]
        images[0].save(a.output/'SITE_TRANSPORT_PHYSX.gif',save_all=True,append_images=images[1:],
                       duration=round(40*dt*1000),loop=1,optimize=False)
        (a.output/'VIDEO_MANIFEST.json').write_text(json.dumps({'source':'actual in-run PhysX frames',
            'frame_count':len(images),'physics_step_stride':40,'speed':'real simulated time',
            'no_training_claim':True,'filename':'SITE_TRANSPORT_PHYSX.gif'},indent=2))
finally:
    # No orchestrator.run() was started. Waiting for an attached annotator's
    # nonexistent workflow at shutdown can otherwise hold the GPU indefinitely.
    app.close(wait_for_replicator=False)
