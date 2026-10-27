"""Actual post-physics/pre-reset recording; cameras never advance physics.

Frames are encoded immediately to H264; only event keyframes are retained as
PNGs. Each policy runs in an independently cold-started Isaac process. These
hooks do not write joint/root state or change a controller/reward/termination.
"""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Encoder:
    def __init__(self,path,ffmpeg,width=1920,height=1080,fps=30):
        self.path=Path(path)
        command=[str(ffmpeg),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                 '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','ultrafast',
                 '-crf','21','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE);self.frames=0
    def write(self,rgb):
        self.process.stdin.write(np.ascontiguousarray(rgb[:,:,:3],dtype=np.uint8).tobytes());self.frames+=1
    def close(self):
        self.process.stdin.close();code=self.process.wait(timeout=90)
        if code:raise RuntimeError(f'Encoder failed {code}: {self.path}')

def configure(env,output,case_ids,ffmpeg,label,matrix=False,training_matrix=False):
    import publication_render_hooks as appearance
    appearance.configure(env,output)
    env.paired_video={'output':Path(output),'cases':list(case_ids),'ffmpeg':ffmpeg,'label':label,
                      'encoders':{},'records':{},'seen':set(),'ended':set(),'event_frames':{},'last_phase':{},'initial':{},'hook_sha256':sha(__file__),
                      'matrix_enabled':matrix or training_matrix,'matrix_records':[],'matrix_encoder':None,
                      'training_matrix':training_matrix,'optimizer_events':[],'learner':None,'training_states':[]}
    (Path(output)/'paired_video_hook_source.py').write_bytes(Path(__file__).read_bytes())
    (Path(output)/'appearance_hook_source.py').write_bytes(Path(appearance.__file__).read_bytes())
    (Path(output)/'video_raw').mkdir();(Path(output)/'video_keyframes').mkdir()

def parameter_hash(learner):
    h=hashlib.sha256()
    for key,tensor in sorted(learner.policy.state_dict().items()):
        h.update(key.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def training_physics_sample(env):
    if not hasattr(env,'paired_video') or not env.paired_video['training_matrix'] or env.evaluation:return
    import torch
    v=env.paired_video;paddle,obj,vel,goal=env.state()
    state=torch.cat((torch.full((env.num_envs,1),env.physics_steps,device=env.device),env.robot.data.joint_pos[:,env.ids],
        env.robot.data.joint_vel[:,env.ids],obj,env.cube.data.root_quat_w,vel,env.cube.data.root_ang_vel_w,paddle,
        env.phase[:,None],env.raw_actions,env.applied_delta,env.recontact_mode[:,None],env.remaining[:,None],
        env.parameter[:,None],env.held_yaw[:,None],env.held_angle[:,None],env.push_direction),dim=-1)
    v['training_states'].append(state.detach().cpu().numpy())

def _wide_training_pose(env):
    import torch
    target=env.scene.env_origins.mean(0)+env.center+torch.tensor([-.1,0.,-.15],device=env.device)
    eye=target+torch.tensor([5.2,-6.5,19.5],device=env.device)
    env.camera.set_world_poses_from_view(eye[None],target[None])

def _contact_sheet(env,index):
    import publication_render_hooks as appearance
    from PIL import Image,ImageDraw
    v=env.paired_video;tick=int(env.physics_steps);sheet=Image.new('RGB',(1920,1080));views=[]
    for i in range(env.num_envs):
        appearance._pose(env,f'single_{i}')
        for _ in range(3):env.sim.render()
        env.camera.update(env.physics_dt,force_recompute=True);assert int(env.physics_steps)==tick
        rgb=env.camera.data.output['rgb'][0].cpu().numpy()[:,:,:3];image=Image.fromarray(rgb)
        path=v['output']/'video_keyframes'/f'training_update{index:02d}_env{i:02d}_tick{tick:06d}.png';image.save(path)
        tile=image.resize((480,270));draw=ImageDraw.Draw(tile);draw.rectangle((0,0,480,21),fill='#203544');draw.text((6,4),f'LIVE Env {i:02d} | same tick {tick} | phase {int(env.phase[i])}',fill='white')
        sheet.paste(tile,((i%4)*480,(i//4)*270));views.append({'env':i,'source_png':str(path.relative_to(v['output'])),'sha256':sha(path),'state':env.physical_snapshot(i)})
    output=v['output']/'video_keyframes'/f'training_update{index:02d}_same_tick_16env_montage.png';sheet.save(output)
    record={'optimizer_update_index':index,'physics_tick':tick,'scope':'ONE_APPLICATION_SAME_PHYSICS_TICK_16_CAMERA_VIEW_MONTAGE_NOT_SINGLE_CAMERA','file':str(output.relative_to(v['output'])),'sha256':sha(output),'views':views}
    v.setdefault('training_montages',[]).append(record)

def bind_optimizer(learner,env):
    v=env.paired_video
    if not v['training_matrix']:return
    v['learner']=learner;v['initial_parameter_sha256']=parameter_hash(learner);original=learner.train
    v['original_train']=original;v['had_instance_train']='train' in learner.__dict__
    def observed_train(*args,**kwargs):
        index=len(v['optimizer_events'])+1;before=parameter_hash(learner);tick=int(env.physics_steps)
        _matrix_frame(env,force=True,event=f'PRE_OPTIMIZER_{index}')
        # The original first-rollout audit saves the learner. Do not let SB3
        # serialize this observation-only closure (which references PhysX).
        # Its class-defined train() remains authoritative during the call.
        learner.__dict__.pop('train',None)
        try:result=original(*args,**kwargs)
        finally:learner.train=observed_train
        after=parameter_hash(learner)
        assert int(env.physics_steps)==tick,'Optimizer/camera unexpectedly advanced physics'
        record={'optimizer_update_index':index,'physics_tick':tick,'online_interactions':int(learner.num_timesteps),
                'parameter_sha256_before':before,'parameter_sha256_after':after,'parameters_changed':before!=after,
                'ppo_epoch_counter_after':int(learner._n_updates),'original_train_called_exactly_once':True}
        v['optimizer_events'].append(record);_matrix_frame(env,force=True,event=f'POST_OPTIMIZER_{index}')
        if index in [1,3]:_contact_sheet(env,index)
        (v['output']/'LIVE_OPTIMIZER_EVENTS.json').write_text(json.dumps({'events':v['optimizer_events'],'initial_parameter_sha256':v['initial_parameter_sha256']},indent=2))
        return result
    learner.train=observed_train

def unbind_optimizer(learner,env):
    v=env.paired_video
    if not v['training_matrix']:return
    if v['had_instance_train']:learner.train=v['original_train']
    else:learner.__dict__.pop('train',None)

def _matrix_frame(env,force=False,event='CADENCE'):
    import publication_render_hooks as appearance
    from PIL import Image
    v=env.paired_video
    training=v['training_matrix']
    if not v['matrix_enabled'] or (training and env.evaluation) or (not training and len(v['matrix_records'])>=73):return
    tick=int(env.physics_steps)
    if not force and v['matrix_records'] and tick-v['matrix_records'][-1]['physics_tick']<20:return
    _wide_training_pose(env) if training else appearance._pose(env,'array')
    for _ in range(3):env.sim.render()
    env.camera.update(env.physics_dt,force_recompute=True)
    assert tick==int(env.physics_steps)
    rgb=env.camera.data.output['rgb'][0].cpu().numpy()[:,:,:3]
    if v['matrix_encoder'] is None:
        v['matrix_file']='video_raw/matrix_live_training.mp4' if training else 'video_raw/matrix_live_evaluation.mp4'
        v['matrix_encoder']=Encoder(v['output']/v['matrix_file'],v['ffmpeg'],fps=6)
        v['matrix_start_tick']=tick
    index=len(v['matrix_records']);v['matrix_encoder'].write(rgb)
    record={'frame_index':index,'physics_tick':tick,'elapsed_physics_s':(tick-v['matrix_start_tick'])*env.physics_dt,
            'scope':'ACTUAL_VECTOR_ENV_ONLINE_TRAINING_RERUN' if training else 'ACTUAL_VECTOR_ENV_EVALUATION_NOT_TRAINING','num_envs':env.num_envs,
            'capture_event':event,'optimizer_updates_completed':len(v['optimizer_events']),
            'online_interactions':int(v['learner'].num_timesteps) if v['learner'] else 0,
            'actual_env_control_steps':int(env.physics_steps//env.cfg.decimation*env.num_envs),
            'parameter_sha256':parameter_hash(v['learner']) if v['learner'] else None,
            'environments':[env.physical_snapshot(i) for i in range(env.num_envs)]}
    if index in [0,18,36,60] or event!='CADENCE':
        path=v['output']/'video_keyframes'/f'matrix_live_{"train" if training else "eval"}_frame{index:03d}_tick{tick:06d}.png';Image.fromarray(rgb).save(path)
        record.update(png=str(path.relative_to(v['output'])),png_sha256=sha(path))
    v['matrix_records'].append(record)

def _pose(env,index):
    import torch
    origin=env.scene.env_origins[index]
    target=origin+env.center+torch.tensor([.093,-.014,-.010],device=env.device)
    eye=target+torch.tensor([.25,-.34,.31],device=env.device)
    env.camera.set_world_poses_from_view(eye[None],target[None])

def _frame(env,case,index,event):
    from PIL import Image
    v=env.paired_video;out=v['output'];_pose(env,index)
    before=int(env.physics_steps)
    # Render-only settling after camera movement. No sim.step() or state write.
    for _ in range(3):env.sim.render()
    env.camera.update(env.physics_dt,force_recompute=True)
    assert int(env.physics_steps)==before
    rgb=env.camera.data.output['rgb'][0].cpu().numpy()[:,:,:3]
    if case not in v['encoders']:
        v['encoders'][case]=Encoder(out/'video_raw'/f'case_{case:03d}.mp4',v['ffmpeg'])
        v['records'][case]=[];v['event_frames'][case]={}
    encoder=v['encoders'][case];snapshot=env.physical_snapshot(index)
    if hasattr(env,'recontact_attempts'):
        snapshot.update(controller_reset_object_xy=env.start_object_xy[index].cpu().tolist(),
            controller_peak_displacement_m=float(env.object_displacement[index]),
            controller_held_parameters=[float(env.parameter[index]),float(env.held_yaw[index]),float(env.held_angle[index])],
            controller_execution_mask=env.action_mask()[index].cpu().tolist(),
            controller_recontact_state=[int(env.recontact_mode[index]),int(env.recontact_attempts[index]),int(env.recontact_cooldown[index]),int(env.remaining[index]),bool(env.refresh_event[index])],
            previous_effect_history=env.effect[index].cpu().tolist(),
            previous_contact_response_memory=[float(env.prior_wz[index]),float(env.prior_vt[index]),float(env.prior_contact[index])])
    q=snapshot['object_quaternion_wxyz'];yaw=float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))
    phi=float(np.arctan2(float(env.geometry[index,6]),float(env.geometry[index,5])))
    goal=env.goals[index,1].cpu().numpy();pos=np.array(snapshot['object_position']);lat=np.array([-np.sin(phi),np.cos(phi)])
    forces=np.array(snapshot['robot_contact_force'])
    record=dict(snapshot,frame_index=encoder.frames,capture_event=event,source='CURRENT_POST_PHYSICS_PRE_RESET',
                elapsed_physics_s=float(snapshot['control_step']*env.step_dt),
                final_goal=goal.tolist(),yaw_error_rad=float(np.arctan2(np.sin(4*(yaw-phi)),np.cos(4*(yaw-phi)))/4),
                lateral_error_m=float((pos[:2]-goal[:2])@lat),final_goal_distance_m=float(np.linalg.norm(pos[:2]-goal[:2])),
                max_robot_net_contact_N=float(np.linalg.norm(forces,axis=-1).max()),
                applied_delta_m=env.applied_delta[index].cpu().tolist())
    encoder.write(rgb);v['records'][case].append(record)
    phase=int(snapshot['phase']);prior=v['last_phase'].get(case,phase);v['last_phase'][case]=phase
    events=[]
    if event=='INITIAL':events.append('initial');v['initial'][case]=snapshot
    if 'first_contact' not in v['event_frames'][case] and record['max_robot_net_contact_N']>.02:events.append('first_contact')
    if phase!=prior:events.append('stage_switch')
    if 'deviation' not in v['event_frames'][case] and abs(record['yaw_error_rad'])>np.radians(8):events.append('deviation')
    if event=='TERMINAL':events.append('terminal')
    for name in events:
        p=out/'video_keyframes'/f'case_{case:03d}_{name}_tick{before:06d}.png'
        Image.fromarray(rgb).save(p);v['event_frames'][case][name]={'file':str(p.relative_to(out)),'frame_index':record['frame_index'],'physics_tick':before,'sha256':sha(p)}

def initial(env):
    if not hasattr(env,'paired_video'):return
    if not env.evaluation:
        if env.paired_video['training_matrix']:_matrix_frame(env)
        return
    for i,case in enumerate(env.cases.cpu().tolist()):
        if int(case) in env.paired_video['cases'] and int(case) not in env.paired_video['initial'] and int(env.episode_length_buf[i])==0:
            _frame(env,int(case),i,'INITIAL')
    _matrix_frame(env)

def post_step(env,done):
    if not hasattr(env,'paired_video'):return
    if not env.evaluation:
        if env.paired_video['training_matrix']:_matrix_frame(env)
        return
    v=env.paired_video
    for i,case0 in enumerate(env.cases.cpu().tolist()):
        case=int(case0)
        if case not in v['cases'] or case in v['ended']:continue
        terminal=bool(done[i])
        if int(env.episode_length_buf[i])%2==0 or terminal or int(env.phase[i])!=v['last_phase'].get(case,int(env.phase[i])):
            _frame(env,case,i,'TERMINAL' if terminal else 'POST_STEP')
        if terminal:v['ended'].add(case)
    _matrix_frame(env)

def finalize(env,output):
    if not hasattr(env,'paired_video'):return
    v=env.paired_video;out=Path(output);results={int(x['case_id']):x for x in env.records}
    for encoder in v['encoders'].values():encoder.close()
    if v['matrix_encoder'] is not None:
        v['matrix_encoder'].close()
        (out/('MATRIX_LIVE_TRAINING_FRAMES.json' if v['training_matrix'] else 'MATRIX_LIVE_EVALUATION_FRAMES.json')).write_text(json.dumps(v['matrix_records'],indent=2))
    if v['training_matrix']:
        (out/'LIVE_TRAINING_MONTAGES.json').write_text(json.dumps(v.get('training_montages',[]),indent=2))
        columns=['physics_tick',*[f'q{i}' for i in range(6)],*[f'qd{i}' for i in range(6)],
                 'obj_x','obj_y','obj_z','obj_qw','obj_qx','obj_qy','obj_qz','vx','vy','vz','wx','wy','wz',
                 'paddle_x','paddle_y','paddle_z','phase',*[f'submitted_action_{i}' for i in range(env.raw_actions.shape[1])],
                 'applied_reference_dx','applied_reference_dy','recontact_mode','hold_remaining','held_contact_m','held_yaw_rad','held_approach_rad','push_dir_x','push_dir_y']
        states=np.stack(v['training_states']);assert states.shape[-1]==len(columns)
        np.savez_compressed(out/'TRAINING_120HZ_STATE.npz',states=states,columns=np.array(columns),sample_dt_s=env.physics_dt,
                            scope='ACTUAL_PRE_PHYSICS_INTEGRATION_AFTER_REFERENCE_UPDATE; TRAINING ONLY; NOT INTERPOLATION')
    if set(v['cases'])!=v['ended']:raise RuntimeError(f"Missing terminal recordings: {set(v['cases'])-v['ended']}")
    files={}
    for case,records in v['records'].items():
        p=out/'video_raw'/f'case_{case:03d}_frames.json';p.write_text(json.dumps(records,indent=2))
        mp4=out/'video_raw'/f'case_{case:03d}.mp4'
        files[str(case)]={'mp4':str(mp4.relative_to(out)),'mp4_sha256':sha(mp4),'frame_records':str(p.relative_to(out)),'frame_records_sha256':sha(p),
                          'frames':len(records),'capture_fps':30,'encoded_fps':30,'initial_state':v['initial'][case],
                          'terminal_episode':results[case],'events':v['event_frames'][case],
                          'duration_note':'Nominal 30 Hz capture plus exact handoff/terminal events. Raw encoded frame index is not the physical clock when event frames are inserted; frame_records is authoritative. Paired composition resamples using recorded elapsed_physics_s.'}
    proof={'scope':'SYNTHETIC ACTUAL CLOSED-LOOP PHYSX; NO STATE PLAYBACK','policy_label':v['label'],'cases':files,
           'camera_size':[1920,1080],'camera_pose':'Task-centered fixed oblique, per-env origin only',
           'render_does_not_step_physics':True,'camera_changes_are_appearance_only':True,'hook_sha256':v['hook_sha256'],
           'state_writes_during_actions':0,'physics_trace_sha256':sha(out/'actual_physics_trace.npz'),
           'action_log_sha256':sha(out/'action_log.json'),'task_source_sha256':sha(out/'task_source.py'),
           'raw_source_camera':'Isaac Lab 2.3 Camera RGB actual post-step before reset',
           'training_120hz_sidecar_sha256':sha(out/'TRAINING_120HZ_STATE.npz') if v['training_matrix'] else None,
           'matrix_overview':{'enabled':v['matrix_enabled'],'is_training':v['training_matrix'],'frames':len(v['matrix_records']),
              'mp4':v.get('matrix_file'),'mp4_sha256':sha(out/v['matrix_file']) if v['matrix_enabled'] else None,
              'scope':'One actual vectorized application; nominal 6Hz plus optimizer events, exact timestamps authoritative; no extra physics stepping'},
           'limitations':['No real robot/field/real tray/grasp claim','No equality claim for hidden PhysX solver/cache state','Contact is simulated net-force proxy, not calibrated force']}
    (out/'VIDEO_CAPTURE_PROVENANCE.json').write_text(json.dumps(proof,indent=2))
