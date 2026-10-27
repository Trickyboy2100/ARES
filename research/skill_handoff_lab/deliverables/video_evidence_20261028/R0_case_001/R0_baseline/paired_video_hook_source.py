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
    def __init__(self,path,ffmpeg,width=1920,height=1080):
        self.path=Path(path)
        command=[str(ffmpeg),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                 '-s',f'{width}x{height}','-r','30','-i','-','-an','-c:v','libx264','-preset','ultrafast',
                 '-crf','21','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE);self.frames=0
    def write(self,rgb):
        self.process.stdin.write(np.ascontiguousarray(rgb[:,:,:3],dtype=np.uint8).tobytes());self.frames+=1
    def close(self):
        self.process.stdin.close();code=self.process.wait(timeout=90)
        if code:raise RuntimeError(f'Encoder failed {code}: {self.path}')

def configure(env,output,case_ids,ffmpeg,label):
    import publication_render_hooks as appearance
    appearance.configure(env,output)
    env.paired_video={'output':Path(output),'cases':list(case_ids),'ffmpeg':ffmpeg,'label':label,
                      'encoders':{},'records':{},'seen':set(),'ended':set(),'event_frames':{},'last_phase':{},'initial':{}}
    (Path(output)/'video_raw').mkdir();(Path(output)/'video_keyframes').mkdir()

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
    if not hasattr(env,'paired_video') or not env.evaluation:return
    for i,case in enumerate(env.cases.cpu().tolist()):
        if int(case) in env.paired_video['cases']:_frame(env,int(case),i,'INITIAL')

def post_step(env,done):
    if not hasattr(env,'paired_video') or not env.evaluation:return
    v=env.paired_video
    for i,case0 in enumerate(env.cases.cpu().tolist()):
        case=int(case0)
        if case not in v['cases'] or case in v['ended']:continue
        terminal=bool(done[i])
        if int(env.episode_length_buf[i])%2==0 or terminal:
            _frame(env,case,i,'TERMINAL' if terminal else 'POST_STEP')
        if terminal:v['ended'].add(case)

def finalize(env,output):
    if not hasattr(env,'paired_video'):return
    v=env.paired_video;out=Path(output);results={int(x['case_id']):x for x in env.records}
    for encoder in v['encoders'].values():encoder.close()
    if set(v['cases'])!=v['ended']:raise RuntimeError(f"Missing terminal recordings: {set(v['cases'])-v['ended']}")
    files={}
    for case,records in v['records'].items():
        p=out/'video_raw'/f'case_{case:03d}_frames.json';p.write_text(json.dumps(records,indent=2))
        mp4=out/'video_raw'/f'case_{case:03d}.mp4'
        files[str(case)]={'mp4':str(mp4.relative_to(out)),'mp4_sha256':sha(mp4),'frame_records':str(p.relative_to(out)),'frame_records_sha256':sha(p),
                          'frames':len(records),'capture_fps':30,'encoded_fps':30,'initial_state':v['initial'][case],
                          'terminal_episode':results[case],'events':v['event_frames'][case],
                          'duration_note':'30fps physical cadence with exact extra terminal frame; final timestamp quantization at most one output frame.'}
    proof={'scope':'SYNTHETIC ACTUAL CLOSED-LOOP PHYSX; NO STATE PLAYBACK','policy_label':v['label'],'cases':files,
           'camera_size':[1920,1080],'camera_pose':'Task-centered fixed oblique, per-env origin only',
           'render_does_not_step_physics':True,'camera_changes_are_appearance_only':True,'hook_sha256':sha(__file__),
           'state_writes_during_actions':0,'physics_trace_sha256':sha(out/'actual_physics_trace.npz'),
           'action_log_sha256':sha(out/'action_log.json'),'task_source_sha256':sha(out/'task_source.py'),
           'raw_source_camera':'Isaac Lab 2.3 Camera RGB actual post-step before reset',
           'limitations':['No real robot/field/real tray/grasp claim','No equality claim for hidden PhysX solver/cache state','Contact is simulated net-force proxy, not calibrated force']}
    (out/'VIDEO_CAPTURE_PROVENANCE.json').write_text(json.dumps(proof,indent=2))
