"""Encode real compact-layout camera frames; never synthesize robot motion."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from build_paired_physx_delivery import Reader,font
from paired_physx_video_hooks import Encoder

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert (a.run/'RUN_COMPLETE.json').exists()
frames=json.loads((a.run/'MATRIX_LIVE_TRAINING_FRAMES.json').read_text())
events=json.loads((a.run/'LIVE_OPTIMIZER_EVENTS.json').read_text())['events']
assert len(events)==3 and all(e['parameters_changed'] for e in events)
assert events[-1]['online_interactions']==12288
patch=json.loads((a.run/'VIDEO_SOURCE_PATCH.json').read_text())
assert {'before':'env_spacing=2.5','after':'env_spacing=0.85'} in patch['patches']
source=a.run/'video_raw/matrix_live_training.mp4'
reader=Reader(source,1920,1080)
target=a.output/'actual_online_training_16env.mp4'
encoder=Encoder(target,'/usr/bin/ffmpeg',fps=12)
times=np.array([f['elapsed_physics_s'] for f in frames]);end=float(times[-1])
for n,t in enumerate(np.arange(0,end+3+1e-9,1/12)):
    index=min(len(frames)-1,max(0,int(np.searchsorted(times,min(t,end),side='right')-1)))
    f=frames[index];im=reader.get(index).copy();draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,1920,113),fill='#182e3c')
    draw.text((24,10),'REAL ONLINE TRAINING | 16 independent PhysX environments',font=font(32,True),fill='white')
    draw.text((24,50),f'Physical time {f["elapsed_physics_s"]:.2f}s | interactions {f["actual_env_control_steps"]:,} | rollout updates {f["optimizer_updates_completed"]}',font=font(25),fill='white')
    draw.text((24,83),'Compact 0.85 m layout + closer camera only | Fresh 12,288-interaction rerun, not formal performance.',font=font(21),fill='#b8d8dc')
    if t>end:
        draw.rectangle((0,1020,1920,1080),fill='#182e3c')
        draw.text((24,1035),'END / DISPLAY HOLD: no additional physics or policy updates',font=font(25),fill='white')
    if n in [24,96]:im.save(a.output/f'preview_{n:03d}.png')
    encoder.write(np.array(im))
encoder.close();reader.close()
record={'scope':'FRESH_ACTUAL_ONLINE_TRAINING_COMPACT_LAYOUT_RERUN_NOT_ORIGINAL_TRAJECTORY_OR_FORMAL_RESULT',
        'source_run':str(a.run.resolve()),'changed':['env_spacing 2.5m -> 0.85m','single-camera position/aim'],
        'unchanged':'16 environments; robot and object models; collision filtering; task/control/reward/termination; seed31; PPO settings; 12288 interaction budget',
        'trajectory_bitwise_identity_claimed':False,'events':events,'duration_seconds':(n+1)/12,
        'source_video_sha256':sha(source),'output_sha256':sha(target),
        'physical_method_AST_proof':patch['physical_policy_method_AST_identity'],
        'no_interpolation_or_motion_generation':True,'terminal_display_hold_seconds':3}
(a.output/'COMPACT_VIDEO_PROVENANCE.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'video':str(target),'sha256':sha(target),'frames':n+1}))
