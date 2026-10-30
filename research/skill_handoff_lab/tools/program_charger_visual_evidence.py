"""Protocol RAW/ANNOTATED pose render; projected measured centers, zero physics steps."""
import argparse,copy,json
from pathlib import Path
import numpy as np,torch,gymnasium as gym
import mani_skill.envs
from mani_skill.utils import sapien_utils
from PIL import Image,ImageDraw,ImageFont
from program_charger_mechanism import load,sha,arrays,mechanism

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
 for seed in [725002,725005]:
  out=a.out/str(seed);out.mkdir(parents=True,exist_ok=True);pairs=[load(a.root,v,seed) for v in ['servo_stop','servo_local_hold_stop']];old=pairs[0][1];x=arrays(old)['x'];eligible=[i for i in range(1,200) if old['steps'][i]['phase']=='CONTACT_SERVO'];idx=min(eligible,key=lambda i:x[i]-x[i-1]);frame=idx+1
  center=np.array(old['steps'][idx]['cube'][:3]);camera_pose=sapien_utils.look_at(center+np.array([-.15,.29,.23]),center+np.array([-.02,0,0]))
  env=gym.make('PlugCharger-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',human_render_camera_configs=dict(mount=None,pose=camera_pose,width=960,height=720,fov=.64));env.reset(seed=seed)
  raw=Image.new('RGB',(1920,720));pixels=[];readback=0
  for col,(path,t) in enumerate(pairs):
   z=t['steps'][idx];s=copy.deepcopy(t['initial']);s['actors']['charger'][0][:7]=z['tool'];s['actors']['receptacle'][0][:7]=z['cube'];s['articulations'][next(iter(s['articulations']))][0][13:22]=z['q'];state={g:{k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in vs.items()} for g,vs in s.items()};env.unwrapped.set_state_dict(state);actual=env.unwrapped.get_state_dict();readback=max(readback,max(float((actual[g][k]-state[g][k]).abs().max()) for g in state for k in state[g]));rgb=env.unwrapped.render_rgb_array()[0].cpu().numpy();raw.paste(Image.fromarray(rgb),(col*960,0))
   camera=next(iter(env.unwrapped._human_render_cameras.values())).camera;K=camera.get_intrinsic_matrix()[0].cpu().numpy();E=camera.get_extrinsic_matrix()[0].cpu().numpy();points={}
   for key in ['tool','cube']:
    xyz=E@np.r_[z[key][:3],1];uv=K@xyz[:3];uv=uv[:2]/uv[2];assert xyz[2]>0 and 0<=uv[0]<960 and 0<=uv[1]<720;points[key]=uv.tolist()
   pixels.append(points)
  env.close();raw.save(out/'RAW.png',dpi=(300,300));ann=Image.new('RGB',(1920,880),'#101d2b');ann.paste(raw,(0,80));d=ImageDraw.Draw(ann)
  d.text((18,10),f'Seed {seed} | step {frame}/200 | MEASURED_POSE_REPLAY | NO NEW RL CLAIM',fill='white',font=font)
  for col,((path,t),points) in enumerate(zip(pairs,pixels)):
   z=t['steps'][idx];v=arrays(t);label=['Original servo: rollback mode','Depth-preserving alignment'][col]
   d.text((col*960+18,46),label,fill='white',font=font)
   for key,color in [('tool','#ffbb52'),('cube','#48e0ef')]:
    u,vv=points[key];u+=col*960;vv+=80;d.ellipse((u-18,vv-18,u+18,vv+18),outline=color,width=4);d.line((u,vv,col*960+45,650 if key=='tool' else 700),fill=color,width=2);d.text((col*960+45,650 if key=='tool' else 700),'Measured charger origin' if key=='tool' else 'Measured socket origin',fill=color,font=small)
   d.text((col*960+18,810),f"Axial {v['x'][idx]:.2f} mm | lateral {v['lateral'][idx]:.2f} mm | final success {int(t['summary']['success'])}",fill='white',font=small)
   d.text((col*960+18,843),f"Final position {t['steps'][-1]['goal_position_error_m']*1000:.2f} mm; native 200 steps",fill='white',font=small)
  ann.save(out/'ANNOTATED.png',dpi=(300,300))
  ms=[mechanism(t) for _,t in pairs]
  caption=f'''# Seed {seed}: contact-mode diagnostic (inspected development case)

**OBSERVED.** Identical recorded ALIGN boundary, same native 200-step task. Original servo / depth-preserving servo final successes: {int(pairs[0][1]['summary']['success'])} / {int(pairs[1][1]['summary']['success'])}. Mode switches: {ms[0]['mode_switches']} / {ms[1]['mode_switches']}; measured backward axial travel during servo: {ms[0]['backward_travel_mm']:.2f} / {ms[1]['backward_travel_mm']:.2f} mm. This is accumulated negative displacement, not net insertion loss. RAW and ANNOTATED show exactly the same two images at physical step {frame}, selected by largest old-servo one-step retreat. Rings project measured actor origins using the actual camera matrices; they do not assert a visible contact point. TRACE uses post-step physical states; servo mode is the action selected from the preceding observation. The goal-triggering action may lack a servo annotation because GoalReached interrupts logging after physical stepping.

**INTERPRETATION.** Returning to the initial standoff when alignment is temporarily lost can undo insertion progress. Holding current depth removes that commanded retreat. Both controllers retain integration and contact dynamics, so residual backward motion can remain.

**NOT ESTABLISHED.** No limit-cycle/stability proof, calibrated force sensing, new RL gain, or independent confirmation from these already-inspected cases. Simulation contact force is a proxy. Privileged object poses are used. Camera replay adds zero physics steps and is not real hardware. Native success requires position <=5 mm AND angle <=0.2 rad at step 200.

Sources: RAW/ANNOTATED = MEASURED_POSE_REPLAY; TRACE = DATA_DERIVED_PLOT. NO NEW RL CLAIM.
'''
  (out/'CAPTION.md').write_text(caption)
  manifest=dict(seed=seed,frame=frame,selection='largest measured one-step backward axial displacement in original servo',physics_steps=0,max_pose_readback_error=readback,source_type='MEASURED_POSE_REPLAY',sources=[dict(path=str(p),sha256=sha(p),execution_manifest=json.loads((p.parent/'manifest.json').read_text())) for p,t in pairs],projected_actor_centers=pixels,camera_intrinsic=K.tolist(),camera_extrinsic=E.tolist(),script_sha256=sha(Path(__file__)),files={n:sha(out/n) for n in ['RAW.png','ANNOTATED.png','TRACE.png','TRACE.svg','CAPTION.md']})
  assert readback<1e-5;(out/'visual_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
