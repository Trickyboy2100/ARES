"""Render measured poses only; zero physics steps, not new experiments.
Unrecorded velocities are irrelevant to graphics; never use these states for dynamics.
"""
import argparse,copy,gzip,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch,gymnasium as gym
import mani_skill.envs
from mani_skill.utils import sapien_utils
from PIL import Image,ImageDraw,ImageFont

def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 pairs=[('pull_grasp','PullCubeTool-v1','design8_v4_retry',712000,['g0.035_y0.00','g0.105_y0.00'],['35 mm handle grasp','105 mm handle grasp']),('charger_handoff','PlugCharger-v1','charger_design4',721000,['a45.0_d0.05','a45.0_d0.075'],['45 deg / 50 mm approach','45 deg / 75 mm approach'])];receipts=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 for name,task,group,seed,modes,labels in pairs:
  paths=[a.root/group/f'{seed}_{m}.json.gz' for m in modes];traces=[load(p) for p in paths];assert traces[0]['summary']['initial_sha256']==traces[1]['summary']['initial_sha256']
  env=gym.make(task,num_envs=1,obs_mode='state',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',human_render_camera_configs=dict(pose=sapien_utils.look_at([.05,.5,.25] if 'Charger' in task else [.48,-.65,.50],[0,0,.09] if 'Charger' in task else [-.06,0,.09]),width=640,height=544,fov=.86));env.reset(seed=seed)
  output=a.out/f'{name}.mp4';proc=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','-','-an','-vcodec','libx264','-crf','20','-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE)
  err=0.;frames=len(traces[0]['steps'])+1
  for i in range(frames):
   im=Image.new('RGB',(1280,720),'#101d2b');d=ImageDraw.Draw(im);d.text((18,10),f'{task} | seed {seed} | SAME INITIAL STATE',font=font,fill='white');d.text((18,40),'RECORDED PHYSICS POSES / GRAPHICS-ONLY REPLAY / NO PHYSICAL ROBOT',font=small,fill='#b8cddd')
   for col,t in enumerate(traces):
    z=t['initial_measure'] if i==0 else t['steps'][i-1];state=copy.deepcopy(t['initial']);tool='charger' if 'Charger' in task else 'l_shape_tool';other='receptacle' if 'Charger' in task else 'cube';state['actors'][tool][0][:7]=z['tool'];state['actors'][other][0][:7]=z['cube'];robot=next(iter(state['articulations']));state['articulations'][robot][0][13:22]=z['q']
    target={g:{k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in values.items()} for g,values in state.items()};env.unwrapped.set_state_dict(target);actual=env.unwrapped.get_state_dict();err=max(err,max(float((actual[g][k]-target[g][k]).abs().max()) for g in target for k in target[g]));rgb=env.unwrapped.render_rgb_array()[0].cpu().numpy();im.paste(Image.fromarray(rgb[:,:,:3]),(640*col,112));d=ImageDraw.Draw(im);d.text((640*col+18,77),labels[col],font=font,fill='white');d.rectangle((640*col,638,640*(col+1),720),fill='#101d2b');d.text((640*col+18,646),f"{z.get('phase','INITIAL')} | step {i}/{frames-1} | grasp {int(z['grasped'])}",font=small,fill='white');d.text((640*col+18,675),f"Native final success: {int(t['summary']['success'])} | relative tool x: {z['tcp_T_tool'][0]*1000:.1f} mm",font=small,fill='#a8e6c4')
   proc.stdin.write(np.asarray(im).tobytes())
   if i in [0,frames//2,frames-1]:im.save(a.out/f'{name}_{i:03d}.png')
  proc.stdin.close();assert proc.wait()==0;env.close();receipts.append(dict(video=output.name,source=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],physics_steps=0,frames=frames,max_state_readback_error=err,pose_source='measured per-step joint and actor poses; unrecorded velocities not reconstructed; graphics only'))
 (a.out/'REPLAY_RECEIPTS.json').write_text(json.dumps(receipts,indent=2))
if __name__=='__main__':main()
