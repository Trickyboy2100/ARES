"""Portable graphics-only renderer for the versioned measured-state subset.

No env.step, policy inference or new outcome. Never import into training.
"""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import gymnasium as gym,torch
import mani_skill.envs
from mani_skill.utils import sapien_utils
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

def tensor_state(state):return {g:{k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in x.items()} for g,x in state.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 with gzip.open(a.states,'rt') as f:cases=json.load(f)
 env=gym.make('StackPyramid-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',human_render_camera_configs=dict(pose=sapien_utils.look_at([.42,.42,.40],[-.02,0,.12]),width=640,height=640,fov=.95));env.reset(seed=0)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);receipts=[]
 for case in cases:
  file=a.output/(case['case']['name']+'.mp4');w=imageio.get_writer(file,fps=20,codec='libx264',quality=8);error=0
  for frame in range(251):
   im=Image.new('RGB',(1280,800),'#101d2b');d=ImageDraw.Draw(im);d.text((22,12),f'Case {case["case"]["seed"]} | ORIGINAL RECORDED PHYSICS STATES',fill='white',font=font);d.text((22,46),'PUBLIC SIMULATION | GRAPHICS-ONLY REPLAY | NO PHYSICAL ROBOT | NOT A NEW EXPERIMENT',fill='#a8c5df',font=small)
   for col,b in enumerate(case['branches']):
    t=None if frame==0 else b['transitions'][frame-1];state=b['initial'] if t is None else t['state'];target=tensor_state(state);env.unwrapped.set_state_dict(target);actual=env.unwrapped.get_state_dict()
    for g in target:
     for k in target[g]:error=max(error,float((actual[g][k]-target[g][k]).abs().max()))
    rgb=env.unwrapped.render_rgb_array()[0].cpu().numpy();im.paste(Image.fromarray(rgb[:,:,:3]),(col*640,102));d=ImageDraw.Draw(im);d.text((col*640+18,77),f'{b["policy"]} | native final {int(b["success"])}',fill='#a8c5df',font=small);d.rectangle((col*640,676,col*640+640,746),fill='#101d2b');d.text((col*640+18,684),f'{"INITIAL" if t is None else t["segment"]} | step {frame}/250',fill='white',font=small)
   d.text((22,756),'Measured A red / B green / C blue | actual joint and object poses | no idealized trajectories',fill='white',font=small);w.append_data(np.asarray(im))
  w.close();receipts.append(dict(video=file.name,max_state_readback_error=error,physics_steps=0))
 env.close();(a.output/'REPLAY_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
if __name__=='__main__':main()
