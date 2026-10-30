"""One diagnostic movie from measured native poses; no new physics steps."""
import argparse,copy,gzip,hashlib,json,subprocess
from pathlib import Path
import gymnasium as gym,numpy as np,torch
import mani_skill.envs
from mani_skill.utils import sapien_utils
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);paths=[a.left,a.right];traces=[]
 for path in paths:
  with gzip.open(path,'rt') as f:traces.append(json.load(f))
 assert traces[0]['summary']['initial_sha256']==traces[1]['summary']['initial_sha256']
 env=gym.make('PlugCharger-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',human_render_camera_configs=dict(pose=sapien_utils.look_at([.05,.5,.25],[0,0,.09]),width=640,height=544,fov=.86));env.reset(seed=traces[0]['summary']['seed']);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 proc=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(a.out/'controller_pair.mp4')],stdin=subprocess.PIPE);error=0
 for frame in range(201):
  im=Image.new('RGB',(1280,720),'#101d2b');d=ImageDraw.Draw(im);d.text((18,10),'EXP-001 | SAME INITIAL STATE / SAME GRASP / NATIVE 200 STEPS',fill='white',font=font);d.text((18,40),'RECORDED SIMULATION POSES / GRAPHICS-ONLY REPLAY / NO PHYSICAL ROBOT / NO RL',fill='#b8cddd',font=small)
  for col,t in enumerate(traces):
   z=t['initial_measure'] if frame==0 else t['steps'][frame-1];s=copy.deepcopy(t['initial']);s['actors']['charger'][0][:7]=z['tool'];s['actors']['receptacle'][0][:7]=z['cube'];s['articulations'][next(iter(s['articulations']))][0][13:22]=z['q'];state={g:{k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in vs.items()} for g,vs in s.items()};env.unwrapped.set_state_dict(state);actual=env.unwrapped.get_state_dict();error=max(error,max(float((actual[g][k]-state[g][k]).abs().max()) for g in state for k in state[g]));rgb=env.unwrapped.render_rgb_array()[0].cpu().numpy();im.paste(Image.fromarray(rgb),(col*640,112));d=ImageDraw.Draw(im);d.text((col*640+18,78),['Frozen integral + goal hold','Bounded alignment servo + goal hold'][col],fill='white',font=font);d.rectangle((col*640,638,(col+1)*640,720),fill='#101d2b');d.text((col*640+18,647),f"{z.get('phase','INITIAL')} | step {frame}/200 | final {int(t['summary']['success'])}",fill='white',font=small)
   if frame:d.text((col*640+18,677),f"Goal error {z['goal_position_error_m']*1000:.2f} mm | angle {z['goal_angle_error_rad']:.3f} rad",fill='#b8e6cd',font=small)
  proc.stdin.write(np.asarray(im).tobytes())
  if frame in [100,200]:im.save(a.out/f'frame_{frame}.png')
 proc.stdin.close();assert proc.wait()==0;env.close();(a.out/'receipt.json').write_text(json.dumps(dict(sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],physics_steps=0,frames=201,max_pose_readback_error=error),indent=2))
if __name__=='__main__':main()
