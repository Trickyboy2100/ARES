"""Render ORIGINAL measured states; never collect training/evaluation outcomes.

set_state_dict appears ONLY in this graphics-only script. Saved native robot q,
object poses and velocities drive the public Panda scene. No env.step is called,
no physics result is inferred from this replay, and no ideal motion is invented.
"""
import argparse,hashlib,json,gzip
from pathlib import Path
import numpy as np
import torch,gymnasium as gym
import mani_skill.envs
from mani_skill.utils import sapien_utils
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def subset(state,index):
    return {group:{key:torch.tensor([value[index]],device='cuda',dtype=torch.float32) for key,value in data.items()} for group,data in state.items()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    camera=dict(pose=sapien_utils.look_at([.42,.42,.40],[-.02,0,.12]),width=640,height=640,fov=.95)
    env=gym.make('StackPyramid-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',human_render_camera_configs=camera);env.reset(seed=0);u=env.unwrapped
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',23)
    # Selected from training-design interventions, explicitly qualitative, not
    # cherry-picked unseen method generalization. A0/A2 both locally verified.
    cases=[dict(name='verified_A_changes_downstream',seed=420010,policies=['A0_C1','A2_C1']),dict(name='same_posterior_second_decision',seed=420002,policies=['A0_C0','A0_C1'])];manifest=[];packed=[]
    dev=json.loads((a.root/'development64'/'RESULTS.json').read_text())['rows'];lookup={(r['method'],seed):(r,i) for r in dev for i,seed in enumerate(r['seeds'])}
    for sd,direction,name in [(37,1,'trained_policy_rescue'),(43,-1,'trained_policy_regression')]:
        plain=f'ppo_s{sd}_b0_full';risk=f'ppo_s{sd}_b2_full';seed=next(seed for seed in range(520000,520064) if int(lookup[risk,seed][0]['success'][lookup[risk,seed][1]])-int(lookup[plain,seed][0]['success'][lookup[plain,seed][1]])==direction);cases.append(dict(name=name,seed=seed,policies=[plain,risk],scope='posthoc qualitative dev example, not an extra statistical sample'))
    for case in cases:
        seed=case['seed'];base_seed=420000 if seed<520000 else 520000;batch=(seed-base_seed)//16;idx=(seed-base_seed)%16;traces=[];sources=[]
        for pol in case['policies']:
            if base_seed==420000:
                file=a.root/'opportunity64'/f'{batch:03d}_{pol}.json';d=json.loads(file.read_text())
            else:
                row,_=lookup[pol,seed];file=a.root/'development64'/row['trajectory_file']
                with gzip.open(file,'rt') as f:d=json.load(f)
                d['summary']=row
            traces.append(d);sources.append(dict(path=str(file),sha256=sha(file),policy=pol,model_sha256=d['summary'].get('model_sha256')))
        packed.append(dict(case=case,sources=sources,initial_group_sha256=traces[0]['summary']['initial_sha256'],branches=[dict(policy=pol,success=bool(d['summary']['success'][idx]),model_sha256=d['summary'].get('model_sha256'),boundaries=d['summary']['boundaries'][idx],initial={g:{k:[v[idx]] for k,v in x.items()} for g,x in d['initial'].items()},transitions=[dict(step=t['step'],segment=t['segments'][idx],state={g:{k:[v[idx]] for k,v in x.items()} for g,x in t['state'].items()}) for t in d['transitions']]) for pol,d in zip(case['policies'],traces)]))
        assert traces[0]['summary']['initial_sha256']==traces[1]['summary']['initial_sha256']
        output=out/(case['name']+'.mp4');writer=imageio.get_writer(output,fps=20,codec='libx264',quality=8);maxerr=0.;keyframes=[]
        for frame in range(251):
            board=Image.new('RGB',(1280,800),'#101d2b');draw=ImageDraw.Draw(board);draw.text((22,12),f'Original native physics trace | case {seed}',fill='white',font=title);draw.text((22,44),'PUBLIC SIMULATION | ORIGINAL MEASURED STATES | NO PHYSICAL ROBOT',fill='#a8c5df',font=small)
            for col,d in enumerate(traces):
                state=d['initial'] if frame==0 else d['transitions'][frame-1]['state'];selected=subset(state,idx);u.set_state_dict(selected);actual=u.get_state_dict()
                for group in selected:
                    for key in selected[group]:maxerr=max(maxerr,float((actual[group][key]-selected[group][key]).abs().max().item()))
                image=u.render_rgb_array()[0].cpu().numpy();board.paste(Image.fromarray(image[:,:,:3]),(col*640,102));draw=ImageDraw.Draw(board);r=d['summary'];b=r['boundaries'][idx];boundary0=next((x for x in b if x['stage']==0),{});segment='INITIAL' if frame==0 else d['transitions'][frame-1]['segments'][idx];valid=frame>=boundary0.get('step',1000) and boundary0.get('valid',False)
                label=case['policies'][col].replace('ppo_s','seed').replace('_b0_full',' ordinary PPO').replace('_b2_full',' risk PPO');draw.text((col*640+18,77),f'{label} | native final: {int(r["success"][idx])}',fill='#8bdbc0' if r['success'][idx] else '#f5aa97',font=font);draw.rectangle((col*640,676,col*640+640,746),fill='#101d2b');draw.text((col*640+18,680),f'{segment} | A boundary verified: {valid}',fill='white',font=small);draw.text((col*640+18,706),f'Original source step {frame}/250 | t={frame*.05:.2f}s',fill='#a8c5df',font=small)
            draw.text((22,753),'A red: first support | B green: second support | C blue: top cube | release includes retreat + stability',fill='white',font=small);draw.rectangle((22,786,1258,792),fill='#33485c');draw.rectangle((22,786,22+int(1236*frame/250),792),fill='#46b8aa');writer.append_data(np.asarray(board))
            if frame in [0,45,80,110,160,250]:
                f=out/f'{case["name"]}_step{frame:03d}.png';board.save(f);keyframes.append(f.name)
        writer.close();manifest.append(dict(**case,sources=sources,video=output.name,video_sha256=sha(output),keyframes=keyframes,max_state_readback_error=maxerr,physics_steps=0,replay_kind='graphics-only original measured state replay; not a new experiment'))
        print(json.dumps(manifest[-1]),flush=True)
    with gzip.open(out/'SELECTED_ORIGINAL_STATES.json.gz','wt') as f:json.dump(packed,f,separators=(',',':'))
    env.close();(out/'RENDER_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
