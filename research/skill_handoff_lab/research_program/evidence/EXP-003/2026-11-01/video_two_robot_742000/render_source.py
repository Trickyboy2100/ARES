"""Native100 two-robot recorded-state video; no dynamics advancement or RL claim."""
import argparse,gzip,inspect,json,shutil,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from program_charger_video_sync import font,pipe,close_pipe,sha,execution_manifest
from program_multiskill_video import viewer
CLAIM='PHYSICAL_HANDOFF_PARAMETER_PAIR_NOT_RL_GAIN'

def series(t,label):
    ms=[t['initial_measure'],*[r['measurement'] for r in t['steps']]]
    return {'label':label,'success':t['summary']['success'],'phase':['INITIAL',*[r['phase'] for r in t['steps']]],'decisions':[dict(x,step=x['boundary_step']) for x in t['decisions']],'skill_results':t['skill_results'],'signals':{'Cube goal distance (mm)':[m['cube_goal_distance_m']*1000 for m in ms],'Cube world y (mm)':[m['cube'][1]*1000 for m in ms],'Cube speed (mm/s)':[float(np.linalg.norm(m['cube_linear_velocity'])*1000) for m in ms],'Right grasped (0/1)':[int(m['1']['grasped']) for m in ms],'Left cube contact 20Hz (N)':[max(m['0']['robot_cube_force_N'].values(),default=0) if m['contact_valid'] else None for m in ms],'Right cube contact 20Hz (N)':[max(m['1']['robot_cube_force_N'].values(),default=0) if m['contact_valid'] else None for m in ms]}}

def annotate(raw,k,d):
    im=Image.new('RGB',(1600,1060),'#132334');im.paste(raw,(0,80));dr=ImageDraw.Draw(im);dr.text((12,5),f"Seed {d['seed']} | step {k}/100 | MEASURED_POSE_REPLAY | NO NEW RL CLAIM",font=font(21),fill='white')
    for i,s in enumerate(d['series']):
        x=i*800;col=['#f4ae59','#52cddd'][i];dr.text((x+12,44),s['label']+' | '+s['phase'][k],font=font(19),fill=col);dr.text((x+12,684),f"OFFLINE final native success: {int(s['success'])} | red cube, green goal",font=font(17),fill='white')
        dec=[a for a in s['decisions'] if a['step']<=k];res=[a for a in s['skill_results'] if a['boundary_step']<=k];first=res[0] if res else None
        handoff='not reached'
        if first:
            m=first['measurement'];handoff={'passed':first['physical_success'],'cube_y_mm':round(m['cube'][1]*1000,2),'speed_mm_s':round(float(np.linalg.norm(m['cube_linear_velocity'])*1000),2),'boundary_step':first['boundary_step']}
        fields=[('Skill1 LEFT_PUSH',dec[0]['parameters'] if dec else 'pending'),('Actual handoff SkillResult',handoff),('Skill2 RIGHT_PICK_TO_GOAL',dec[1]['parameters'] if len(dec)>1 else 'NOT REACHED'),('Right result',res[-1]['physical_success'] if len(res)>1 else 'pending final native100')]
        for j,(name,val) in enumerate(fields):
            for h,line in enumerate(textwrap.wrap(name+': '+json.dumps(val),90)[:2]):dr.text((x+12,717+j*39+h*17),line,font=font(14),fill='#dce8f5')
    for row,key in enumerate(['Cube goal distance (mm)','Cube world y (mm)']):
        yy=890+row*80;v=[x for s in d['series'] for x in s['signals'][key]];lo,hi=min(v),max(v);pad=max(1,hi-lo)*.05;lo-=pad;hi+=pad;dr.text((12,yy),key,font=font(15),fill='white')
        for s,col in zip(d['series'],['#f4ae59','#52cddd']):dr.line([(250+j/100*1320,yy+60-(v-lo)/(hi-lo)*55) for j,v in enumerate(s['signals'][key])],fill=col,width=2)
        dr.line((250+k/100*1320,yy,250+k/100*1320,yy+65),fill='white',width=2)
    return im

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--left',type=Path,required=True);ap.add_argument('--right',type=Path,required=True);ap.add_argument('--left-label',default='Push target y = 20 mm');ap.add_argument('--right-label',default='Push target y = 80 mm');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();paths=[a.left.resolve(),a.right.resolve()];ts=[json.load(gzip.open(p,'rt')) for p in paths];assert all(len(t['steps'])==100 for t in ts);assert ts[0]['summary']['initial_sha256']==ts[1]['summary']['initial_sha256'];ss=[series(t,label) for t,label in zip(ts,[a.left_label,a.right_label])];d={'task':'TwoRobotPickCube-v1','seed':ts[0]['summary']['seed'],'fps':20,'horizon':100,'claim':CLAIM,'series':ss};a.out.mkdir(parents=True,exist_ok=False)
    import torch,gymnasium as gym,mani_skill.envs
    from mani_skill.utils import sapien_utils
    eye=[1.3,0,.9];target=[0,0,.20];env=gym.make('TwoRobotPickCube-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',human_render_camera_configs=dict(pose=sapien_utils.look_at(eye,target),width=800,height=600,fov=1.05));env.reset(seed=d['seed']);camera=next(iter(env.unwrapped._human_render_cameras.values())).camera;K=camera.get_intrinsic_matrix()[0].cpu().numpy();E=camera.get_extrinsic_matrix()[0].cpu().numpy();native_sha=sha(Path(inspect.getfile(type(env.unwrapped))));rp=pipe(a.out/'RAW.mp4',1600,600,20);apip=pipe(a.out/'ANNOTATED.mp4',1600,1060,20);err=0;projected=[]
    try:
        for k in range(101):
            raw=Image.new('RGB',(1600,600))
            for i,t in enumerate(ts):
                state=t['initial'] if k==0 else t['steps'][k-1]['state'];st={g:{n:torch.tensor(v,dtype=torch.float32,device='cuda') for n,v in values.items()} for g,values in state.items()};env.unwrapped.set_state_dict(st);actual=env.unwrapped.get_state_dict();err=max(err,max(float((actual[g][n]-st[g][n]).abs().max()) for g in st for n in st[g]));raw.paste(Image.fromarray(env.unwrapped.render_rgb_array()[0].cpu().numpy()[:,:,:3]),(i*800,0));m=t['initial_measure'] if k==0 else t['steps'][k-1]['measurement']
                for name in ['cube','goal']:
                    cv=E@np.r_[m[name][:3],1];uv=K@cv[:3];assert cv[2]>0;projected.append((uv[:2]/uv[2]).tolist())
            rp.stdin.write(np.asarray(raw).tobytes());ann=annotate(raw,k,d);apip.stdin.write(np.asarray(ann).tobytes())
            if k in [0,40,70,100]:raw.save(a.out/f'RAW_{k:03d}.png');ann.save(a.out/f'ANNOTATED_{k:03d}.png')
    finally:close_pipe(rp);close_pipe(apip);env.close()
    assert err<1e-5;(a.out/'DATA.json').write_text(json.dumps(d,indent=2)+'\n');viewer(a.out,d);h=(a.out/'SYNC_VIEWER.html').read_text().replace('height="600"','height="720"').replace('Native PullCubeTool-v1: full100-step horizon; terminal cube/base distance must be below600mm.','Native TwoRobotPickCube-v1: full100-step horizon. Final cube within25mm of goal AND right arm static.').replace('Frame0 contact readback is unvalidated (before first physical step) and excluded from force curves; raw value is retained in DATA.','Frame0 force is unavailable and excluded. Forces are original20Hz samples, not continuous-time peaks. Full curves and final outcomes are offline analysis, not actor inputs.').replace('NO NEW RL CLAIM',CLAIM);(a.out/'SYNC_VIEWER.html').write_text(h)
    for p,n in zip(paths,['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz']):shutil.copy2(p,a.out/n)
    shutil.copy2(__file__,a.out/'render_source.py');manifest={'task':d['task'],'claim':CLAIM,'source_kind':'MEASURED_POSE_REPLAY','physics_steps_added':0,'frames':101,'fps':20,'max_state_readback_error':err,'frame_mapping':'frame k=physical step k, PTS k/20; initial frame0','camera_eye':eye,'camera_target':target,'camera_intrinsic':K.tolist(),'camera_extrinsic':E.tolist(),'all_cube_goal_center_pixel_min':np.min(projected,axis=0).tolist(),'all_cube_goal_center_pixel_max':np.max(projected,axis=0).tolist(),'native_model_source_sha256':native_sha,'sources':[{'path':str(p),'sha256':sha(p),'execution_manifest':execution_manifest(p)} for p in paths],'render_source_sha256':sha(Path(__file__))};manifest['outputs']={p.name:sha(p) for p in a.out.iterdir() if p.is_file()};(a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(a.out)
if __name__=='__main__':main()
