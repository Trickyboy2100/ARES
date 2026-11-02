"""Native250 full-state Pyramid evidence: controller repair, never an RL claim."""
import argparse,copy,gzip,inspect,json,shutil,textwrap
from pathlib import Path
import numpy as np
from program_charger_video_sync import font,pipe,close_pipe,sha,execution_manifest
from program_multiskill_video import viewer
from program_charger_mechanism import rotation

CLAIM='SHARED_CONTROLLER_REPAIR_NOT_RL_GAIN'

def series(t,label):
    states=[t['initial'],*[x['state'] for x in t['transitions']]]
    actors=[{name:np.asarray(v[0]) for name,v in state['actors'].items()} for state in states]
    A=np.array([x['cubeA'][:3] for x in actors]);B=np.array([x['cubeB'][:3] for x in actors]);C=np.array([x['cubeC'][:3] for x in actors]);q=np.array([next(iter(t['initial']['articulations'].values()))[0][13:22],*[x['q'][0] for x in t['transitions']]])
    dist=np.linalg.norm(A[:,:2]-B[:,:2],axis=1)*1000
    return {'label':label,'phase':['INITIAL',*[x['segment'] for x in t['transitions']]],'success':t['summary']['success'],'error':t['summary'].get('error'),'signals':{'Actual A-B XY distance (mm)':dist.tolist(),'Controller support margin (mm)':(61.568-dist).tolist(),'Measured finger opening (mm)':(q[:,-2:].sum(axis=1)*1000).tolist(),'Actual C height (mm)':(C[:,2]*1000).tolist(),'Max cube linear speed (mm/s)':[max(np.linalg.norm(a[n][7:10])*1000 for n in ['cubeA','cubeB','cubeC']) for a in actors]},'decisions':[dict(d,step=d['boundary_step']) for d in t['decisions']],'skill_results':[dict(skill=['PLACE_SUPPORT','PLACE_TOP'][i],boundary_step=b['step'],physical_success=b['valid'],measurement=b) for i,b in enumerate(t['summary']['boundaries'])]}

def annotate(raw,k,h,seed,ss):
    from PIL import Image,ImageDraw
    im=Image.new('RGB',(1600,1060),'#132334');im.paste(raw,(0,80));d=ImageDraw.Draw(im);d.text((12,6),f'Seed {seed} | step{k}/{h} | MEASURED_POSE_REPLAY | {CLAIM}',font=font(20),fill='white')
    for i,s in enumerate(ss):
        x=i*800;color=['#f4ae59','#52cddd'][i];d.text((x+12,44),s['label']+' | '+s['phase'][k],font=font(20),fill=color);d.text((x+12,686),f"OFFLINE final result: {int(s['success'])} | A red / B green / C blue",font=font(18),fill='white')
        visible=[a for a in s['decisions'] if a['step']<=k];done=[a for a in s['skill_results'] if a['boundary_step']<=k]
        result='pending release / stability verification'
        if done:
            b=done[0]['measurement'];result={n:b[n] for n in ['valid','finger_opening_m','objects_static','target_xy_error_m','AB_distance_m']}
        fields=[('Skill1 parameter',visible[0]['parameters'] if visible else 'pending'),('Actual support SkillResult',result),('Successor prerequisite','released, stable support; actual attachment/geometry'),('Skill2 chosen parameter',visible[1]['parameters'] if len(visible)>1 else 'NOT REACHED / not selected')]
        for row,(name,value) in enumerate(fields):
            for j,line in enumerate(textwrap.wrap(name+': '+json.dumps(value,separators=(',',':')),94)[:2]):d.text((x+12,717+row*39+j*17),line,font=font(14),fill='#dce8f5')
    for row,key in enumerate(['Actual A-B XY distance (mm)','Measured finger opening (mm)']):
        y=890+row*80;vals=[v for s in ss for v in s['signals'][key]];lo=min(vals);hi=max(vals);pad=(hi-lo or 1)*.05;lo-=pad;hi+=pad;d.text((12,y),key,font=font(15),fill='white')
        for s,color in zip(ss,['#f4ae59','#52cddd']):d.line([(250+j/h*1320,y+60-(v-lo)/(hi-lo)*55) for j,v in enumerate(s['signals'][key])],fill=color,width=2)
        d.line((250+k/h*1320,y,250+k/h*1320,y+65),fill='white',width=2)
    return im

def main():
    import torch,gymnasium as gym,mani_skill.envs
    from mani_skill.utils import sapien_utils
    from PIL import Image
    ap=argparse.ArgumentParser();ap.add_argument('--left',type=Path,required=True);ap.add_argument('--right',type=Path,required=True);ap.add_argument('--left-label',default='Fixed-side gripper controller');ap.add_argument('--right-label',default='Axis-aware gripper controller');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();paths=[a.left.resolve(),a.right.resolve()];ts=[json.load(gzip.open(p,'rt')) for p in paths];assert ts[0]['summary']['initial_sha256']==ts[1]['summary']['initial_sha256'];assert all(len(t['transitions'])==250 for t in ts);seed=ts[0]['summary']['seed'];ss=[series(t,l) for t,l in zip(ts,[a.left_label,a.right_label])];a.out.mkdir(parents=True,exist_ok=False)
    env=gym.make('StackPyramid-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',human_render_camera_configs=dict(pose=sapien_utils.look_at([.42,.42,.40],[-.02,0,.08]),width=800,height=600,fov=.95));env.reset(seed=seed)
    camera=next(iter(env.unwrapped._human_render_cameras.values())).camera;K=camera.get_intrinsic_matrix()[0].cpu().numpy();E=camera.get_extrinsic_matrix()[0].cpu().numpy();points=[];native_sha=sha(Path(inspect.getfile(type(env.unwrapped))));rawp=pipe(a.out/'RAW.mp4',1600,600,20);annp=pipe(a.out/'ANNOTATED.mp4',1600,1060,20);maxerr=0.
    try:
        for k in range(251):
            raw=Image.new('RGB',(1600,600))
            for i,t in enumerate(ts):
                state=t['initial'] if k==0 else t['transitions'][k-1]['state'];target={g:{n:torch.tensor(v,dtype=torch.float32,device='cuda') for n,v in values.items()} for g,values in state.items()};env.unwrapped.set_state_dict(target);actual=env.unwrapped.get_state_dict();maxerr=max(maxerr,max(float((actual[g][n]-target[g][n]).abs().max()) for g in target for n in target[g]));rgb=env.unwrapped.render_rgb_array()[0].cpu().numpy();raw.paste(Image.fromarray(rgb[:,:,:3]),(800*i,0))
                for n in ['cubeA','cubeB','cubeC']:
                    z=state['actors'][n][0];corners=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [-.02,.02]])@rotation(z[3:7]).T+z[:3];cv=(E@np.c_[corners,np.ones(8)].T).T;uv=(K@cv[:,:3].T).T;points.extend((uv[:,:2]/uv[:,2:]).tolist());assert (cv[:,2]>0).all()
            rawp.stdin.write(np.asarray(raw).tobytes());ann=annotate(raw,k,250,seed,ss);annp.stdin.write(np.asarray(ann).tobytes())
            if k in [0,100,150,250]:raw.save(a.out/f'RAW_{k:03d}.png');ann.save(a.out/f'ANNOTATED_{k:03d}.png')
    finally:close_pipe(rawp);close_pipe(annp);env.close()
    assert maxerr<1e-5;data={'task':'StackPyramid-v1','seed':seed,'fps':20,'horizon':250,'claim':CLAIM,'series':ss};(a.out/'DATA.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');viewer(a.out,data);html=(a.out/'SYNC_VIEWER.html').read_text().replace('Native PullCubeTool-v1: full100-step horizon; terminal cube/base distance must be below600mm.','Native StackPyramid-v1: full250-step horizon; no changed success criterion. Support margin is an existing controller contract, not the complete native success test.').replace('Frame0 contact readback is unvalidated (before first physical step) and excluded from force curves; raw value is retained in DATA.','Actual three-cube geometry and measured finger opening are recorded; contact forces are unavailable and not invented. Full curves and final outcome are offline analysis, not policy inputs.').replace('NO NEW RL CLAIM',CLAIM).replace('two executable skill decisions','verified skill boundaries and controller repair');(a.out/'SYNC_VIEWER.html').write_text(html)
    for p,n in zip(paths,['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz']):shutil.copy2(p,a.out/n)
    shutil.copy2(__file__,a.out/'render_source.py');m={'task':'StackPyramid-v1','claim':CLAIM,'source_kind':'MEASURED_POSE_REPLAY','physics_steps_added':0,'frames':251,'fps':20,'frame_mapping':'frame k=step k, timestamp k/20; frame0 initial','max_state_readback_error':maxerr,'all_cube_corner_pixel_min':np.min(points,axis=0).tolist(),'all_cube_corner_pixel_max':np.max(points,axis=0).tolist(),'camera_intrinsic':K.tolist(),'camera_extrinsic':E.tolist(),'sources':[{'path':str(p),'sha256':sha(p),'execution_manifest':execution_manifest(p)} for p in paths],'native_model_source_sha256':native_sha,'render_source_sha256':sha(Path(__file__)),'shared_viewer_source_sha256':sha(Path(__file__).with_name('program_multiskill_video.py')),'outputs':{p.name:sha(p) for p in a.out.iterdir() if p.is_file()}};(a.out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'out':str(a.out),'frames':251,'maxerr':maxerr}))
if __name__=='__main__':main()
