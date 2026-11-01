"""Full-episode two-skill physical evidence. Graphics replay only: never env.step()."""
import argparse,copy,gzip,inspect,json,shutil,textwrap
from pathlib import Path
import numpy as np
from program_charger_video_sync import font,pipe,close_pipe,sha,execution_manifest


def series(trace,label):
    states=[dict(trace['initial_measure'],phase='INITIAL'),*trace['steps']]
    result={'label':label,'phase':[s['phase'] for s in states],
        'signals':{'Cube/base distance (mm)':[s['cube_base_distance_m']*1000 for s in states],
        'Tool lift from start (mm)':[(s['tool'][2]-states[0]['tool'][2])*1000 for s in states],
        'Measured tool grasp (0/1)':[int(s['grasped']) for s in states],
        'Tool/cube contact proxy (N)':[None,*[s['tool_cube_force_N'] for s in states[1:]]],
        'TCP-relative tool x (mm)':[s['tcp_T_tool'][0]*1000 for s in states]},
        'target_y_m':trace['summary'].get('target_y'),'initial_contact_readback_N_unvalidated':states[0]['tool_cube_force_N'],'contact_validity':'frame0 buffer read before first physics step; unvalidated, excluded from plot; source preserved',
        'decisions':[dict(d,step=int(d.get('step',d.get('boundary_step',0)))) for d in trace.get('decisions',[])],'skill_results':trace.get('skill_results',[]),'native_success':bool(trace['summary']['success']),'success':bool(trace['summary'].get('task_success',trace['summary']['success']))}
    target=trace['summary'].get('target_y')
    if target is not None:result['signals']['Declared target distance (mm)']=[float(np.linalg.norm(np.array(s['cube'][:2])-np.array([-.045,target])))*1000 for s in states]
    assert len(result['decisions'])>=2,'Need two real parameter decision records, not inferred waypoints'
    for values in result['signals'].values():assert np.isfinite([v for v in values if v is not None]).all()
    return result


def viewer(out,data):
    html=r'''<!doctype html><meta charset="utf-8"><title>ARES two-skill measured evidence</title>
<style>body{background:#132334;color:white;font:16px system-ui;margin:20px}main{max-width:1500px;margin:auto}video,canvas{width:100%;background:#0c1723}button,select{padding:8px;margin:5px}#cards{display:flex;gap:20px}pre{white-space:pre-wrap;font-size:13px;flex:1;max-width:49%}canvas{cursor:crosshair}a{color:#7acaff}</style>
<main><h2>ARES | two executable skill decisions | measured physical evidence</h2>
<p>MEASURED_POSE_REPLAY / zero new physics / NO NEW RL CLAIM. ARES-ToolTarget-v0 (native-derived): full100-step horizon; native cube/base distance below600mm AND declared target error at most40mm. Full-task means this conjunction; native-only result is separately preserved in DATA. Frame0 contact readback is unvalidated (before first physical step) and excluded from force curves; raw value is retained in DATA. Parameters and SkillResult below are source records, never future labels fed to the policy.</p>
<select id="choice"><option>ANNOTATED</option><option>RAW</option></select><button id="play">Play / pause</button><span id="status"></span>
<video id="video" controls></video><div id="events"></div><canvas id="trace" width="1400" height="720"></canvas><div id="cards"><pre id="left"></pre><pre id="right"></pre></div>
<p><a href="DATA.json">DATA</a> · <a href="MANIFEST.json">source hashes</a> · <a href="CAPTION.md">interpretation and limitations</a></p></main>
<script>const D=__DATA__,v=document.querySelector('#video'),c=document.querySelector('#trace'),ctx=c.getContext('2d'),choice=document.querySelector('#choice');let media=null;
function step(){return Math.max(0,Math.min(D.horizon,Math.floor((media===null?v.currentTime:media)*D.fps+1e-7)));}
function swap(){const t=v.currentTime||0;media=null;v.src=choice.value+'.mp4';v.currentTime=t;}choice.onchange=swap;swap();
D.series.forEach((s,i)=>s.decisions.forEach(d=>{const b=document.createElement("button");b.textContent=(i?"Right":"Left")+" "+d.skill+" decision at step "+d.step;b.onclick=()=>{v.currentTime=d.step/D.fps;media=null;draw()};document.querySelector("#events").appendChild(b)}));
if(v.requestVideoFrameCallback){const next=(_,m)=>{media=m.mediaTime;draw();v.requestVideoFrameCallback(next)};v.requestVideoFrameCallback(next);}
function draw(){ctx.fillStyle='#132334';ctx.fillRect(0,0,c.width,c.height);const keys=Object.keys(D.series[0].signals),colors=['#f4ae59','#52cddd'],k=step();document.querySelector('#status').textContent='Physical step '+k+'/'+D.horizon+' | '+(k/D.fps).toFixed(2)+' s';
keys.forEach((key,row)=>{const y=20+row*110,all=D.series.flatMap(s=>s.signals[key]).filter(v=>v!==null),lo0=Math.min(...all),hi0=Math.max(...all),pad=(hi0-lo0||1)*.05,lo=lo0-pad,hi=hi0+pad;ctx.fillStyle='#fff';ctx.font='16px sans-serif';ctx.fillText(key,10,y+15);ctx.fillText(lo0.toFixed(2)+' … '+hi0.toFixed(2),1140,y+15);D.series.forEach((s,i)=>{ctx.strokeStyle=colors[i];ctx.beginPath();let started=false;s.signals[key].forEach((a,j)=>{if(a===null)return;const x=210+j/D.horizon*1140,yy=y+90-(a-lo)/(hi-lo)*70;started?ctx.lineTo(x,yy):ctx.moveTo(x,yy);started=true});ctx.stroke();s.decisions.forEach(d=>{const x=210+d.step/D.horizon*1140;ctx.fillStyle=colors[i];ctx.fillRect(x,y+20+i*6,3,6)})});ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(210+k/D.horizon*1140,y+20);ctx.lineTo(210+k/D.horizon*1140,y+95);ctx.stroke();});
D.series.forEach((s,i)=>{const visible=s.decisions.filter(d=>d.step<=k);document.querySelector(i?'#right':'#left').textContent=s.label+' | '+s.phase[k]+'\nFull-task terminal success: '+s.success+'\nRecorded decisions through current step:\n'+JSON.stringify(visible,null,2)+'\nMeasured completed SkillResult:\n'+JSON.stringify(s.skill_results.filter(r=>r.boundary_step<=k),null,2);});}
c.onclick=e=>{const r=c.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*c.width;v.currentTime=Math.round(Math.max(0,Math.min(1,(x-210)/1140))*D.horizon)/D.fps;media=null;draw()};document.querySelector('#play').onclick=()=>v.paused?v.play():v.pause();v.onseeked=draw;v.ontimeupdate=draw;function tick(){if(!v.paused)draw();requestAnimationFrame(tick)};tick();draw();</script>'''
    (out/'SYNC_VIEWER.html').write_text(html.replace('__DATA__',json.dumps(data,allow_nan=False).replace('</','<\\/')))


def annotation(raw,k,h,seed,ss,labels):
    from PIL import Image,ImageDraw
    ann=Image.new('RGB',(1600,1060),'#132334');ann.paste(raw,(0,80));draw=ImageDraw.Draw(ann);draw.text((12,8),f'Seed {seed} | physical step {k}/{h} | MEASURED_POSE_REPLAY | NO NEW RL CLAIM',font=font(23),fill='white')
    for i,s in enumerate(ss):
        x=i*800;draw.text((x+12,45),labels[i]+' | '+s['phase'][k],font=font(21),fill=('#f4ae59','#52cddd')[i]);draw.text((x+12,688),f"Final task: {int(s['success'])} | native: {int(s['native_success'])} | target error now: {s['signals']['Declared target distance (mm)'][k]:.1f}mm (limit40)",font=font(16),fill='white')
        visible=[d for d in s['decisions'] if d['step']<=k]
        first=visible[0] if visible else {};second=visible[1] if len(visible)>1 else {}
        completed=[r for r in s['skill_results'] if r['boundary_step']<=k and r['skill']=='PICK_TOOL']
        result=completed[-1].get('measurement',{}) if completed else {}
        if result:result={'physical_success':completed[-1]['physical_success'],'grasped':result.get('grasped'),'tcp_relative_tool_xyz_m':[round(v,4) for v in result.get('tcp_T_tool',[])[:3]],'remaining_steps':completed[-1].get('remaining_steps')}

        items=[('Skill 1 parameter',first.get('parameters','pending')),
               ('Measured SkillResult',result if completed else 'pending physical skill completion'),
               ('Successor preconditions',second.get('precondition',first.get('next_skill_requirement','pending'))),
               ('Skill 2 chosen parameter',second.get('parameters','pending'))]
        for row,(name,value) in enumerate(items):
            txt=name+': '+json.dumps(value,ensure_ascii=True,separators=(',',':'))
            for sub,line in enumerate(textwrap.wrap(txt,96)[:2]):draw.text((x+12,720+row*39+sub*17),line,font=font(14),fill='#dce8f5')
    for row,key in enumerate(['Cube/base distance (mm)','Tool/cube contact proxy (N)']):
        y=890+row*80;values=[v for s in ss for v in s['signals'][key] if v is not None];lo=min(values);hi=max(values);pad=(hi-lo or 1)*.05;lo-=pad;hi+=pad;draw.text((12,y),key,font=font(16),fill='white')
        for s,col in zip(ss,['#f4ae59','#52cddd']):draw.line([(220+j/h*1350,y+60-(v-lo)/(hi-lo)*55) for j,v in enumerate(s['signals'][key]) if v is not None],fill=col,width=2)
        draw.line((220+k/h*1350,y,220+k/h*1350,y+65),fill='white',width=2)
    return ann


def main():
    import torch,gymnasium as gym,mani_skill.envs
    from mani_skill.utils import sapien_utils
    from PIL import Image,ImageDraw
    ap=argparse.ArgumentParser();ap.add_argument('--left',type=Path,required=True);ap.add_argument('--right',type=Path,required=True);ap.add_argument('--left-label',required=True);ap.add_argument('--right-label',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    paths=[a.left.resolve(),a.right.resolve()];ts=[json.load(gzip.open(p,'rt')) for p in paths];assert ts[0]['summary']['initial_sha256']==ts[1]['summary']['initial_sha256'];h=len(ts[0]['steps']);assert all(len(t['steps'])==h for t in ts);assert h==100,'Preserve native PullCubeTool horizon'
    labels=[a.left_label,a.right_label];ss=[series(t,l) for t,l in zip(ts,labels)];seed=ts[0]['summary']['seed'];a.out.mkdir(parents=True,exist_ok=False)
    env=gym.make('PullCubeTool-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',human_render_camera_configs=dict(pose=sapien_utils.look_at([.55,-.80,.65],[.02,-.10,.08]),width=800,height=600,fov=.95));env.reset(seed=seed)
    camera=next(iter(env.unwrapped._human_render_cameras.values())).camera;camera_K=camera.get_intrinsic_matrix()[0].cpu().numpy();camera_E=camera.get_extrinsic_matrix()[0].cpu().numpy();cube_pixels=[];tool_pixels=[]
    native_sha=sha(Path(inspect.getfile(type(env.unwrapped))));rawp=pipe(a.out/'RAW.mp4',1600,600,20);annp=pipe(a.out/'ANNOTATED.mp4',1600,1060,20);maxerr=0.
    try:
        for k in range(h+1):
            raw=Image.new('RGB',(1600,600));boxes=[];goal_rings=[]
            for i,t in enumerate(ts):
                z=t['initial_measure'] if k==0 else t['steps'][k-1];state=copy.deepcopy(t['initial']);state['actors']['l_shape_tool'][0][:7]=z['tool'];state['actors']['cube'][0][:7]=z['cube'];state['articulations'][next(iter(state['articulations']))][0][13:22]=z['q'];target={g:{n:torch.tensor(v,dtype=torch.float32,device='cuda') for n,v in values.items()} for g,values in state.items()};env.unwrapped.set_state_dict(target);actual=env.unwrapped.get_state_dict();maxerr=max(maxerr,max(float((actual[g][n]-target[g][n]).abs().max()) for g in target for n in target[g]));rgb=env.unwrapped.render_rgb_array()[0].cpu().numpy();raw.paste(Image.fromarray(rgb[:,:,:3]),(800*i,0))
                from program_charger_mechanism import rotation
                corners=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [-.02,.02]])@rotation(z['cube'][3:]).T+np.asarray(z['cube'][:3]);cv=(camera_E@np.c_[corners,np.ones(8)].T).T;uv=(camera_K@cv[:,:3].T).T;uv=uv[:,:2]/uv[:,2:];cube_pixels.extend(uv.tolist());boxes.append((uv.min(0),uv.max(0)));assert (cv[:,2]>0).all()
                tc=np.array([[x,y,zz] for x in [0.,.2] for y in [-.025,.1] for zz in [-.025,.025]])@rotation(z['tool'][3:]).T+np.asarray(z['tool'][:3]);cv=(camera_E@np.c_[tc,np.ones(8)].T).T;uv=(camera_K@cv[:,:3].T).T;uv=uv[:,:2]/uv[:,2:];tool_pixels.extend(uv.tolist());assert (cv[:,2]>0).all()
                angles=np.linspace(0,2*np.pi,65);ring=np.c_[-.045+.04*np.cos(angles),t['summary']['target_y']+.04*np.sin(angles),np.full(65,.022),np.ones(65)];cv=(camera_E@ring.T).T;uv=(camera_K@cv[:,:3].T).T;goal_rings.append(uv[:,:2]/uv[:,2:])
            rawp.stdin.write(np.asarray(raw).tobytes());ann=annotation(raw,k,h,seed,ss,labels)
            draw=ImageDraw.Draw(ann)
            for i,(box,ring) in enumerate(zip(boxes,goal_rings)):
                col=['#f4ae59','#52cddd'][i];lo,hi=box;draw.rectangle((800*i+lo[0]-3,80+lo[1]-3,800*i+hi[0]+3,80+hi[1]+3),outline=col,width=2);draw.line([(800*i+x,80+y) for x,y in ring],fill='#f3f39a',width=2)
            annp.stdin.write(np.asarray(ann).tobytes())
            if k in [0,h//2,h]:raw.save(a.out/f'RAW_{k:03d}.png');ann.save(a.out/f'ANNOTATED_{k:03d}.png')
    finally:close_pipe(rawp);close_pipe(annp);env.close()
    assert maxerr<1e-5
    data={'task':'ARES-ToolTarget-v0','seed':seed,'fps':20,'horizon':h,'series':ss};(a.out/'DATA.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');viewer(a.out,data)
    for p,n in zip(paths,['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz']):shutil.copy2(p,a.out/n)
    shutil.copy2(__file__,a.out/'render_source.py')
    manifest={'source_kind':'MEASURED_POSE_REPLAY','physics_steps_added':0,'frames':h+1,'fps':20,'frame_mapping':'frame k = actual step k, timestamp k/20 seconds; initial frame0','max_state_readback_error':maxerr,'sources':[{'path':str(p),'sha256':sha(p),'execution_manifest':execution_manifest(p)} for p in paths],'camera_eye_world':[.55,-.80,.65],'camera_target_world':[.02,-.10,.08],'camera_fov_rad':.95,'camera_intrinsic':camera_K.tolist(),'camera_extrinsic':camera_E.tolist(),'all_tool_bound_corner_pixel_min':np.min(tool_pixels,axis=0).tolist(),'all_tool_bound_corner_pixel_max':np.max(tool_pixels,axis=0).tolist(),'all_cube_corner_pixel_min':np.min(cube_pixels,axis=0).tolist(),'all_cube_corner_pixel_max':np.max(cube_pixels,axis=0).tolist(),'native_model_source_sha256':native_sha,'render_source_sha256':sha(Path(__file__)),'outputs':{p.name:sha(p) for p in a.out.iterdir() if p.is_file()}}
    (a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'out':str(a.out),'frames':h+1,'maxerr':maxerr}))

if __name__=='__main__':main()
