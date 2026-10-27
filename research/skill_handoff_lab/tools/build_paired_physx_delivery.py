"""Compose audited ACTUAL PhysX camera streams; never simulates or edits states.

Only time-aligned juxtaposition, factual labels and explicit terminal-frame hold
are added. All policies retain their independently executed underlying clocks.
"""
import argparse,csv,hashlib,html,json,math,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SCRIPT_BYTES=Path(__file__).read_bytes()
SCRIPT_SHA256=hashlib.sha256(SCRIPT_BYTES).hexdigest()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)
def csvwrite(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class Reader:
    def __init__(self,path,w=960,h=540):
        self.w=w;self.h=h;self.i=-1;self.image=None;self.cache={};self.keep=set()
        self.p=subprocess.Popen(['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-i',str(path),'-vf',f'scale={w}:{h}',
            '-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    def get(self,index):
        while self.i<index:
            data=self.p.stdout.read(self.w*self.h*3)
            if len(data)!=self.w*self.h*3:raise RuntimeError(f'Video ended before frame {index}')
            self.i+=1;self.image=Image.fromarray(np.frombuffer(data,np.uint8).reshape(self.h,self.w,3))
            if self.i in self.keep:self.cache[self.i]=self.image.copy()
        return self.image
    def close(self):
        self.p.stdout.close();self.p.wait(timeout=30)

def audit_path(run):
    if load(run/'manifest.json').get('tool_contract') and not (run/'INDEPENDENT_CLOSED_LOOP.json').exists():
        raise FileNotFoundError('Declared tool contract requires the completed FK/tool audit, not only the older contact check')
    return run/('INDEPENDENT_CLOSED_LOOP.json' if (run/'INDEPENDENT_CLOSED_LOOP.json').exists() else 'INDEPENDENT_TASK_AND_CONTACT.json')
def accepted(run,case):return next(x for x in load(audit_path(run))['cases'] if x['case_id']==case)['independent_task_pass']
def case_result(run,case):return next(x for x in load(run/'frozen_evaluation.json') if x['case_id']==case)
def initial_check(left,right):
    fields=['q','qd','object_position','object_quaternion_wxyz','object_linear_velocity','object_angular_velocity','paddle_position']
    for key in ['controller_reset_object_xy','controller_peak_displacement_m','controller_held_parameters','controller_execution_mask','controller_recontact_state','previous_effect_history','previous_contact_response_memory']:
        if key in left or key in right:
            assert key in left and key in right
            fields.append(key)
    delta={k:float(np.max(np.abs(np.array(left[k])-np.array(right[k])))) for k in fields}
    if max(delta.values())>1e-7:raise ValueError(f'Initial state mismatch: {delta}')
    return delta
def style_axes(ax):ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)

def trajectories(left,right,case,out,labels,frames):
    fig,axes=plt.subplots(2,3,figsize=(14,7.5));rows=[]
    for run,color,label,video in zip([left,right],['#657284','#157f91'],labels,frames):
        trace_sha=sha(run/'actual_physics_trace.npz');a=np.load(run/'actual_physics_trace.npz');cols={str(k):i for i,k in enumerate(a['columns'])};data=a['states'];x=data[data[:,:,cols['case_id']]==case]
        r=case_result(run,case);m=load(run/'manifest.json');phi=m['scene_specs'][r['env']].get('direction',0.)
        t=(x[:,cols['physics_tick']]-x[0,cols['physics_tick']])*float(a['sample_dt_s']);obj=x[:,6:9];paddle=x[:,9:12];goal=np.array(r['post_step_state']['goal'])
        q=x[:,16:20];yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2));yawerr=np.degrees(np.arctan2(np.sin(4*(yaw-phi)),np.cos(4*(yaw-phi)))/4)
        speed=np.linalg.norm(x[:,12:15],axis=1)*1000;omega=np.linalg.norm(x[:,31:34],axis=1);applied=np.linalg.norm(x[:,29:31],axis=1)*1000
        axes[0,0].plot((obj[:,0]-goal[0])*1000,(obj[:,1]-goal[1])*1000,color=color,label=label);axes[0,0].plot((paddle[:,0]-goal[0])*1000,(paddle[:,1]-goal[1])*1000,color=color,ls='--',alpha=.6)
        for ax,y in zip([axes[0,1],axes[0,2],axes[1,0],axes[1,1]],[yawerr,speed,omega,applied]):ax.plot(t,y,color=color,label=label)
        axes[1,2].plot([v['elapsed_physics_s'] for v in video],[v['max_robot_net_contact_N'] for v in video],color=color,label=label)
        h=r.get('handoff_state')
        if h:
            for ax in axes.flat:
                if ax is not axes[0,0]:ax.axvline(h['control_step']/60,color=color,ls=':',alpha=.6)
        for i in range(0,len(x),4):rows.append({'policy':label,'case_id':case,'time_s':float(t[i]),'phase':int(x[i,15]),'object_x_m':float(obj[i,0]),'object_y_m':float(obj[i,1]),'tool_x_m':float(paddle[i,0]),'tool_y_m':float(paddle[i,1]),'yaw_error_deg':float(yawerr[i]),'speed_mm_s':float(speed[i]),'angular_speed_rad_s':float(omega[i]),'applied_reference_delta_mm':float(applied[i]),'source_trace_sha256':trace_sha})
    axes[0,0].scatter([0],[0],marker='*',c='#203544',s=70);axes[0,0].set(xlabel='BODY X relative to final goal (mm)',ylabel='BODY Y relative to final goal (mm)',title='Actual object (solid) and tool (dashed) paths');axes[0,0].legend(frameon=False,fontsize=8)
    for ax,title,ylabel in zip([axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2]],['Orientation error (square symmetry)','Actual workpiece speed','Actual angular motion','Executed reference increment','Simulated contact proxy (30 Hz)'],['Yaw error (deg)','Speed (mm/s)','Angular speed (rad/s)','Reference delta (mm/control)','Maximum robot net contact (N)']):ax.set(title=title,xlabel='Physical time (s)',ylabel=ylabel)
    for ax in axes.flat:style_axes(ax)
    fig.suptitle(f'Case {case}: independent cold-start closed-loop physics, not state replay',fontsize=15,fontweight='bold')
    fig.text(.01,.005,'120 Hz recorded state; plotted CSV decimated 4x. Dotted lines: actual per-policy stage handoff. Contact is a net-force proxy, not calibrated sensing.\nSeparate execution may have different solver/cache histories; raw success and independent qualification remain distinct.',fontsize=8.5,color='#526676');fig.tight_layout(rect=(0,.06,1,.95))
    for ext in ['pdf','svg','png']:fig.savefig(out/f'case_{case:03d}_trajectories.{ext}',dpi=160,bbox_inches='tight');plt.close(fig) if ext=='png' else None
    csvwrite(out/f'case_{case:03d}_states.csv',rows)
    # v8 records actual link6 orientation in addition to object motion. Keep this
    # separate from commanded yaw; do not confuse a reference with execution.
    toolrows=[];toolfig,toolaxes=plt.subplots(1,2,figsize=(11,3.7));have_tool=False
    for run,color,label in zip([left,right],['#657284','#157f91'],labels):
        archive=np.load(run/'actual_physics_trace.npz');columns={str(k):i for i,k in enumerate(archive['columns'])}
        if 'actual_link6_qw' not in columns:continue
        have_tool=True;data=archive['states'];x=data[data[:,:,columns['case_id']]==case];m=load(run/'manifest.json');reference=np.array(m['tool_contract']['reference_level_rotation'])
        quat=x[:,[columns['actual_link6_q'+component] for component in ['w','x','y','z']]]
        w,qx,qy,qz=quat.T
        rotation=np.stack([1-2*(qy*qy+qz*qz),2*(qx*qy-w*qz),2*(qx*qz+w*qy),2*(qx*qy+w*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-w*qx),2*(qx*qz-w*qy),2*(qy*qz+w*qx),1-2*(qx*qx+qy*qy)],axis=-1).reshape(-1,3,3)
        relative=rotation@reference.T;actual=np.arctan2(relative[:,1,0],relative[:,0,0]);tilt=np.arccos(np.clip(relative[:,2,2],-1,1));command=x[:,columns['tool_yaw_command_rad']]
        t=(x[:,columns['physics_tick']]-x[0,columns['physics_tick']])*float(archive['sample_dt_s'])
        toolaxes[0].plot(t,np.degrees(actual),c=color,label=label+' actual');toolaxes[0].plot(t,np.degrees(command),c=color,ls='--',alpha=.6)
        toolaxes[1].plot(t,np.degrees(tilt),c=color,label=label)
        tool_trace_sha=sha(run/'actual_physics_trace.npz')
        for i in range(0,len(x),4):toolrows.append({'policy':label,'case_id':case,'time_s':float(t[i]),'actual_link6_yaw_deg':float(np.degrees(actual[i])),'commanded_yaw_deg':float(np.degrees(command[i])),'actual_link6_tilt_deg':float(np.degrees(tilt[i])),'source_trace_sha256':tool_trace_sha})
    if have_tool:
        toolaxes[0].set(title='Actual tool yaw (solid) vs command (dashed)',ylabel='Yaw relative to calibrated level frame (deg)',xlabel='Physical time (s)');toolaxes[0].legend(fontsize=7,frameon=False)
        toolaxes[1].set(title='Actual tool horizontal tilt',ylabel='Horizontal tilt (deg)',xlabel='Physical time (s)')
        for ax in toolaxes:style_axes(ax)
        toolfig.suptitle(f'Case {case}: tool motion from recorded actual link6 quaternion, not only action output',fontsize=12,fontweight='bold');toolfig.tight_layout(rect=(0,0,1,.92))
        for ext in ['pdf','svg','png']:toolfig.savefig(out/f'case_{case:03d}_tool_contract.{ext}',dpi=160,bbox_inches='tight')
        csvwrite(out/f'case_{case:03d}_tool_contract.csv',toolrows)
    plt.close(toolfig)

def compose_pair(a,case):
    left=a.left.resolve();right=a.right.resolve();out=a.output.resolve()/f'{a.iteration}_case_{case:03d}'
    if (out/'PAIR_PROVENANCE.json').exists():raise FileExistsError('Completed comparison is immutable; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'composition_source.py').write_bytes(SCRIPT_BYTES)
    metadata=[load(d/'VIDEO_CAPTURE_PROVENANCE.json') for d in [left,right]];case_meta=[m['cases'][str(case)] for m in metadata]
    for d,m in zip([left,right],metadata):
        assert (d/'RUN_COMPLETE.json').exists() and m['state_writes_during_actions']==0
        assert load(d/'APPEARANCE_IDENTITY.json')['equal']
    initial=initial_check(case_meta[0]['initial_state'],case_meta[1]['initial_state'])
    manifests=[load(d/'manifest.json') for d in [left,right]]
    assert manifests[0]['asset_sha256']==manifests[1]['asset_sha256'] and manifests[0]['scene_specs']==manifests[1]['scene_specs']
    assert manifests[0].get('success_contract')==manifests[1].get('success_contract') and manifests[0].get('tool_contract')==manifests[1].get('tool_contract')
    assert load(left/'VIDEO_SOURCE_PATCH.json')['base_source_sha256']==load(right/'VIDEO_SOURCE_PATCH.json')['base_source_sha256']
    frames=[load(d/m['frame_records']) for d,m in zip([left,right],case_meta)];times=[np.array([r['elapsed_physics_s'] for r in f]) for f in frames]
    ends=[float(t[-1]) for t in times];results=[case_result(d,case) for d in [left,right]];passes=[accepted(d,case) for d in [left,right]]
    readers=[Reader(d/m['mp4']) for d,m in zip([left,right],case_meta)]
    contact=[r['elapsed_physics_s'] for f in frames for r in f if r['max_robot_net_contact_N']>.02]
    handoffs=[r['elapsed_physics_s'] for f in frames for r in f if r['phase']==1]
    event_times=sorted(set([0.,min(contact,default=max(ends)*.2),min(handoffs,default=max(ends)*.45),max(ends)*.75,max(ends)]))
    for reader,t in zip(readers,times):reader.keep={max(0,int(np.searchsorted(t,tt,side='right')-1)) for tt in event_times}
    mp4=out/'paired_actual_physx.mp4';encoder=subprocess.Popen(['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],stdin=subprocess.PIPE)
    intro=1.5;duration=max(15.,intro+max(ends)+2.);timeline=[]
    for index in range(math.ceil(duration*30)):
        video_t=index/30;physical_t=min(max(0.,video_t-intro),max(ends));canvas=Image.new('RGB',(1920,1080),'#edf2f5');draw=ImageDraw.Draw(canvas)
        draw.text((35,20),f'{a.iteration} | Case {case} | Actual same-case PhysX comparison',fill='#203544',font=font(38,True))
        if a.iteration=='R3':draw.text((35,65),'Pre-specified training seed31. Documentary example, not a cross-seed performance summary.',fill='#526676',font=font(22))
        for side,(reader,t,f,label,passed,res) in enumerate(zip(readers,times,frames,[a.left_label,a.right_label],passes,results)):
            j=max(0,int(np.searchsorted(t,physical_t,side='right')-1));j=min(j,len(f)-1);image=reader.get(j);x=side*960
            canvas.paste(image,(x,145));draw.text((x+25,90),label[:48],fill='#203544',font=font(29,True))
            state=f[j];ended=physical_t>=ends[side];clock=min(physical_t,ends[side]);status=('PASS' if passed else 'FAIL') if ended else 'RUNNING'
            draw.text((x+28,710),f'Physics {clock:5.2f}s   Stage {state["phase"]}   {status}',fill=('#168268' if passed else '#b45a41') if ended else '#203544',font=font(31,True))
            draw.text((x+28,770),f'Goal distance {state["final_goal_distance_m"]*1000:6.2f} mm',fill='#334e5d',font=font(28))
            draw.text((x+28,815),f'Yaw {math.degrees(state["yaw_error_rad"]):6.2f} deg  |  lateral {state["lateral_error_m"]*1000:6.2f} mm',fill='#334e5d',font=font(25))
            draw.text((x+28,860),f'Contact proxy {state["max_robot_net_contact_N"]:6.2f} N',fill='#334e5d',font=font(25))
            if ended:draw.text((x+28,910),f'END / held terminal frame ({res["failure_reason"]})',fill='#8b5d3d',font=font(24,True))
        draw.line((960,80,960,955),fill='#b6c2cb',width=2)
        foot='SYNTHETIC state-based Isaac Sim 5.1 / Lab 2.3 | independent cold starts | no field / grasp claim'
        draw.text((28,980),foot,fill='#526676',font=font(24));draw.text((28,1022),f'Initial states matched | L trace {metadata[0]["physics_trace_sha256"][:12]} | R trace {metadata[1]["physics_trace_sha256"][:12]} | intro/final holds are display-only',fill='#526676',font=font(20))
        encoder.stdin.write(np.asarray(canvas).tobytes())
        timeline.append({'video_frame':index,'video_time_s':video_t,'shared_physics_time_s':physical_t,'left_frame_index':max(0,int(np.searchsorted(times[0],physical_t,side='right')-1)),'right_frame_index':max(0,int(np.searchsorted(times[1],physical_t,side='right')-1))})
    encoder.stdin.close();assert encoder.wait(timeout=120)==0
    for r in readers:r.close()
    strip=Image.new('RGB',(len(event_times)*384,600),'white');draw=ImageDraw.Draw(strip)
    draw.text((15,10),f'{a.iteration} / case {case}: actual time-aligned closed-loop frames',fill='#203544',font=font(24,True))
    subtitle=f'Independent qualification: left {"PASS" if passes[0] else "FAIL"}, right {"PASS" if passes[1] else "FAIL"}; independent cold starts, original physical clocks'
    if a.iteration=='R3':subtitle='Pre-specified seed31 (not a cross-seed summary). '+subtitle
    draw.text((15,42),subtitle,fill='#526676',font=font(18))
    for side,(reader,t,label) in enumerate(zip(readers,times,[a.left_label,a.right_label])):
        for k,tt in enumerate(event_times):
            j=max(0,int(np.searchsorted(t,tt,side='right')-1));im=reader.cache[j].resize((384,216));y=108+side*245;strip.paste(im,(k*384,y))
            status='END/HOLD' if tt>ends[side]+1e-6 else ('END' if tt>=ends[side]-1e-6 else f'Stage {frames[side][j]["phase"]}')
            draw.text((k*384+7,y-23),f'{label[:18]} | {min(tt,ends[side]):.2f}s | {status}',fill='#203544',font=font(15,True))
    draw.text((15,578),'Frames after an earlier termination are explicitly held; no new physics or trajectory playback. SYNTHETIC ARES tasks.',fill='#526676',font=font(15))
    strip.save(out/'filmstrip.png')
    fig=plt.figure(figsize=(12.8,strip.height/150));ax=fig.add_axes((0,0,1,1));ax.imshow(strip);ax.axis('off')
    fig.savefig(out/'filmstrip.pdf',dpi=150);plt.close(fig)
    subprocess.run(['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(mp4),'-vf','fps=5,scale=768:-1','-loop','0',str(out/'preview.gif')],check=True)
    csvwrite(out/'video_physics_timeline.csv',timeline);trajectories(left,right,case,out,[a.left_label,a.right_label],frames)
    provenance={'iteration':a.iteration,'case_id':case,'scope':'ACTUAL_PHYSX_TWO_POLICY_INDEPENDENT_COLD_START','left_run':str(left),'right_run':str(right),
                'left_label':a.left_label,'right_label':a.right_label,'left_independent_pass':passes[0],'right_independent_pass':passes[1],
                'left_raw_success':results[0]['success'],'right_raw_success':results[1]['success'],'left_terminal_time_s':ends[0],'right_terminal_time_s':ends[1],
                'initial_component_maximum_differences':initial,'hidden_solver_state_equal_claimed':False,'source_capture_provenance_sha256':[sha(d/'VIDEO_CAPTURE_PROVENANCE.json') for d in [left,right]],
                'physics_trace_sha256':[m['physics_trace_sha256'] for m in metadata],'action_trace_sha256':[m['action_log_sha256'] for m in metadata],
                'base_source_sha256':[load(d/'VIDEO_SOURCE_PATCH.json')['base_source_sha256'] for d in [left,right]],
                'asset_sha256':manifests[0]['asset_sha256'],'checkpoint_sha256':[m.get('loaded_policy_sha256') for m in manifests],
                'training_seed':[m['seed'] for m in manifests],
                'independent_audit_sha256':[sha(audit_path(d)) for d in [left,right]],
                'fps':30,'resolution':[1920,1080],'intro_hold_s':intro,'output_duration_s':duration,
                'selection_protocol':'Predeclared category and first matching ID; all-case paired table retained; no best checkpoint selection',
                'state_writes_during_actions':0,'render_extra_physics_steps':0,'script_sha256':SCRIPT_SHA256,
                'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}}
    (out/'PAIR_PROVENANCE.json').write_text(json.dumps(provenance,indent=2));return provenance

def main(a):
    a.output.mkdir(parents=True,exist_ok=True);records=[compose_pair(a,c) for c in a.cases]
    release=a.release;release.mkdir(parents=True,exist_ok=True);(release/'figures').mkdir(exist_ok=True);(release/'tables').mkdir(exist_ok=True)
    left_cases={r['case_id']:r for r in load(a.left/'frozen_evaluation.json')};right_cases={r['case_id']:r for r in load(a.right/'frozen_evaluation.json')}
    left_audit={r['case_id']:r for r in load(audit_path(a.left))['cases']};right_audit={r['case_id']:r for r in load(audit_path(a.right))['cases']}
    assert left_cases.keys()==right_cases.keys()==left_audit.keys()==right_audit.keys()
    paired=[]
    for case in sorted(left_cases):
        l=left_cases[case];r=right_cases[case];paired.append({'iteration':a.iteration,'case_id':case,'family':l['family'],'scene_group':l['scene_group'],
            'left_raw_success':l['success'],'right_raw_success':r['success'],'left_independent_pass':left_audit[case]['independent_task_pass'],
            'right_independent_pass':right_audit[case]['independent_task_pass'],'left_failure_reason':l['failure_reason'],'right_failure_reason':r['failure_reason'],
            'left_physics_time_s':l['steps']/60,'right_physics_time_s':r['steps']/60,'selected_for_video':case in a.cases})
    csvwrite(release/'tables'/f'{a.iteration}_all_paired_cases.csv',paired)
    existing=load(release/'VIDEO_PROVENANCE.json') if (release/'VIDEO_PROVENANCE.json').exists() else {'comparisons':[]}
    for record in records:
        d=a.output/f"{record['iteration']}_case_{record['case_id']:03d}"
        # Small documentary artifacts only; raw MP4/NPZ remain non-Git.
        for p in d.iterdir():
            if p.name in ['preview.gif','filmstrip.png','filmstrip.pdf'] or p.name.startswith('case_'):
                dest=release/'figures'/f"{record['iteration']}_case{record['case_id']:03d}_{p.name}";dest.write_bytes(p.read_bytes())
        record['mp4_local_path']=str((d/'paired_actual_physx.mp4').resolve());record['public_preview']=str((release/'figures'/f"{record['iteration']}_case{record['case_id']:03d}_preview.gif").name)
        prior=[r for r in existing['comparisons'] if (r['iteration'],r['case_id'])==(record['iteration'],record['case_id'])]
        if prior:
            assert a.replace_display_only, 'Duplicate case requires explicit display-only rebuild flag'
            assert len(prior)==1 and prior[0]['physics_trace_sha256']==record['physics_trace_sha256'] and prior[0]['action_trace_sha256']==record['action_trace_sha256']
            record['display_only_rebuild_of_mp4']=prior[0]['mp4_local_path']
            existing['comparisons']=[r for r in existing['comparisons'] if r not in prior]
        existing['comparisons'].append(record)
    existing['comparisons'].sort(key=lambda r:(r['iteration'],r['case_id']))
    (release/'VIDEO_PROVENANCE.json').write_text(json.dumps(existing,indent=2))
    rows=[{'iteration':r['iteration'],'case_id':r['case_id'],'baseline':r['left_label'],'candidate':r['right_label'],'baseline_pass':r['left_independent_pass'],'candidate_pass':r['right_independent_pass'],'mp4_local_path':r['mp4_local_path'],'left_trace_sha256':r['physics_trace_sha256'][0],'right_trace_sha256':r['physics_trace_sha256'][1]} for r in existing['comparisons']]
    csvwrite(release/'comparison_index.csv',rows)
    cards=[]
    for r in existing['comparisons']:
        video=os.path.relpath(r['mp4_local_path'],release);preview='figures/'+r['public_preview'];cards.append(f'<section><h2>{html.escape(r["iteration"])} / case {r["case_id"]}</h2><p>{html.escape(r["left_label"])}: {r["left_independent_pass"]} | {html.escape(r["right_label"])}: {r["right_independent_pass"]}</p><video controls preload="none" width="960" poster="{preview}"><source src="{video}" type="video/mp4"></video><p>LOCAL_ONLY original: <code>{html.escape(r["mp4_local_path"])}</code></p></section>')
    (release/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>ARES actual PhysX comparisons</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#203544}section{padding:20px 0;border-bottom:1px solid #ddd}video{max-width:100%}code{font-size:11px;word-break:break-all}</style><h1>Actual same-case PhysX comparisons</h1><p>SYNTHETIC ARES tasks. Local artifacts, not public hosted video. Independent post-step qualification, no real robot or field data.</p><p>For browser playback serve the workspace locally: <code>python3 -m http.server 8765 --bind 127.0.0.1 --directory /home/andyee/robot-research/ARES-Skilllab</code>; then open this index under that local root.</p>'+''.join(cards)+'</html>')
    print(json.dumps({'comparisons_added':len(records),'total':len(existing['comparisons']),'index':str(release/'index.html')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--release',type=Path,required=True)
    p.add_argument('--iteration',required=True);p.add_argument('--cases',type=int,nargs='+',required=True);p.add_argument('--left-label',required=True);p.add_argument('--right-label',required=True);p.add_argument('--replace-display-only',action='store_true');main(p.parse_args())
