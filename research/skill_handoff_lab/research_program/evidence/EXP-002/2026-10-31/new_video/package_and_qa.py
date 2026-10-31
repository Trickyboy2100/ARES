"""Package the source-linked native-model replay and audit exact temporal provenance."""
import gzip, hashlib, json, pathlib, re, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=pathlib.Path(__file__).resolve().parent
raw=pathlib.Path(sys.argv[1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['RAW.mp4','ANNOTATED.mp4','RAW.png','ANNOTATED.png','DATA.json','SYNC_VIEWER.html','MANIFEST.json','render_source.py']:
    shutil.copy2(raw/n,out/n)
m=json.loads((out/'MANIFEST.json').read_text());d=json.loads((out/'DATA.json').read_text())
qa={'physics_steps_added':0,'videos':{},'source_copy_sha':{},'measured_lateral_max_error_mm':{},'terminal':{}}
ts=[]
for i,s in enumerate(d['series']):
    path=pathlib.Path(s['source']);dst=out/('LEFT_SOURCE.json.gz' if i==0 else 'RIGHT_SOURCE.json.gz');shutil.copy2(path,dst)
    assert sha(dst)==s['source_sha256'];qa['source_copy_sha'][dst.name]=sha(dst)
    t=json.load(gzip.open(dst,'rt'));ts.append(t)
    error=max(abs(np.linalg.norm(np.asarray(z['obs'])[1:3]*25)-s['measured_lateral_mm'][z['step']]) for z in t['transitions'])
    assert error<.001;qa['measured_lateral_max_error_mm'][s['policy']]=error
    qa['terminal'][s['policy']]={'success':s['terminal_success'],'true_position_mm':1000*t['steps'][-1]['goal_position_error_m'],'true_lateral_mm':s['lateral_mm'][-1],'measured_lateral_mm':s['measured_lateral_mm'][-1]}
assert ts[0]['summary']['initial_sha256']==ts[1]['summary']['initial_sha256'];assert ts[0]['prefix_boundary']==ts[1]['prefix_boundary'];qa['same_initial_hash_and_prefix_boundary']=True
for n in ['RAW.mp4','ANNOTATED.mp4']:
    v=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate,width,height,duration','-show_entries','frame=best_effort_timestamp_time','-of','json',str(out/n)]))
    assert v['streams'][0]['nb_frames']=='201';pts=[float(f['best_effort_timestamp_time']) for f in v['frames']];assert max(abs(t-k/20) for k,t in enumerate(pts))<1e-8;qa['videos'][n]=v['streams'][0];qa['videos'][n]['pts_k_div_20_pass']=True
html=(out/'SYNC_VIEWER.html').read_text().replace('</body>', '''<script>
const QA={errors:[]};window.addEventListener('error',e=>QA.errors.push(e.message));
function finish(){const pre=document.createElement('pre');pre.id='qa-result';pre.textContent=JSON.stringify(QA);document.body.appendChild(pre);}
setTimeout(()=>{try{QA.series201=D.series.every(s=>s.axial_mm.length===201&&s.physical_step[200]===200);QA.nativeDuration=video.duration;mediaTime=7.549;QA.floorFrame150=currentStep()===150;const rect=canvas.getBoundingClientRect();canvas.onclick({clientX:rect.left+(.5*(canvas.width-132)+92)/canvas.width*rect.width});QA.clickSeeks100=Math.abs(video.currentTime-5)<1e-8;mediaTime=null;choice.value='raw';swap();QA.rawSwitch=video.src.endsWith('/RAW.mp4');windowSelect.value='80';windowSelect.onchange();canvas.onclick({clientX:rect.left+(.5*(canvas.width-132)+92)/canvas.width*rect.width});QA.contactClickSeeks140=Math.abs(video.currentTime-7)<1e-8;finish();}catch(e){QA.errors.push(e.stack);finish();}},2500);
</script></body>''')
(out/'QA_BROWSER.html').write_text(html)
browser=subprocess.run(['/opt/google/chrome/chrome','--headless','--disable-gpu','--no-sandbox','--disable-dev-shm-usage','--no-first-run','--user-data-dir=/tmp/ares-new-video-qa-chrome','--allow-file-access-from-files','--virtual-time-budget=5000','--dump-dom',(out/'QA_BROWSER.html').as_uri()],capture_output=True,text=True,check=True)
qa['browser']=json.loads(re.search(r'<pre id="qa-result">(.*?)</pre>',browser.stdout)[1]);assert not qa['browser']['errors'];assert all(qa['browser'][k] for k in ['series201','floorFrame150','clickSeeks100','rawSwitch','contactClickSeeks140'])
fig,axes=plt.subplots(6,1,figsize=(10,11),sharex=True,layout='constrained')
for ax,key,label in zip(axes,['axial_mm','lateral_mm','measured_lateral_mm','angle_rad','force_N','mode'],['True axial (mm)','True lateral (mm)','Measured lateral (mm)','Angle (rad)','Contact proxy (N)','Command mode']):
    for s,c in zip(d['series'],['#bc640f','#007c95']):ax.plot(s['physical_step'],s[key],color=c,label=s['policy'],lw=1.3)
    ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend(fontsize=9);axes[-1].set_yticks([0,1,2,3],['ALIGN','ADVANCE','HOLD','BACKOFF']);axes[-1].set_xlabel('Physical step (20 Hz; frame k = step k)');axes[-1].set_xlim(0,200);fig.suptitle('ARES-AdaptivePlug-v0 | seed 729000 | socket observation bias -1.2mm\nDATA_DERIVED_PLOT | deterministic controller comparison, NO RL GAIN')
fig.savefig(out/'TRACE.png',dpi=300);fig.savefig(out/'TRACE.svg')
for ax,key in zip(axes,['axial_mm','lateral_mm','measured_lateral_mm','angle_rad','force_N','mode']):
    values=[v for s in d['series'] for v in s[key][80:] if v is not None];lo=min(values);hi=max(values);pad=(hi-lo or 1)*.1;ax.set_ylim(lo-pad,hi+pad)
axes[-1].set_xlim(80,200);fig.savefig(out/'TRACE_CONTACT.png',dpi=300);plt.close(fig)
qa['max_pose_readback_error']=m['max_pose_readback_error'];(out/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
for p in out.iterdir():
    if p.is_file() and p.name!='MANIFEST.json':m['outputs'][p.name]=sha(p)
m['portable_trace_copies']=['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz'];(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(qa,indent=2))
