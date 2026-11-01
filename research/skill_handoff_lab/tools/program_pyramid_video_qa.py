"""CPU source/frame/browser QA for native250 controller-repair evidence."""
import argparse,gzip,json,re,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from program_charger_video_sync import sha
from program_pyramid_video import series,CLAIM

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();d=json.loads((out/'DATA.json').read_text());m=json.loads((out/'MANIFEST.json').read_text());q={'claim':CLAIM,'physics_steps_added':0,'sources':[],'videos':{}};ts=[]
    for i,n in enumerate(['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz']):
        p=out/n;assert sha(p)==m['sources'][i]['sha256'];t=json.load(gzip.open(p,'rt'));ts.append(t);expected=series(t,d['series'][i]['label']);assert expected==d['series'][i];q['sources'].append({'sha256':sha(p),'success':t['summary']['success'],'decision_steps':[z['boundary_step'] for z in t['decisions']],'verified_boundary_steps':[z['step'] for z in t['summary']['boundaries'] if z['valid']],'error':t['summary']['error']})
    assert ts[0]['summary']['initial_sha256']==ts[1]['summary']['initial_sha256'];q['same_initial_hash']=True
    for n in ['RAW.mp4','ANNOTATED.mp4']:
        v=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate,width,height,duration','-show_entries','frame=best_effort_timestamp_time','-of','json',str(out/n)]));assert int(v['streams'][0]['nb_frames'])==251;assert max(abs(float(f['best_effort_timestamp_time'])-k/20) for k,f in enumerate(v['frames']))<1e-8;q['videos'][n]=dict(v['streams'][0],all_pts_pass=True)
    html=(out/'SYNC_VIEWER.html').read_text().replace('</main>',r'''</main><script>window.addEventListener('load',()=>setTimeout(()=>{const Q={};try{Q.duration=v.duration;media=5.049;Q.floor100=step()===100;const r=c.getBoundingClientRect();c.onclick({clientX:r.left+(210+570)/c.width*r.width});Q.click125=Math.abs(v.currentTime-6.25)<1e-5;const bs=document.querySelectorAll('#events button');Q.source_decision_count=bs.length===D.series.reduce((n,s)=>n+s.decisions.length,0);bs[bs.length-1].click();Q.last_decision_seek=Math.abs(v.currentTime-D.series[1].decisions.at(-1).step/20)<1e-5;choice.value='RAW';swap();Q.raw_switch=v.src.endsWith('/RAW.mp4');media=0;draw();Q.no_future_result=!document.querySelector('#left').textContent.includes('"physical_success"');}catch(e){Q.error=e.stack}const p=document.createElement('pre');p.id='qa-result';p.textContent=JSON.stringify(Q);document.body.appendChild(p)},2500));</script>''');(out/'QA_BROWSER.html').write_text(html)
    b=subprocess.run(['/opt/google/chrome/chrome','--headless','--disable-gpu','--no-sandbox','--disable-dev-shm-usage','--no-first-run','--user-data-dir=/tmp/ares-pyramid-qa','--allow-file-access-from-files','--virtual-time-budget=5000','--dump-dom',(out/'QA_BROWSER.html').as_uri()],capture_output=True,text=True,check=True);match=re.search(r'<pre id="qa-result">(.*?)</pre>',b.stdout);assert match,b.stderr;q['browser']=json.loads(match[1]);assert 'error' not in q['browser'],q['browser'];assert all(q['browser'][key] for key in ['floor100','click125','source_decision_count','last_decision_seek','raw_switch','no_future_result']),q['browser']
    fig,axs=plt.subplots(5,1,figsize=(11,11),sharex=True,layout='constrained')
    for ax,key in zip(axs,d['series'][0]['signals']):
        for s,col in zip(d['series'],['#b66b14','#00899c']):
            ax.plot(np.arange(251),s['signals'][key],label=s['label'],color=col)
            for r in s['skill_results']:ax.axvline(r['boundary_step'],color=col,ls=':' if r['physical_success'] else '--',alpha=.65)
        ax.set_ylabel(key);ax.grid(alpha=.2)
    axs[0].legend(fontsize=9);axs[-1].set_xlim(0,250);axs[-1].set_xlabel('Actual native physical step (20Hz); dotted: passed boundary, dashed: failed boundary');fig.suptitle('Native StackPyramid-v1 / full250 / DATA_DERIVED_PLOT\n'+CLAIM);fig.savefig(out/'TRACE.png',dpi=300);fig.savefig(out/'TRACE.svg');plt.close(fig)
    q['max_state_readback_error']=m['max_state_readback_error'];q['all_three_cube_bounds_inside_frame']=all(0<=lo<=hi<=lim for lo,hi,lim in zip(m['all_cube_corner_pixel_min'],m['all_cube_corner_pixel_max'],[800,600]));assert q['all_three_cube_bounds_inside_frame'];(out/'QA.json').write_text(json.dumps(q,indent=2)+'\n');m['outputs']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='MANIFEST.json'};(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(q,indent=2))
if __name__=='__main__':main()
