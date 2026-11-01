"""CPU QA of full physical evidence: frames, source measurements and browser seeking."""
import argparse,gzip,json,re,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from program_charger_video_sync import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();d=json.loads((out/'DATA.json').read_text());m=json.loads((out/'MANIFEST.json').read_text());qa={'physics_steps_added':0,'sources':[],'videos':{}}
    ts=[]
    for i,name in enumerate(['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz']):
        p=out/name;assert sha(p)==m['sources'][i]['sha256'];t=json.load(gzip.open(p,'rt'));ts.append(t);states=[t['initial_measure'],*t['steps']];s=d['series'][i];assert len(states)==d['horizon']+1
        assert [int(z['grasped']) for z in states]==s['signals']['Measured tool grasp (0/1)']
        assert np.max(np.abs(np.array([z['cube_base_distance_m']*1000 for z in states])-s['signals']['Cube/base distance (mm)']))==0
        assert s['signals']['Tool/cube contact proxy (N)'][0] is None
        assert s['initial_contact_readback_N_unvalidated']==t['initial_measure']['tool_cube_force_N']
        assert s['signals']['Tool/cube contact proxy (N)'][1:]==[z['tool_cube_force_N'] for z in t['steps']]
        assert [z['boundary_step'] for z in t['decisions']]==[z['step'] for z in s['decisions']]
        qa['sources'].append({'sha256':sha(p),'frames':len(states),'decision_steps':[z['step'] for z in s['decisions']],'initial_contact_readback_N_unvalidated':s['initial_contact_readback_N_unvalidated'],'contact_data_starts_at_physics_step':1,'terminal_success':s['success'],'terminal_cube_base_distance_mm':s['signals']['Cube/base distance (mm)'][-1]})
    assert ts[0]['summary']['initial_sha256']==ts[1]['summary']['initial_sha256'];qa['same_initial_hash']=True
    for n in ['RAW.mp4','ANNOTATED.mp4']:
        v=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate,width,height,duration','-show_entries','frame=best_effort_timestamp_time','-of','json',str(out/n)]));assert int(v['streams'][0]['nb_frames'])==d['horizon']+1;pts=[float(f['best_effort_timestamp_time']) for f in v['frames']];assert max(abs(t-k/d['fps']) for k,t in enumerate(pts))<1e-8;qa['videos'][n]=dict(v['streams'][0],all_frame_pts_verified=True)
    html=(out/'SYNC_VIEWER.html').read_text().replace('</main>','''</main><script>window.addEventListener('load',()=>setTimeout(()=>{const Q={};try{Q.duration=v.duration;media=2.549;Q.floor50=step()===50;const r=c.getBoundingClientRect();c.onclick({clientX:r.left+(210+570)/c.width*r.width});Q.click50=Math.abs(v.currentTime-2.5)<1e-8;const buttons=document.querySelectorAll('#events button');Q.four_decisions=buttons.length===4;buttons[3].click();Q.second_decision_seek_time=v.currentTime;Q.second_decision_expected_time=D.series[1].decisions[1].step/20;Q.second_decision_seek=Math.abs(v.currentTime-Q.second_decision_expected_time)<1e-5;choice.value='RAW';swap();Q.raw_switch=v.src.endsWith('/RAW.mp4');media=0;draw();Q.no_future_result=!document.querySelector('#left').textContent.includes('"physical_success"');media=100/20;Q.note='No physics; browser interface only';}catch(e){Q.error=e.stack}const p=document.createElement('pre');p.id='qa-result';p.textContent=JSON.stringify(Q);document.body.appendChild(p)},2500));</script>''');(out/'QA_BROWSER.html').write_text(html)
    b=subprocess.run(['/opt/google/chrome/chrome','--headless','--disable-gpu','--no-sandbox','--disable-dev-shm-usage','--no-first-run','--user-data-dir=/tmp/ares-multiskill-video-qa','--allow-file-access-from-files','--virtual-time-budget=5000','--dump-dom',(out/'QA_BROWSER.html').as_uri()],capture_output=True,text=True,check=True);match=re.search(r'<pre id="qa-result">(.*?)</pre>',b.stdout);assert match,b.stderr;qa['browser']=json.loads(match[1]);assert 'error' not in qa['browser'],qa['browser'];assert all(qa['browser'][key] for key in ['floor50','click50','four_decisions','second_decision_seek','raw_switch','no_future_result']),qa['browser']
    count=len(d['series'][0]['signals']);fig,axes=plt.subplots(count,1,figsize=(10,2*count),sharex=True,layout='constrained')
    for ax,key in zip(axes,d['series'][0]['signals']):
        for s,color in zip(d['series'],['#a5600c','#087d93']):
            ax.plot(np.arange(d['horizon']+1),s['signals'][key],label=s['label'],color=color)
            for decision in s['decisions'][1:]:ax.axvline(decision['step'],color=color,ls=':',alpha=.7)
        ax.set_ylabel(key);ax.grid(alpha=.2)
    axes[0].axhline(600,color='grey',ls='--',label='Native terminal threshold 600mm');axes[0].legend(fontsize=7);axes[-1].set_xlim(0,d['horizon']);axes[-1].set_xlabel('Actual physical control step (20Hz); dotted lines: second skill decision');fig.suptitle('PullCubeTool-v1: two real parameter decisions / native100\nDATA_DERIVED_PLOT / deterministic controls, NO NEW RL CLAIM');fig.savefig(out/'TRACE.png',dpi=300);fig.savefig(out/'TRACE.svg');plt.close(fig)
    qa['max_state_readback_error']=m['max_state_readback_error']
    if 'all_cube_corner_pixel_min' in m:
        qa['all_cube_and_tool_bounds_in_frame']=all(0<=lo<=hi<=limit for name in ['cube','tool_bound'] for lo,hi,limit in zip(m['all_'+name+'_corner_pixel_min'],m['all_'+name+'_corner_pixel_max'],[800,600]))
        assert qa['all_cube_and_tool_bounds_in_frame']
    (out/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    m['outputs']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='MANIFEST.json'};(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
