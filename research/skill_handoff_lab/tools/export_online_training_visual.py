"""Export actual camera/optimizer/120Hz evidence from a separate online rerun.

No physics execution, state replay or generated frames: only read original RGB,
resample by its recorded clock, add factual labels and explicit terminal hold.
"""
import argparse,csv,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_paired_physx_delivery import Reader,sha,font,csvwrite
from paired_physx_video_hooks import Encoder

def load(p):return json.loads(p.read_text())

def main(a):
    assert (a.run/'RUN_COMPLETE.json').exists()
    m=load(a.run/'manifest.json');cap=load(a.run/'VIDEO_CAPTURE_PROVENANCE.json')
    assert m['online_training_interactions']==12288 and m['num_envs']==16
    assert cap['matrix_overview']['is_training'] and load(a.run/'APPEARANCE_IDENTITY.json')['equal']
    events=load(a.run/'LIVE_OPTIMIZER_EVENTS.json');ev=events['events'];assert len(ev)==3
    last=events['initial_parameter_sha256']
    for i,e in enumerate(ev,1):
        assert e['optimizer_update_index']==i and e['online_interactions']==4096*i
        assert e['parameters_changed'] and e['parameter_sha256_before']==last
        assert e['parameter_sha256_before']!=e['parameter_sha256_after']
        last=e['parameter_sha256_after']
    frames=load(a.run/'MATRIX_LIVE_TRAINING_FRAMES.json');montages=load(a.run/'LIVE_TRAINING_MONTAGES.json')
    assert len(montages)==2 and all(f['num_envs']==16 for f in frames)
    for e in ev:
        before=next(f for f in frames if f['capture_event']==f'PRE_OPTIMIZER_{e["optimizer_update_index"]}')
        after=next(f for f in frames if f['capture_event']==f'POST_OPTIMIZER_{e["optimizer_update_index"]}')
        assert before['physics_tick']==after['physics_tick']==e['physics_tick']
        assert before['environments']==after['environments'],'Optimizer must not move physics'
        assert before['parameter_sha256']==e['parameter_sha256_before'] and after['parameter_sha256']==e['parameter_sha256_after']
    data=np.load(a.run/'TRAINING_120HZ_STATE.npz');states=data['states'];columns=[str(c) for c in data['columns']]
    assert states.shape[:2]==(1536,16) and np.isfinite(states).all()
    assert np.array_equal(states[:,0,0],np.arange(1536)) and float(data['sample_dt_s'])==1/120
    c={k:i for i,k in enumerate(columns)}
    assert np.max(np.ptp(states[:,:,c['obj_x']],axis=0))>.01,'Real physical object motion required'
    a.output.mkdir(parents=True,exist_ok=False);figs=a.release/'figures';tables=a.release/'tables';outputs=[]
    source=a.run/cap['matrix_overview']['mp4'];assert sha(source)==cap['matrix_overview']['mp4_sha256']
    reader=Reader(source,1920,1080);encoder=Encoder(a.output/'actual_online_training_16env.mp4','/usr/bin/ffmpeg',fps=12)
    times=np.array([f['elapsed_physics_s'] for f in frames]);end=float(times[-1]);preview=[]
    for t in np.arange(0,end+3+1e-9,1/12):
        index=min(len(frames)-1,max(0,int(np.searchsorted(times,min(t,end),side='right')-1)))
        f=frames[index];im=reader.get(index).copy();draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,1920,113),fill='#182e3c')
        draw.text((24,10),'REAL ONLINE TRAINING RERUN | 16 independent PhysX environments',font=font(32,True),fill='white')
        draw.text((24,50),f'Physical time {f["elapsed_physics_s"]:.2f}s | interactions {f["actual_env_control_steps"]:,} | rollout updates {f["optimizer_updates_completed"]}',font=font(25),fill='white')
        draw.text((24,83),'12,288 extra interactions; not formal performance. Single wide camera; no physics replay.',font=font(21),fill='#b8d8dc')
        if t>end:
            draw.rectangle((0,1020,1920,1080),fill='#182e3c');draw.text((24,1035),'END / DISPLAY HOLD: no further physics or policy updates',font=font(25),fill='white')
        encoder.write(np.array(im))
        if int(round(t*12))%2==0:preview.append(im.resize((768,432)))
    encoder.close();reader.close();outputs.append(a.output/'actual_online_training_16env.mp4')
    gif=figs/'R3_actual_online_training_16env.gif';preview[0].save(gif,save_all=True,append_images=preview[1:],duration=167,loop=0);outputs.append(gif)
    selected=montages[-1];assert len(selected['views'])==16
    original=a.run/selected['file'];assert sha(original)==selected['sha256']
    png=figs/'R3_online_training_16env_montage.png';png.write_bytes(original.read_bytes());outputs.append(png)
    for view in selected['views']:assert sha(a.run/view['source_png'])==view['sha256']
    wide=next(f for f in reversed(frames) if 'png' in f);widepng=figs/'R3_online_training_wide.png';widepng.write_bytes((a.run/wide['png']).read_bytes());outputs.append(widepng)
    # 2x2 optimizer evidence: real RGB before/after at two unchanged ticks.
    evidence=Image.new('RGB',(1920,1200),'white');draw=ImageDraw.Draw(evidence)
    draw.text((20,12),'Actual optimizer updates: same physical state; different network tensor hashes',font=font(30,True),fill='#203544')
    for row,update in enumerate([1,3]):
        for col,tag in enumerate(['PRE','POST']):
            f=next(f for f in frames if f['capture_event']==f'{tag}_OPTIMIZER_{update}')
            original=a.run/f['png'];assert sha(original)==f['png_sha256']
            image=Image.open(original).resize((960,540));evidence.paste(image,(col*960,70+row*565))
            draw.rectangle((col*960,70+row*565,col*960+960,70+row*565+48),fill='#203544')
            draw.text((col*960+12,78+row*565),f'{tag} update{update}: tick {f["physics_tick"]}, hash {f["parameter_sha256"][:12]}',font=font(23),fill='white')
    epng=figs/'R3_optimizer_before_after.png';evidence.save(epng);outputs.append(epng)
    for image,stem in [(png,'R3_online_training_16env_montage'),(epng,'R3_optimizer_before_after')]:
        fig,ax=plt.subplots(figsize=(7.2,4.5 if 'before' in stem else 4.05));ax.imshow(Image.open(image));ax.axis('off');fig.tight_layout(pad=0)
        p=figs/(stem+'.pdf');fig.savefig(p,bbox_inches='tight',pad_inches=0);plt.close(fig);outputs.append(p)
    rows=[]
    for f in frames:
        for i,s in enumerate(f['environments']):
            rows.append({'frame_index':f['frame_index'],'physics_tick':f['physics_tick'],'physical_time_s':f['elapsed_physics_s'],'env':i,'phase':s['phase'],'online_interactions':f['actual_env_control_steps'],'updates_completed':f['optimizer_updates_completed'],'parameter_sha256':f['parameter_sha256'],**{f'object_{k}_m':v for k,v in zip('xyz',s['object_position'])},'object_speed_m_s':float(np.linalg.norm(s['object_linear_velocity']))})
    csvpath=tables/'ACTUAL_TRAINING_CAMERA_ENV_STATES.csv';csvwrite(csvpath,rows);outputs.append(csvpath)
    eventcsv=tables/'ACTUAL_OPTIMIZER_UPDATES.csv';csvwrite(eventcsv,ev);outputs.append(eventcsv)
    sidecarcsv=tables/'ACTUAL_TRAINING_120HZ_SAMPLED.csv';outrows=[]
    for tick in range(0,1536,12):
        for i in range(16):outrows.append({'env':i,**{k:float(v) for k,v in zip(columns,states[tick,i])}})
    csvwrite(sidecarcsv,outrows);outputs.append(sidecarcsv)
    metadata={'scope':'EXTRA_REAL_ONLINE_TRAINING_VISUAL_RERUN_NOT_FORMAL_METHOD_PERFORMANCE','run':str(a.run.resolve()),'online_interactions':12288,'rollout_updates':3,'ppo_optimizer_minibatch_updates':m['optimizer_updates'],'initial_parameter_sha256':events['initial_parameter_sha256'],'final_parameter_sha256':last,'source_120hz_training_shape':list(states.shape),'source_120hz_sampling':'PRE_PHYSICS_INTEGRATION_AFTER_REFERENCE_UPDATE; no interpolation','training_physics_ticks':1536,'num_envs':16,'same_tick_montage_is_not_single_camera':True,'montage_physics_tick':selected['physics_tick'],'extra_physics_steps_for_camera':0,'state_writes_during_actions':0,'formal_performance_data':False,'camera_parameter_hash_chain_verified':True,'optimizer_preserves_physical_state_verified':True,'appearance_identity_verified':True,'script_sha256':sha(Path(__file__)),'inputs':{p.name:sha(p) for p in [a.run/'manifest.json',a.run/'run_config.json',a.run/'LIVE_OPTIMIZER_EVENTS.json',a.run/'LIVE_TRAINING_MONTAGES.json',a.run/'MATRIX_LIVE_TRAINING_FRAMES.json',a.run/'TRAINING_120HZ_STATE.npz',a.run/'VIDEO_CAPTURE_PROVENANCE.json',a.run/'VIDEO_SOURCE_PATCH.json',a.run/'policy.zip',source]},'outputs':{str(p.resolve()):{'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs},'limitations':['Self-built synthetic P2, not real robot or field transfer','Montage is16actual same-tick camera views in ONE physics app, not a single wide camera','Camera footage is an independent retraining rerun, not the formal model training recording','4posttraining episodes are train-check only, never added to method comparison','Initial failed4096-interaction recording attempt is retained and budgeted separately']}
    (a.output/'TRAINING_VIDEO_PROVENANCE.json').write_text(json.dumps(metadata,indent=2)+'\n');(a.release/'TRAINING_VIDEO_PROVENANCE.json').write_text(json.dumps(metadata,indent=2)+'\n')
    index=a.release/'index.html';text=index.read_text();text+='<section id="actual-training"><h2>R3: actual online PPO training rerun</h2><p>One 16-environment PhysX application;12288 extra interactions,3rollout updates, actual tensor-hash changes. Independent visual rerun, not formal performance. Montage:16actual views at same physics tick, not one camera.</p>'+f'<img style="max-width:100%" src="figures/{png.name}"><img style="max-width:100%" src="figures/{gif.name}"><p><a href="{os.path.relpath(outputs[0],a.release)}">LOCAL_ONLY actual training MP4</a></p><img style="max-width:100%" src="figures/{epng.name}"></section>';index.write_text(text)
    print(json.dumps({'video':str(outputs[0]),'montage':str(png),'optimizer_updates':3,'interactions':12288,'raw_120hz_shape':list(states.shape)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--release',type=Path,required=True);main(p.parse_args())
