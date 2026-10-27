"""Presentation-only refresh of actual recorded frames, never simulator execution.

Replaces filmstrip captions only; refreshes precisely those artifact hashes and
retains original composition provenance. MP4/GIF/physics/data are not changed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    module_path=Path(__file__).with_name('build_paired_physx_delivery.py')
    spec=importlib.util.spec_from_file_location('paired_composer',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    global_path=a.release/'VIDEO_PROVENANCE.json';global_provenance=json.loads(global_path.read_text())
    refresh=[]
    for record in global_provenance['comparisons']:
        if record['iteration']!=a.iteration:continue
        out=Path(record['mp4_local_path']).parent
        runs=[Path(record['left_run']),Path(record['right_run'])]
        case=record['case_id'];meta=[module.load(run/'VIDEO_CAPTURE_PROVENANCE.json')['cases'][str(case)] for run in runs]
        frames=[module.load(run/m['frame_records']) for run,m in zip(runs,meta)]
        times=[np.array([r['elapsed_physics_s'] for r in f]) for f in frames];ends=[float(t[-1]) for t in times]
        contact=[r['elapsed_physics_s'] for f in frames for r in f if r['max_robot_net_contact_N']>.02]
        handoffs=[r['elapsed_physics_s'] for f in frames for r in f if r['phase']==1]
        events=sorted(set([0.,min(contact,default=max(ends)*.2),min(handoffs,default=max(ends)*.45),max(ends)*.75,max(ends)]))
        strip=Image.new('RGB',(len(events)*384,600),'white');draw=ImageDraw.Draw(strip)
        draw.text((15,10),f'{a.iteration} / case {case}: actual time-aligned closed-loop frames',fill='#203544',font=module.font(24,True))
        subtitle='Historical tuned feedback vs ordinary residual PPO (median training seed41)' if a.iteration=='R0' else 'Paired independent cold starts; exact original clocks and qualified outcomes'
        draw.text((15,42),subtitle,fill='#526676',font=module.font(18))
        labels=[a.left_label,a.right_label]
        for side,(run,m,t,end,label) in enumerate(zip(runs,meta,times,ends,labels)):
            reader=module.Reader(run/m['mp4'])
            for k,tt in enumerate(events):
                j=max(0,int(np.searchsorted(t,tt,side='right')-1));image=reader.get(j).resize((384,216))
                y=108+side*245;strip.paste(image,(k*384,y))
                status='END/HOLD' if tt>end+1e-6 else ('END' if tt>=end-1e-6 else f'Stage {frames[side][j]["phase"]}')
                caption=f'{label} | {min(tt,end):.2f}s | {status}'
                draw.text((k*384+7,y-23),caption,fill='#203544',font=module.font(16,True))
            reader.close()
        draw.text((15,578),'Frames after an earlier termination are explicitly held; no new physics or trajectory playback. SYNTHETIC ARES tasks.',fill='#526676',font=module.font(15))
        before={ext:sha(out/f'filmstrip.{ext}') for ext in ['png','pdf']}
        strip.save(out/'filmstrip.png');fig=plt.figure(figsize=(12.8,4));ax=fig.add_axes((0,0,1,1));ax.imshow(strip);ax.axis('off');fig.savefig(out/'filmstrip.pdf',dpi=150);plt.close(fig)
        for ext in ['png','pdf']:
            p=out/f'filmstrip.{ext}';record['files'][p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
            (a.release/'figures'/f'{a.iteration}_case{case:03d}_{p.name}').write_bytes(p.read_bytes())
        entry={'case_id':case,'old_hashes':before,'new_hashes':{ext:sha(out/f'filmstrip.{ext}') for ext in ['png','pdf']},'scope':'caption/layout only; exact source frames; MP4/GIF/trace unchanged','script_sha256':sha(Path(__file__))}
        record['filmstrip_layout_refresh']=entry
        local=module.load(out/'PAIR_PROVENANCE.json');local['files'].update({f'filmstrip.{ext}':record['files'][f'filmstrip.{ext}'] for ext in ['png','pdf']});local['filmstrip_layout_refresh']=entry
        (out/'PAIR_PROVENANCE.json').write_text(json.dumps(local,indent=2));refresh.append(entry)
    global_path.write_text(json.dumps(global_provenance,indent=2))
    (a.release/f'{a.iteration}_FILMSTRIP_LAYOUT_REFRESH.json').write_text(json.dumps({'refreshes':refresh,'no_physics_or_data_modified':True},indent=2))
    print(json.dumps({'filmstrips_refreshed':len(refresh)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--iteration',required=True);p.add_argument('--left-label',required=True);p.add_argument('--right-label',required=True);main(p.parse_args())
