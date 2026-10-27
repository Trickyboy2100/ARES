"""Compose only observed Isaac pixels; no geometry generation or retouching.

Input frames are immutable 1920x1200 raw sensor PNGs from live evaluation.
Episode outcome annotations are retrospective, not terminal-frame claims.
"""
import argparse,hashlib,json,shutil,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def make(root,dest):
    v2=root/'physics_capture_v2';v3=root/'physics_capture_v3'
    for r in [v2,v3]:
        assert (r/'RUN_COMPLETE.json').exists()
        assert json.loads((r/'APPEARANCE_IDENTITY.json').read_text())['equal']
    records=json.loads((v2/'PUBLICATION_CAMERA_RECORDS.json').read_text())
    audits={x['case_id']:x for x in json.loads((v2/'INDEPENDENT_TASK_AND_CONTACT.json').read_text())['cases']}
    selected=[('Fig3a_success_episode_raw.png',v2/'publication_raw/contact_3_tick000800.png',3,True),
              ('Fig3b_timeout_episode_raw.png',v2/'publication_raw/contact_1_tick001184.png',1,False),
              ('Fig3c_16env_raw.png',v3/'publication_raw/array_tick000800.png',None,None)]
    evidence=[];dest.mkdir(parents=True,exist_ok=True)
    for name,p,case,passed in selected:
        shutil.copyfile(p,dest/name)
        r=v3 if case is None else v2;m=json.loads((r/'manifest.json').read_text())
        record=next(x for x in json.loads((r/'PUBLICATION_CAMERA_RECORDS.json').read_text()) if x['file']==str(p.relative_to(r)))
        if case is not None:assert record['case_id']==case and audits[case]['independent_task_pass']==passed
        evidence.append({'file':name,'sha256':digest(dest/name),'raw_source':str(p),'raw_sha256':digest(p),
            'source_task_sha256':m['task_source_sha256'],'asset_sha256':m['asset_sha256'],'checkpoint_sha256':m['loaded_policy_sha256'],
            'physics_steps':record['physics_steps'],'case_id':case,'independent_eventual_episode_success':passed,
            'pixel_size':list(Image.open(p).size),'capture_role':'LIVE_PHYSX_EVALUATION_NOT_TRAINING','terminal_frame':False,
            'appearance_identity':json.loads((r/'APPEARANCE_IDENTITY.json').read_text()),'camera_record':record})
    fig=plt.figure(figsize=(13,7.7),facecolor='white',layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=[1.3,1.],hspace=.1,wspace=.06)
    ax0=fig.add_subplot(gs[:,0]);ax1=fig.add_subplot(gs[0,1]);ax2=fig.add_subplot(gs[1,1])
    for ax,name,title in [(ax0,'Fig3c_16env_raw.png','A  |  One application, 16 independent PhysX environments'),
        (ax1,'Fig3a_success_episode_raw.png','B  |  P2 case 3: successful episode (nonterminal frame)'),
        (ax2,'Fig3b_timeout_episode_raw.png','C  |  P2 case 1: timeout episode (nonterminal frame)')]:
        ax.imshow(Image.open(dest/name));ax.axis('off');ax.set_title(title,loc='left',fontsize=10,fontweight='bold',color='#16324a',pad=8)
    fig.suptitle('ARES contact manipulation | Actual Isaac Sim 5.1 / Isaac Lab 2.3.0',fontsize=15,fontweight='bold',color='#16324a')
    fig.supxlabel('SYNTHETIC self-built v6b task; live policy re-evaluation, not B3 training. Free 40 mm cube; fixed research paddle; no contact grasp or field scene.',fontsize=8,color='#4b5b69')
    plt.rcParams['pdf.fonttype']=42;plt.rcParams['svg.fonttype']='none'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'Fig3_physx_scene_board.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    prov=dest/'provenance';prov.mkdir(exist_ok=True)
    (prov/'FIG3_PROVENANCE.json').write_text(json.dumps({'figure':'Fig3','scope':'SYNTHETIC / REPRODUCED_LIVE_PHYSX / STATE_BASED',
        'script_sha256':digest(Path(__file__)),'pixel_modifications':'None; raw frames copied byte-for-byte, only arranged with external vector labels',
        'limits':['Not a terminal-state photograph','Not field/CAD tray/grasping','Not online training footage','Not the new v7 pilot'],
        'images':evidence},indent=2))
    data=dest/'data';data.mkdir(exist_ok=True)
    with (data/'physx_camera_frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file','case_id','physics_steps','independent_eventual_episode_success','sha256']);w.writeheader()
        w.writerows({k:x[k] for k in w.fieldnames} for x in evidence)
    print(json.dumps({'raw_images':len(evidence),'board':str(dest/'Fig3_physx_scene_board.pdf'),'case3_qualified':True,'case1_qualified':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();make(a.root.resolve(),a.dest.resolve())
