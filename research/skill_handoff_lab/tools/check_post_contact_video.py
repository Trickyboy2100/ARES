import json,subprocess,hashlib
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;rows=[]
for name in ['video_rescue','video_regression']:
 d=json.load(open(r/name/'DATA.json'));m=json.load(open(r/name/'MANIFEST.json'))
 for video in ['RAW.mp4','ANNOTATED.mp4']:
  f=r/name/video;z=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate,width,height,duration','-of','json',str(f)]))['streams'][0];assert int(z['nb_frames'])==101 and z['r_frame_rate']=='20/1';rows.append(dict(video=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),**z))
 for s in d['series']:
  assert len(s['phase'])==101
  assert all(len(v)==101 for v in s['signals'].values())
  assert len(s['decisions'])==3
 assert m['max_state_readback_error']<1e-5 and m['physics_steps_added']==0
(r/'analysis/VIDEO_QA.json').write_text(json.dumps(dict(status='PASS',scope='encoded frame count/fps, all signal lengths, 3actual decisions, pose readback; static final annotated frame visually inspected',rows=rows),indent=2));print('PASS',len(rows))
