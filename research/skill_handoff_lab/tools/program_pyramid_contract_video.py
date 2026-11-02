"""Reuse verified full-state viewer for actual contract interventions, zero steps."""
import numpy as np
from transforms3d.quaternions import quat2mat
import program_pyramid_video as video
original=video.series
original_viewer=video.viewer

def series(t,label):
    out=original(t,label)
    target=t['summary']['yaw']
    states=[t['initial'],*[x['state'] for x in t['transitions']]]
    errors=[]
    for s in states:
        R=quat2mat(s['actors']['cubeC'][0][3:7]);yaw=np.degrees(np.arctan2(R[1,0],R[0,0]));errors.append(float(abs((yaw-target+45)%90-45)))
    out['signals']['Actual top yaw error modulo90 (deg)']=errors
    out['contract_success']=t['summary']['contract_success']
    for i,d in enumerate(out['decisions']):d['skill']=['PLACE_SUPPORT','PLACE_TOP'][i]
    return out
def viewer(out,data):
    original_viewer(out,data)
    path=out/'SYNC_VIEWER.html';html=path.read_text().replace('height="600"','height="720"')
    html=html.replace('<main><h2>', '<main><p>Adaptation contract: native success AND top yaw within 10 degrees (modulo90). Native and contract results are separate in DATA.json. No contact forces recorded.</p><h2>')
    path.write_text(html)

if __name__=='__main__':
    video.CLAIM='CONTRACT_INTERVENTION_NO_NEW_RL_CLAIM'
    video.series=series
    video.viewer=viewer
    video.main()
    import sys,json,shutil,hashlib
    from pathlib import Path
    out=Path(sys.argv[sys.argv.index('--out')+1]);src=Path(__file__)
    shutil.copy2(src,out/'contract_render_source.py')
    p=out/'MANIFEST.json';m=json.loads(p.read_text());m['contract_renderer_sha256']=hashlib.sha256(src.read_bytes()).hexdigest();m['outputs']['SYNC_VIEWER.html']=hashlib.sha256((out/'SYNC_VIEWER.html').read_bytes()).hexdigest();p.write_text(json.dumps(m,indent=2))
