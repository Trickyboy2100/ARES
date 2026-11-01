"""CPU-only add matched-repeat100Hz contact peaks; preserve original20Hz source."""
import argparse,gzip,json,shutil,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from program_multiskill_video import annotation,viewer
from program_charger_video_sync import font,pipe,close_pipe,sha,execution_manifest

KEY='Matched 100Hz peak (N)'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--left-probe',type=Path,required=True);ap.add_argument('--right-probe',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists();shutil.copytree(a.source,a.out);data=json.loads((a.out/'DATA.json').read_text());ss=data['series'];audits=[]
    for i,(source,probe) in enumerate(zip(['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz'],[a.left_probe,a.right_probe])):
        t=json.load(gzip.open(a.out/source,'rt'));p=json.load(gzip.open(probe,'rt'));assert len(p['steps'])==len(t['steps'])==100;assert t['summary']['initial_sha256']==p['summary']['initial_sha256'];diff={}
        for key in ['action','q','qvel','tcp','tool','cube','tcp_T_tool']:
            diff[key]=max(float(np.max(np.abs(np.asarray(x[key])-np.asarray(y[key])))) for x,y in zip(t['steps'],p['steps']));assert diff[key]==0,(key,diff)
        peaks=[None];robots=[None];first=None
        for k,z in enumerate(p['steps'],1):
            c=z['physics_substep_contacts'];assert len(c)==5;assert [v['substep'] for v in c]==[1,2,3,4,5];assert all(v['control_step']==k for v in c);peaks.append(max(v['force_norms_N']['held_tool'] for v in c));robots.append(max(vv for v in c for n,vv in v['force_norms_N'].items() if n!='held_tool'))
            if first is None:
                for v in c:
                    if v['force_norms_N']['held_tool']>.5:first={'control_step':k,'substep':v['substep'],'time_s':(k-1)/20+v['substep']/100,'phase':z['phase'],'force_N':v['force_norms_N']['held_tool']};break
        name=['LEFT_CONTACT_REPEAT.json.gz','RIGHT_CONTACT_REPEAT.json.gz'][i];shutil.copy2(probe,a.out/name);audit={'original_source_sha256':sha(a.out/source),'probe_source':str(probe.resolve()),'probe_sha256':sha(probe),'portable_copy':name,'same_initial_hash':True,'max_abs_recorded_state_action_differences':diff,'control_steps':100,'physics_substeps':500,'all_five_substeps_per_control_step':True,'first_tool_contact_over_half_N':first,'tool_max_N':max(peaks[1:]),'all_robot_link_cube_force_max_N':max(robots[1:]),'source_manifest':execution_manifest(probe)};audits.append(audit);ss[i]['signals'][KEY]=peaks;ss[i]['matched_repeat_robot_peak_N']=robots;ss[i]['contact_secondary_source']=audit
    data['contact_peak_semantics']='Separate physical repeat, all100recorded action/state samples exactly equal. Each20Hz frame k shows maximum of five100Hz substeps in ((k-1)/20,k/20]. Not instantaneous force and not original episode substep sampling.';(a.out/'DATA.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');viewer(a.out,data);html=(a.out/'SYNC_VIEWER.html').read_text().replace('height="600"','height="720"').replace('<p>MEASURED_POSE_REPLAY','<p>Matched100Hz peak is a secondary exactly matched physical repeat, binned over five substeps; the original20Hz samples remain separate.<p>MEASURED_POSE_REPLAY');(a.out/'SYNC_VIEWER.html').write_text(html)
    read=subprocess.Popen(['/usr/bin/ffmpeg','-v','error','-i',str(a.out/'RAW.mp4'),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE);write=pipe(a.out/'ANNOTATED.mp4',1600,1060,20)
    for k in range(101):
        buf=read.stdout.read(1600*600*3);assert len(buf)==1600*600*3;raw=Image.fromarray(np.frombuffer(buf,dtype=np.uint8).reshape(600,1600,3));im=annotation(raw,k,100,data['seed'],ss,[s['label'] for s in ss]);d=ImageDraw.Draw(im);d.rectangle((0,967,1599,1059),fill='#132334');d.text((12,970),'Tool/cube force: thin=original20Hz | thick=matched100Hz within-frame peak',font=font(15),fill='white');hi=max(v for s in ss for v in s['signals'][KEY] if v is not None)*1.05
        for s,col in zip(ss,['#f4ae59','#52cddd']):
            for key,width in [('Tool/cube contact proxy (N)',1),(KEY,3)]:d.line([(250+j/100*1320,1050-v/hi*55) for j,v in enumerate(s['signals'][key]) if v is not None],fill=col,width=width)
        d.line((250+k/100*1320,990,250+k/100*1320,1055),fill='white',width=2);write.stdin.write(np.asarray(im).tobytes())
        if k in [0,50,100]:im.save(a.out/f'ANNOTATED_{k:03d}.png')
    assert read.stdout.read()==b'';assert read.wait()==0;close_pipe(write);shutil.copy2(__file__,a.out/'contact_overlay_source.py');m=json.loads((a.out/'MANIFEST.json').read_text());m['secondary_contact_sources']=audits;m['contact_overlay_source_sha256']=sha(Path(__file__));m['contact_overlay_physics_steps_added']=0;m['secondary_probe_existing_physics_budget']={'control_steps':200,'substeps':1000,'note':'already counted in physical_design contact-probe experiment; not counted again as rendering'};m['original_raw_video_sha256']=sha(a.out/'RAW.mp4');assert m['original_raw_video_sha256']==sha(a.source/'RAW.mp4');m['prior_video_package']={'path':str(a.source.resolve()),'manifest_sha256':sha(a.source/'MANIFEST.json')};m['outputs']={p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.name!='MANIFEST.json'};(a.out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(audits,indent=2))
if __name__=='__main__':main()
