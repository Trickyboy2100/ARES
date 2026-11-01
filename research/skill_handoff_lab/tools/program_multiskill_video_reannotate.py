"""Refresh trace-derived overlays from existing RAW frames, no GPU or physics."""
import argparse,gzip,json,shutil,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from program_multiskill_video import annotation,series,viewer
from program_charger_video_sync import pipe,close_pipe,sha
import program_multiskill_video

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists();shutil.copytree(a.source,a.out)
    data=json.loads((a.out/'DATA.json').read_text());ts=[json.load(gzip.open(a.out/n,'rt')) for n in ['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz']];labels=[s['label'] for s in data['series']];ss=[series(t,l) for t,l in zip(ts,labels)];data['series']=ss;h=data['horizon']
    reader=subprocess.Popen(['/usr/bin/ffmpeg','-v','error','-i',str(a.out/'RAW.mp4'),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE);writer=pipe(a.out/'ANNOTATED.mp4',1600,1060,20)
    for k in range(h+1):
        buf=reader.stdout.read(1600*600*3);assert len(buf)==1600*600*3;raw=Image.fromarray(np.frombuffer(buf,dtype=np.uint8).reshape(600,1600,3));ann=annotation(raw,k,h,data['seed'],ss,labels);writer.stdin.write(np.asarray(ann).tobytes())
        if k in [0,h//2,h]:raw.save(a.out/f'RAW_{k:03d}.png');ann.save(a.out/f'ANNOTATED_{k:03d}.png')
    assert reader.stdout.read()==b'';assert reader.wait()==0;close_pipe(writer);(a.out/'DATA.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');viewer(a.out,data)
    source=Path(program_multiskill_video.__file__);shutil.copy2(source,a.out/'annotation_source.py');m=json.loads((a.out/'MANIFEST.json').read_text());m['annotation_source_sha256']=sha(source);m['raw_video_unchanged_sha256']=sha(a.out/'RAW.mp4');m['prior_package']={'path':str(a.source.resolve()),'manifest_sha256':sha(a.source/'MANIFEST.json')};m['annotation_revision']='Excluded pre-first-step unvalidated contact buffer readback; raw source value retained in DATA and gzip';m['outputs']={p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.name!='MANIFEST.json'};(a.out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
