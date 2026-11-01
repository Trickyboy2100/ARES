"""CPU actual-pixel magnification; immutable RAW and no physics."""
import argparse,gzip,json,shutil,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from program_pyramid_video import annotate
from program_charger_video_sync import font,pipe,close_pipe,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('package',type=Path);a=ap.parse_args();p=a.package;d=json.loads((p/'DATA.json').read_text());m=json.loads((p/'MANIFEST.json').read_text());ts=[json.load(gzip.open(p/n,'rt')) for n in ['LEFT_SOURCE.json.gz','RIGHT_SOURCE.json.gz']];K=np.array(m['camera_intrinsic']);E=np.array(m['camera_extrinsic']);raw_sha=sha(p/'RAW.mp4');crops=[]
    rd=subprocess.Popen(['/usr/bin/ffmpeg','-v','error','-i',str(p/'RAW.mp4'),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE);wr=pipe(p/'ANNOTATED.mp4',1600,1060,20)
    for k in range(251):
        buf=rd.stdout.read(1600*600*3);assert len(buf)==1600*600*3;raw=Image.fromarray(np.frombuffer(buf,np.uint8).reshape(600,1600,3));ann=annotate(raw,k,250,d['seed'],d['series']);dr=ImageDraw.Draw(ann);pair=[]
        if 40<=k<=90:
            for i,t in enumerate(ts):
                st=t['transitions'][k-1]['state'];ab=np.array([st['actors'][n][0][:3] for n in ['cubeA','cubeB']]).mean(0);cv=E@np.r_[ab,1];uv=K@cv[:3];u,v=uv[:2]/uv[2];x=int(np.clip(round(u)-90,0,620));y=int(np.clip(round(v)-100,0,450));box=(x+i*800,y,x+i*800+180,y+150);crop=raw.crop(box);crops.append({'frame':k,'branch':i,'RAW_pixel_box':box,'inset_magnification':2});pair.append(crop)
                xx=i*800+428;yy=354;ann.paste(crop.resize((360,300),Image.Resampling.NEAREST),(xx,yy));dr.rectangle((xx-1,yy-1,xx+360,yy+300),outline=['#f4ae59','#52cddd'][i],width=2);dr.rectangle((xx,yy,xx+359,yy+22),fill='#132334');dr.text((xx+4,yy+2),'2x RAW pixels | A red / B green',font=font(14),fill='white');dr.rectangle((box[0],box[1]+80,box[2],box[3]+80),outline=['#f4ae59','#52cddd'][i],width=1)
            if k==68:
                zoom=Image.new('RGB',(1080,560),'#132334');zd=ImageDraw.Draw(zoom);zd.text((12,8),'Same physical step 68 / 3.40 s | both A_OPEN | 3x actual RAW pixels',font=font(21),fill='white');zd.text((12,36),'SHARED_CONTROLLER_REPAIR_NOT_RL_GAIN | A red / B green / C blue',font=font(19),fill='white')
                for i,crop in enumerate(pair):zoom.paste(crop.resize((540,450),Image.Resampling.NEAREST),(i*540,90));zd.text((12+i*540,66),d['series'][i]['label'],font=font(18),fill=['#f4ae59','#52cddd'][i])
                zd.text((12,542),'Pixel enlargement only; visible finger orientation is recorded geometry. No inferred contact overlays.',font=font(15),fill='white');zoom.save(p/'CONTACT_CLOSEUP.png')
        wr.stdin.write(np.asarray(ann).tobytes())
        if k in [0,68,100,150,250]:ann.save(p/f'ANNOTATED_{k:03d}.png')
    assert rd.stdout.read()==b'';assert rd.wait()==0;close_pipe(wr);assert sha(p/'RAW.mp4')==raw_sha
    (p/'CONTACT_CROPS.json').write_text(json.dumps(crops,indent=2)+'\n');shutil.copy2(__file__,p/'contact_zoom_source.py');html=(p/'SYNC_VIEWER.html').read_text();html=html.replace('</main>','<details><summary>Release-contact detail: same physical step 68, both A_OPEN</summary><a href="CONTACT_CLOSEUP.png"><img src="CONTACT_CLOSEUP.png" style="width:100%;max-width:1080px"></a><p>3x original RAW pixels, nearest-neighbor magnification; no generated scene content. The annotated full video shows a 2x synchronized inset during steps 40–90. RAW remains unchanged.</p></details></main>');(p/'SYNC_VIEWER.html').write_text(html);m['contact_zoom']={'physics_steps_added':0,'RAW_sha256_unchanged':raw_sha,'frames':list(range(40,91)),'geometry':'recorded A/B mean projected with stored camera K/E; box 180x150, source pixel coordinates in CONTACT_CROPS.json','video_magnification':2,'standalone_magnification':3,'standalone_frame':68,'resampling':'NEAREST; no generated or sharpened pixels','source_sha256':sha(Path(__file__))};m['outputs']={x.name:sha(x) for x in p.iterdir() if x.is_file() and x.name!='MANIFEST.json'};(p/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
