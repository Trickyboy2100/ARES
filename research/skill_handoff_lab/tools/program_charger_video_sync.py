#!/usr/bin/env python3
"""Source-linked paired pose-replay videos and time-synchronized science viewer.

This renders recorded PhysX states; it NEVER calls env.step(). Videos are NOT
new physical rollouts. Full-task results are taken only from source traces.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from program_charger_mechanism import arrays, load, sha


def font(size):
    from PIL import ImageFont
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def pipe(path, width, height, fps):
    cmd = ["/usr/bin/ffmpeg", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-crf", "20",
           "-pix_fmt", "yuv420p", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def close_pipe(process):
    assert process.stdin is not None
    process.stdin.close()
    assert process.wait() == 0, "ffmpeg failed"


def trace_series(trace):
    a = arrays(trace)
    for key in ("x", "lateral", "angle", "force"):
        v = np.asarray(a[key], dtype=float)
        assert len(v) == 200 and np.isfinite(v).all(), key
    return {k: np.asarray(a[k], dtype=float) for k in ("x", "lateral", "angle", "force", "mode")}


def plot_panel(draw, y, name, key, unit, series, colors, step):
    from PIL import ImageFont
    x0, x1, h = 115, 1870, 49
    vals = np.concatenate([s[key] for s in series])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if abs(hi-lo) < 0.001:
        hi, lo = hi+0.5, lo-0.5
    else:
        pad = (hi-lo)*0.08
        hi, lo = hi+pad, lo-pad
    draw.line([(x0, y+h), (x1, y+h)], fill="#475569", width=1)
    for i, src in enumerate(series):
        xy = [(int(x0 + j/199*(x1-x0)),
               int(y+h - (float(v)-lo)/(hi-lo)*h))
              for j, v in enumerate(src[key])]
        draw.line(xy, fill=colors[i], width=3, joint="curve")
    pos = int(x0 + max(0, min(step-1, 199))/199*(x1-x0))
    draw.line([(pos, y-3), (pos, y+h+2)], fill="#f8fafc", width=2)
    f = font(16)
    draw.text((10, y+6), name, fill="#e2e8f0", font=f)
    draw.text((x1-110, y+4), f"{lo:.2f}..{hi:.2f} {unit}", fill="#cbd5e1", font=f)


def projected_origin(camera, z, key):
    K = camera.get_intrinsic_matrix()[0].cpu().numpy()
    E = camera.get_extrinsic_matrix()[0].cpu().numpy()
    coord = E @ np.r_[np.asarray(z[key][:3], dtype=float), 1.]
    assert coord[2] > 0
    uv = K @ coord[:3]
    return (float(uv[0]/uv[2]), float(uv[1]/uv[2]))


def build_viewer(out, payload, fps):
    raw = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>ARES | PhysX evidence synchronized viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
body{background:#101b29;color:#f1f5f9;font:15px system-ui;margin:0;padding:20px}
main{max-width:1450px;margin:auto}h1{font-size:22px}p{color:#cbd5e1}
video{width:100%;max-height:76vh;background:#000;object-fit:contain}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}
button,select{background:#24394b;color:#fff;padding:9px 14px;border:1px solid #54718c;border-radius:6px;cursor:pointer}
canvas{width:100%;background:#182838;border:1px solid #415466;cursor:crosshair}
small{color:#94a3b8}a{color:#79d4f7}
</style></head><body><main><h1>ARES | Time-aligned physical execution evidence</h1>
<p>Both videos are <b>MEASURED_POSE_REPLAY</b> of recorded physics; no new physics simulation.
The vertical cursor on measured curves corresponds to the video's physical control step.
Orange and cyan rings mark <b>measured actor origins, not plug-tip/socket-opening contact points</b>.</p>
<div class="controls"><label>Video <select id="choice"><option value="annotated">Annotated + trace MP4</option>
<option value="raw">Unannotated RAW MP4</option></select></label>
<button id="back">Previous event</button><button id="forward">Next event</button>
<button id="play">Play / Pause</button><b id="step">Step 0</b></div>
<video id="movie" controls preload="metadata"></video>
<h3>Recorded signals (click any point to seek video)</h3>
<canvas id="plot" width="1400" height="390"></canvas>
<div id="events"></div><p><small>Panels: signed axial position (mm), lateral error (mm),
contact-force proxy (N), and event markers. Raw source SHA and render receipt are in MANIFEST.json.
The displayed servo mode refers to the command selected using the preceding observation.
All old and improved controller results are diagnostic; NO NEW RL CLAIM.</small></p>
<p><a href="RAW.mp4">Original MP4</a> · <a href="ANNOTATED.mp4">Annotated MP4</a>
· <a href="DATA.json">Measured time series</a> · <a href="MANIFEST.json">Source and file hashes</a></p>
</main><script>
const D=__BUNDLE__;
const video=document.querySelector('#movie'), choice=document.querySelector('#choice'),
 canvas=document.querySelector('#plot'),ctx=canvas.getContext('2d'),stepEl=document.querySelector('#step');
const fps=D.fps;
function swap(){const t=video.currentTime||0,wasPlaying=!video.paused;video.src=choice.value==='raw'?'RAW.mp4':'ANNOTATED.mp4';video.currentTime=t;if(wasPlaying)video.play().catch(()=>{});}
choice.addEventListener('change',swap);swap();
function currentStep(){return Math.max(0,Math.min(200,Math.round(video.currentTime*fps)));}
function draw(){
const w=canvas.width, x0=92,x1=w-40,top=30,ph=94;
ctx.fillStyle='#182838';ctx.fillRect(0,0,w,canvas.height);
const keys=[['axial_mm','Axial (mm)'],['lateral_mm','Lateral (mm)'],['force_N','Contact proxy (N)']];
const cols=['#f2994a','#4fd1e7'];
for(let p=0;p<keys.length;p++){const key=keys[p][0],y=top+p*ph;
 const all=D.series.flatMap(s=>s[key]),lo0=Math.min(...all),hi0=Math.max(...all),
 lo=lo0-(hi0-lo0||1)*.08,hi=hi0+(hi0-lo0||1)*.08;
 ctx.strokeStyle='#3b5264';ctx.beginPath();ctx.moveTo(x0,y+ph-18);ctx.lineTo(x1,y+ph-18);ctx.stroke();
 ctx.fillStyle='#d7e5ed';ctx.font='15px system-ui';ctx.fillText(keys[p][1],8,y+20);
 D.series.forEach((s,k)=>{ctx.beginPath();ctx.strokeStyle=cols[k];ctx.lineWidth=2;
  s[key].forEach((v,j)=>{const x=x0+(j+1)/200*(x1-x0),yy=y+ph-18-(v-lo)/(hi-lo)*(ph-25);
   if(j===0)ctx.moveTo(x,yy);else ctx.lineTo(x,yy);});ctx.stroke();});
 const cursor=x0+currentStep()/200*(x1-x0);
 ctx.strokeStyle='#f8fafc';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(cursor,y);ctx.lineTo(cursor,y+ph-18);ctx.stroke();
}
ctx.fillStyle='#f2994a';ctx.fillText('Original rollback servo',x0,350);
ctx.fillStyle='#4fd1e7';ctx.fillText('Depth-preserving alignment',x0+300,350);
ctx.fillStyle='#fff';ctx.fillText('step '+currentStep()+'/200  |  '+(currentStep()/fps).toFixed(2)+' s',w-270,350);
stepEl.textContent='Physical step '+currentStep()+'/200';
}
canvas.onclick=e=>{const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*canvas.width,
 step=Math.round(Math.max(0,Math.min(1,(x-92)/(canvas.width-132)))*200);video.currentTime=step/fps;draw();};
document.querySelector('#play').onclick=()=>video.paused?video.play():video.pause();
const events=D.events;
function seekEvent(sign){const now=currentStep(),sorted=events.map(e=>e.step);
let n=sign>0?sorted.find(x=>x>now):sorted.filter(x=>x<now).at(-1);if(n===undefined)n=sign>0?200:0;video.currentTime=n/fps;draw();}
document.querySelector('#back').onclick=()=>seekEvent(-1);
document.querySelector('#forward').onclick=()=>seekEvent(1);
document.querySelector('#events').textContent='Key events: '+events.map(e=>'step '+e.step+' '+e.label).join(' · ');
video.addEventListener('timeupdate',draw);video.addEventListener('seeked',draw);
function animate(){if(!video.paused)draw();requestAnimationFrame(animate);}animate();draw();
</script></body></html>'''
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    (out/"SYNC_VIEWER.html").write_text(raw.replace("__BUNDLE__",embedded), encoding="utf-8")


def main():
    from PIL import Image, ImageDraw
    import torch
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.utils import sapien_utils

    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True,help="EXP-001 source runs directory")
    ap.add_argument("--out",type=Path,required=True,help="New output directory; never overwrite")
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--fps",type=int,default=20)
    args=ap.parse_args()
    assert args.fps==20, "native PlugCharger uses 20Hz; preserve step/time alignment"
    args.out.mkdir(parents=True,exist_ok=False)

    variants=("servo_stop","servo_local_hold_stop")
    sources=[load(args.root,k,args.seed) for k in variants]
    assert len({t["summary"]["initial_sha256"] for _,t in sources})==1
    assert all(len(t["steps"])==200 for _,t in sources)
    series=[trace_series(t) for _,t in sources]
    mid=np.asarray(sources[0][1]["steps"][70]["cube"][:3],dtype=float)
    pose=sapien_utils.look_at(mid+np.array([-.15,.29,.23]),mid+np.array([-.02,0,0]))
    env=gym.make("PlugCharger-v1",num_envs=1,obs_mode="state",
        control_mode="pd_joint_pos",render_mode="rgb_array",
        sim_backend="physx_cuda",render_backend="gpu",
        human_render_camera_configs=dict(mount=None,pose=pose,width=960,height=720,fov=.64))
    env.reset(seed=args.seed)
    camera=next(iter(env.unwrapped._human_render_cameras.values())).camera
    raw_mp4=pipe(args.out/"RAW.mp4",1920,720,args.fps)
    annotated_mp4=pipe(args.out/"ANNOTATED.mp4",1920,1080,args.fps)
    max_readback=0.
    colors=("#f2994a","#4fd1e7")
    heading=font(25);sub=font(19)
    try:
        for step in range(201):
            raw=Image.new("RGB",(1920,720))
            measurements=[]
            for i,(path,t) in enumerate(sources):
                z=t["initial_measure"] if step==0 else t["steps"][step-1]
                state=copy.deepcopy(t["initial"])
                state["actors"]["charger"][0][:7]=z["tool"]
                state["actors"]["receptacle"][0][:7]=z["cube"]
                state["articulations"][next(iter(state["articulations"]))][0][13:22]=z["q"]
                tensors={grp:{k:torch.tensor(v,dtype=torch.float32,device="cuda")
                             for k,v in vv.items()} for grp,vv in state.items()}
                env.unwrapped.set_state_dict(tensors)
                observed=env.unwrapped.get_state_dict()
                max_readback=max(max_readback,max(float((observed[g][k]-tensors[g][k]).abs().max())
                      for g in tensors for k in tensors[g]))
                rgb=env.unwrapped.render_rgb_array()[0].cpu().numpy()
                raw.paste(Image.fromarray(np.asarray(rgb,dtype=np.uint8)),(960*i,0))
                measurements.append((z,[(name,projected_origin(camera,z,name))
                                        for name in ("tool","cube")]))
            assert raw_mp4.stdin and annotated_mp4.stdin
            raw_mp4.stdin.write(np.asarray(raw).tobytes())
            annotated=Image.new("RGB",(1920,1080),"#102030")
            annotated.paste(raw,(0,70))
            draw=ImageDraw.Draw(annotated)
            draw.text((16,7),f"CASE {args.seed} | PHYSICS STEP {step:03d}/200 | MEASURED POSE REPLAY | NO NEW RL CLAIM",
                      font=heading,fill="#f7fafc")
            for i,(z,origins) in enumerate(measurements):
                left=960*i
                draw.text((left+20,38),("Original servo: rollback","Depth-preserving servo")[i],
                          font=sub,fill=colors[i])
                for key,(u,v) in origins:
                    x,y=u+left,v+70
                    if 0<=u<960 and 0<=v<720:
                        col="#f4b34b" if key=="tool" else "#4fd1e7"
                        draw.ellipse((x-10,y-10,x+10,y+10),outline=col,width=3)
                a=series[i];idx=max(0,step-1)
                draw.rectangle((left,746,left+960,790),fill="#102030")
                txt=(f"Axial {a['x'][idx]:.2f} mm  |  Lateral {a['lateral'][idx]:.2f} mm"
                     f"  |  Force proxy {a['force'][idx]:.2f} N  |  final {int(sources[i][1]['summary']['success'])}")
                draw.text((left+12,756),txt,font=sub,fill="#f3f4f6")
            for panel,(name,key,unit) in enumerate([
                 ("Axial position","x","mm"),
                 ("Lateral error","lateral","mm"),
                 ("Contact proxy","force","N")]):
                plot_panel(draw,800+panel*86,name,key,unit,series,colors,step)
            annotated_mp4.stdin.write(np.asarray(annotated).tobytes())
        assert max_readback<1e-5, "pose replay mismatch"
    finally:
        close_pipe(raw_mp4)
        close_pipe(annotated_mp4)
        env.close()

    dat=[]
    for i,(p,t) in enumerate(sources):
        a=series[i];dat.append({
          "policy":variants[i],"source_sha256":sha(p),
          "source":str(p),"terminal_success":bool(t["summary"]["success"]),
          "axial_mm":a["x"].tolist(),"lateral_mm":a["lateral"].tolist(),
          "angle_rad":a["angle"].tolist(),"force_N":a["force"].tolist(),
          "mode":[None if not np.isfinite(x) else int(x) for x in a["mode"]]
        })
    events=[]
    old=series[0]
    for j in range(1,200):
        if old["x"][j]-old["x"][j-1]<-0.5:
            events.append({"step":j+1,"label":"old-servo rollback >0.5mm"})
        if old["force"][j]>0.1 and old["force"][j-1]<=0.1:
            events.append({"step":j+1,"label":"old-servo contact onset (proxy)"})
        if series[1]["force"][j]>0.1 and series[1]["force"][j-1]<=0.1:
            events.append({"step":j+1,"label":"depth-hold contact onset (proxy)"})
    events.sort(key=lambda x:x["step"])
    payload={"seed":args.seed,"fps":args.fps,"horizon":200,"series":dat,"events":events}
    (args.out/"DATA.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    build_viewer(args.out,payload,args.fps)
    receipt={"seed":args.seed,"physics_steps":0,"source_kind":"MEASURED_POSE_REPLAY",
         "camera":"fixed world camera; measured actor origins projected, NOT contact points",
         "step_to_time":"frame k represents physical step k, 20 fps; frame 0 is initial",
         "max_pose_readback_error":max_readback,
         "source":[{"path":str(p),"sha256":sha(p)} for p,t in sources],
         "source_script_sha256":sha(Path(__file__)),
         "outputs":{name:sha(args.out/name) for name in
            ["RAW.mp4","ANNOTATED.mp4","DATA.json","SYNC_VIEWER.html"]}}
    (args.out/"MANIFEST.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"out":str(args.out),"source_kind":"MEASURED_POSE_REPLAY",
                      "physics_steps":0,"frames":201,"max_readback_error":max_readback},indent=2))


if __name__=="__main__":
    main()
