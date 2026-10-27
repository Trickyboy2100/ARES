#!/usr/bin/env python3
"""Render evidence-linked Chinese visual brief; never generates experiment data."""
from pathlib import Path
import importlib.util
import re
import hashlib
import json
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Spacer, PageBreak, Table, TableStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('client_style', ROOT.parent/'build_client_documents.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)

def picture(name, caption, maxheight=228):
    path = ROOT/'figures'/name
    iw, ih = ImageReader(str(path)).getSize()
    scale = min((style.W-76)/iw, maxheight/ih)
    return [Image(str(path),width=iw*scale,height=ih*scale),Spacer(1,6),
            style.para(caption,'caption')]

def main():
    source = ROOT/'MENTOR_VISUAL_BRIEF_CN.md'
    sections=re.split(r'^## ',source.read_text(),flags=re.M)[1:]
    figures = [
        ('R0_case001_filmstrip.png','R0｜P2 同一 case001，反馈 FAIL / 历史普通残差 PASS。实际重执行，不是新算法增益。'),
        ('R1_case008_filmstrip.png','R1｜同一 case008 两方法共同失败。两端真实动态延续，不能以动作不同代替任务改善。'),
        ('R2_case005_filmstrip.png','R2｜同一 case005 共同失败。旋转包络与有限工具yaw为同权限共享工程层，不作为新算法收益。'),
        ('R3_case005_filmstrip.png','R3｜seed31同一 case005，普通PPO失败、执行维目标成功；全三seed不支持稳定增益，不是学会恢复的证据。'),
    ]
    story=[]
    for i,section in enumerate(sections):
        heading,body=section.split('\n',1)
        if i: story.append(PageBreak())
        story += [style.para('ACTUAL PHYSX  /  EVIDENCE FIRST','tag'), style.para(heading,'h1')]
        blocks=body.strip().split('\n\n')
        story.append(style.para(blocks[0],'lead'))
        story.extend(picture(*figures[i],maxheight=140 if i==3 else 228))
        if i==3:
            montage=ROOT/'figures/R3_online_training_16env_montage.png'
            iw,ih=ImageReader(str(montage)).getSize()
            image_flow=Image(str(montage),width=260,height=260*ih/iw)
            panel=Table([[style.para(blocks[1]),[image_flow,style.para('真实在线训练｜单应用16相机同tick，不是单相机全景。独立展示run，不替代正式模型。','caption')]]],colWidths=[245,274])
            panel.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),8)]))
            story += [panel,Spacer(1,8)]
            story.extend(style.para(b) for b in blocks[2:])
        else:
            story.extend(style.para(b) for b in blocks[1:])
    output=ROOT/'MENTOR_VISUAL_BRIEF_CN.pdf'
    style.document(output,story,'ARES 真实接触强化学习可视化简报')
    manifest={'version':'R0_R1_R2_R3_formal','pdf':output.name,'pages':len(PdfReader(output).pages),
        'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'inputs':[{'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in [source,*[ROOT/'figures'/f[0] for f in figures],ROOT/'figures/R3_online_training_16env_montage.png']],
        'status':'Real R0-R3 evidence; three-seed primary gain unsupported; single-seed fixed-refresh time signal retained'}
    (ROOT/'VISUAL_BRIEF_BUILD_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
