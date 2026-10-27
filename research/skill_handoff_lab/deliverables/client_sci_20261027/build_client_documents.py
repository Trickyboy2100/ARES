#!/usr/bin/env python3
"""Build Chinese client/mentor PDFs from editable Markdown and real figure assets.

No field files are read. Rendering is presentation only; experiment data and
provenance are owned by the corresponding figure/run generators.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    Image, Table, TableStyle, KeepTogether,
)

ROOT = Path(__file__).resolve().parent
FONT = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
pdfmetrics.registerFont(TTFont('ClientCJK', str(FONT)))
pdfmetrics.registerFont(TTFont('ClientLatin', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('ClientLatinBold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
W, H = 595.276, 841.89
NAVY = colors.HexColor('#132E45')
TEAL = colors.HexColor('#007C83')
INK = colors.HexColor('#233443')
LIGHT = colors.HexColor('#EAF4F4')
GRAY = colors.HexColor('#627283')

ST = {
    'body': ParagraphStyle('body', fontName='ClientCJK', fontSize=10.1, leading=16.8,
                           textColor=INK, wordWrap='CJK', spaceAfter=9),
    'lead': ParagraphStyle('lead', fontName='ClientCJK', fontSize=11.2, leading=18.2,
                           textColor=NAVY, wordWrap='CJK', spaceAfter=12),
    'h1': ParagraphStyle('h1', fontName='ClientCJK', fontSize=22, leading=30,
                        textColor=NAVY, wordWrap='CJK', spaceAfter=16),
    'h2': ParagraphStyle('h2', fontName='ClientCJK', fontSize=13, leading=20,
                        textColor=TEAL, wordWrap='CJK', spaceBefore=12, spaceAfter=8),
    'caption': ParagraphStyle('caption', fontName='ClientCJK', fontSize=8.2, leading=12.3,
                             textColor=GRAY, wordWrap='CJK', spaceAfter=8),
    'tag': ParagraphStyle('tag', fontName='ClientLatinBold', fontSize=8.3, leading=12,
                         textColor=TEAL, spaceAfter=8),
    'table': ParagraphStyle('table', fontName='ClientCJK', fontSize=8.7, leading=13,
                           textColor=INK, wordWrap='CJK'),
}

def esc(t):
    # Droid CJK supplies Chinese glyphs but not Latin glyph outlines. Explicit
    # script runs avoid the silent missing-Latin failure of a single CJK font.
    cmap=pdfmetrics.getFont('ClientCJK').face.charToGlyph
    pieces=[]
    for ch in t:
        font='ClientCJK' if ord(ch)>127 and ord(ch) in cmap else 'ClientLatin'
        if pieces and pieces[-1][0]==font:
            pieces[-1]=(font,pieces[-1][1]+ch)
        else:
            pieces.append((font,ch))
    out=[]
    for font,piece in pieces:
        s=piece.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        out.append(('<font name="'+font+'">'+s+'</font>').replace('\n','<br/>'))
    return ''.join(out)

def para(t, style='body'):
    return Paragraph(esc(t), ST[style])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, H-20, W, 20, fill=1, stroke=0)
    canvas.setFont('ClientLatinBold', 8.3)
    canvas.setFillColor(colors.white)
    canvas.drawString(38, H-13, 'ARES / CONTACT SKILL RL')
    canvas.setFont('ClientLatin', 8)
    canvas.drawRightString(W-38, H-13, 'RESEARCH WORKING DRAFT  |  2026-10-27')
    canvas.setStrokeColor(colors.HexColor('#CADADA'))
    canvas.line(38, 36, W-38, 36)
    canvas.setFillColor(GRAY)
    canvas.setFont('ClientCJK', 7.5)
    canvas.drawString(38, 24, '自建合成任务　实际物理仿真　非现场或实机　未验证增益不作结论')
    canvas.setFont('ClientLatin', 8)
    canvas.drawRightString(W-38, 24, f'{doc.page}')
    canvas.restoreState()

def document(path, story, title):
    doc = BaseDocTemplate(str(path), pagesize=(W,H), leftMargin=38, rightMargin=38,
                          topMargin=42, bottomMargin=47, title=title,
                          author='ARES Research Working Group')
    frame = Frame(38, 47, W-76, H-91, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='client', frames=[frame], onPage=footer))
    doc.build(story)

def figure(filename, caption, maxheight=235):
    path = ROOT/'03_FIGURES'/filename
    if not path.exists():
        return []
    ir = ImageReader(str(path))
    iw, ih = ir.getSize()
    scale = min((W-76)/iw, maxheight/ih)
    return [Spacer(1,5), Image(str(path), width=iw*scale, height=ih*scale),
            Spacer(1,5), para(caption,'caption')]

def box(text):
    table=Table([[para(text,'lead')]], colWidths=[W-94])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),LIGHT),
        ('BOX',(0,0),(-1,-1),0.6,colors.HexColor('#AFD5D5')),
        ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
        ('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),2),
    ]))
    return [table,Spacer(1,12)]

def table(rows, widths):
    data=[[para(str(c),'table') for c in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),LIGHT),
        ('LINEABOVE',(0,0),(-1,0),1,TEAL),
        ('LINEBELOW',(0,0),(-1,0),.6,TEAL),
        ('LINEBELOW',(0,-1),(-1,-1),.7,TEAL),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    return [t,Spacer(1,10)]

def md_sections(name):
    txt=(ROOT/name).read_text()
    chunks=re.split(r'^## ',txt,flags=re.M)[1:]
    return [(c.split('\n',1)[0],c.split('\n',1)[1].strip().split('\n\n')) for c in chunks]

def build_proposal():
    sections=md_sections('01_CLIENT_PROPOSAL_CN.md')
    story=[]
    figures={
        0:('Fig1_system_layers.png','图1｜系统分层与本文边界。实线已实现，拟实现模块不作为成果。来源：03_FIGURES/provenance。',225),
        2:('Fig2_method_graph.png','图2｜从名义动作到合法参数与物理结果；拟机制必须实际进入Actor，不能止于预测头。',245),
        3:('Fig3_physx_scene_board.png','图3｜旧v6b策略实时PhysX重评，非新B3训练或固定轨迹播放。3号案例后续成功、1号后续超时，所示均非终帧；16环境总评55/64独立通过。外观未改物理属性。SYNTHETIC自由方块，非现场料盘或接触抓取。',240),
        4:('Fig5_historical_frozen_tests.png','图5｜历史冻结测试重建。每split32同案例，三seed；P1/P2单列，训练seed SD不作置信区间。',245),
        5:('FigS1_actual_contact_state_diagnostic.png','物理机制诊断｜不同案例的失败/成功实测轨迹，不是同初态因果实验。旋转、偏离和速度问题须通过新配对干预验证。',240),
        6:('Fig4_historical_online_learning.png','图4｜历史真实在线学习记录；训练回报与独立测试成功不是同一指标。',230),
        8:('Fig6_new_pilot.png','图6｜新P2开发验证与真实输入干预。每模型32例，B1/B2/B3三种子；干预是同checkpoint的重评，不是额外训练种子。原始数据见NEW_PILOT_RESULTS.csv与因果轨迹表。',230),
    }
    for i,(heading,blocks) in enumerate(sections):
        if i: story.append(PageBreak())
        story.append(para(f'RESEARCH PROPOSAL  /  {i+1:02d}', 'tag'))
        story.append(para(heading,'h1'))
        for j,b in enumerate(blocks):
            if i==0 and j==0:
                story.extend(box(b))
            else:
                story.append(para(b))
        if i in figures:
            story.extend(figure(*figures[i]))
        if i==7:
            story.extend(table([
                ['证据层级','当前可以说什么','不可提前声称'],
                ['EXACT / VERIFIED','16环境真实在线物理；历史三种子残差比较','完整现场料盘/双臂自主完成'],
                ['PROTOTYPE','任务局部残差与结果反馈pilot，以run记录为准','单seed即显著、已证实新算法'],
                ['PROPOSED / NOT TESTED','后续条件消融、5种子及新封测试','跨机器人/视觉部署/工业安全'],
            ],[90,212,217]))
    document(ROOT/'01_CLIENT_PROPOSAL_CN.pdf',story,'ARES 甲方SCI接触技能强化学习研究方案')

def build_brief():
    sections=md_sections('07_MENTOR_BRIEF_CN.md')
    story=[para('MENTOR BRIEF  /  01','tag'),para('接触技能强化学习\n从真实结果到可检验方法','h1')]
    for i,(heading,blocks) in enumerate(sections):
        if i==2:
            story.append(PageBreak())
            story.append(para('MENTOR BRIEF  /  02','tag'))
        story.append(para(heading,'h2'))
        story.extend(para(b) for b in blocks)
    document(ROOT/'07_MENTOR_BRIEF_CN.pdf',story,'ARES 强化学习SCI导师简报')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest-only',action='store_true',
                        help='Refresh hashes without rebuilding unchanged Chinese PDFs')
    args=parser.parse_args()
    cmaps=[pdfmetrics.getFont(n).face.charToGlyph for n in ['ClientCJK','ClientLatin']]
    for source in ['01_CLIENT_PROPOSAL_CN.md','07_MENTOR_BRIEF_CN.md']:
        missing=sorted({c for c in (ROOT/source).read_text() if not c.isspace()
                        and not any(ord(c) in m for m in cmaps)})
        if missing:
            raise ValueError(f'Unsupported document glyphs in {source}: {missing}')
    if not args.manifest_only:
        build_proposal();build_brief()
    outputs=['01_CLIENT_PROPOSAL_CN.pdf','07_MENTOR_BRIEF_CN.pdf']
    if (ROOT/'02_PAPER_SEED/main.pdf').is_file():
        outputs.append('02_PAPER_SEED/main.pdf')
    manifest=[]
    from pypdf import PdfReader
    for name in outputs:
        p=ROOT/name
        manifest.append({'path':name,'bytes':p.stat().st_size,
                         'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'pages':len(PdfReader(p).pages)})
    (ROOT/'DOCUMENT_BUILD_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
