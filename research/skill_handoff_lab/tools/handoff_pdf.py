"""Chinese mentor/client PDF, with mixed CJK/Latin font fallback and real figures."""
import argparse,html,itertools,json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output;d=json.loads((o/'STATISTICAL_SUMMARY.json').read_text());cost=json.loads((o/'COST_LEDGER.json').read_text());stats={r['method']:r for r in d['methods']};ci=d['candidate_minus_ordinary']['reset_batch_and_seed_bootstrap_ci_pp']
 pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('CN','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
 def mixed(s):return ''.join('<font name="Latin">'+html.escape(''.join(cs))+'</font>' if latin else html.escape(''.join(cs)) for latin,cs in itertools.groupby(s,lambda c:ord(c)<0x2e80))
 body=ParagraphStyle('body',fontName='CN',fontSize=10.5,leading=17,textColor=HexColor('#203b51'),wordWrap='CJK',spaceAfter=10)
 heading=ParagraphStyle('heading',parent=body,fontSize=19,leading=28,spaceAfter=18)
 small=ParagraphStyle('small',parent=body,fontSize=9,leading=14)
 story=[]
 def text(s,style=body):story.append(Paragraph(mixed(s),style))
 def title(s):text(s,heading)
 def image(name,height):
  f=o/'FIGURES'/name;w,h=ImageReader(str(f)).getSize();scale=min(490/w,height/h);story.append(Image(str(f),w*scale,h*scale));story.append(Spacer(1,8))
 def page():story.append(PageBreak())
 title('研究结论：物理机会成立，候选机制未增益')
 text('唯一论文问题：多阶段操作中，当前技能实际松爪并稳定后，仍可能给后续技能留下不利状态。研究在两个真实执行边界上选择合法参数，能否改善同权限普通RL的收益与退化。')
 text('本轮已完成：公开Panda / 原生StackPyramid-v1连续物理执行器；64个新场景的3×3干预；普通双边界PPO与风险约束候选各3种子；3个等预算消融；全部64未见开发场景对照。不是实机、视觉Actor或现场成功。')
 rows=[['独立开发64例','原生成功数'],['名义 / 几何规则 / 固定搜索','53 / 55 / 53'],['绝对Q / 成对风险评分','53 / 50'],['普通PPO：31 / 37 / 43','51 / 53 / 56'],['候选PPO：31 / 37 / 43','51 / 54 / 55']]
 table=Table([[Paragraph(mixed(x),body) for x in row] for row in rows],colWidths=[300,185]);table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e2edf0')),('GRID',(0,0),(-1,-1),.5,HexColor('#c5d3da')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.append(table);story.append(Spacer(1,14))
 text(f'候选减普通PPO平均0个百分点，三种子差值0、+1、−1例。按重置批次和训练种子交叉bootstrap95%描述区间为[{ci[0]:.2f}, {ci[1]:.2f}]个百分点。结果不支持本候选的独立方法收益，不选最佳种子。')
 text(f'总计{cost["total_external_control_interactions"]:,}次新外部控制交互，全部计入工程失败和250步中的终态保持。主在线RL288,000，消融48,000，公共物理采集144,000，独立开发240,000，工程{cost["engineering_interactions"]:,}。')
 text('当前不足以作为已验证SCI新方法投稿。下一步应由明确的物理失败和模型校准问题定义一个可证伪机制；有可靠增量才扩五种子、封存测试和第二物理任务。',small)
 page();title('两个决策边界是真实发生的物理事件');image('Fig2_two_boundary_method.png',300)
 text('第一决策：A支撑放置方向取−90 / 0 / +90度。执行实际抓取、100mm抬起、放置、开爪和60mm退让。只有独立验收合格后才调用第二策略。')
 text('第二决策：观察真实A/B/C状态，再选C抓取朝向−0.35 / 0 / +0.35弧度；真实抓取和放置到观测到的A/B中点，随后松爪验收。技能之间没有reset、瞬移或缓存结果。')
 text('验收：连续3拍夹爪开度>65mm、物体不再被抓持；所有方块线速度<10mm/s、角速度<0.5rad/s；机器人速度<0.15rad/s；XY误差<15mm。A需桌面接触和高度误差<8mm；C需支撑接触和高度误差<8mm。原生最终成功另行计算。')
 text('有限时域gamma=1，真实记录持续时间。前段失败没有虚拟第二token；每条评价实际运行满250步（12.5秒仿真时间），逻辑提前完成不意味着节省物理时间。',small)
 page();title('同初态干预：前段合格仍可改变后续结果');image('Fig3a_physical_opportunity_grid.png',250);image('Fig3b_intervention_physical_state.png',210)
 text('64个全新训练设计场景：名义52/64，最佳固定59/64，九候选已见并集62/64。并集只是机会探针，不是部署策略分数。9例在A均通过真实验收时改变后续结果。181个可比前缀的C决策输入完全相同，其中18组因C参数不同而改变原生成功。')
 text('Case420010两条A放置都通过松爪稳定；相同C后缀后，A0支撑间距越过原生阈值而失败，A2保持成功。图为保存的物体和关节状态。可证明参数的总物理效应，不能单凭相关散点声称margin是唯一因果中介。',small)
 page();title('真实训练与全部种子：不能把工程修复算算法收益');image('Fig4_online_learning.png',245);image('Fig5a_development_all_seeds.png',315)
 text('所有学习方法共用57维特权状态、三选一动作、名义planner和原生250步。普通PPO和候选共用预采集、初始化、网络和48,000在线交互/种子。候选只加固定成对退化风险正则。强规则与评分器同样保留。',small)
 page();title('消融和反例：机制证据仍不成立');image('Fig6_information_and_redecision_ablations.png',245)
 text('三个消融各训练16,000在线交互，与完整候选的16,000检查点比较：完整51/64；去next几何52；去物体后验55；初态提交两段参数53。单种子且冻结教师可能受掩码分布漂移影响，不能解释成后验无用或提出正向机制结论。')
 text('普通PPO与候选训练种子差异大于正则增量。成对风险评分器仅50/64，提示风险估计本身尚不能稳定指导选择。绝对结果模型训练Brier0.0104、独立校准批次0.0779，存在泛化缺口；风险头训练仅9/546个正标签；独立校准Brier0.0233，弱于常数基线0.0207。不能把预测风险当安全置信保证。')
 text('保存两种定性对照：训练种子37的真实救回，以及种子43的真实退化。影片来自独立开发评价保存的实际关节/物体状态，逐项读回误差约1.2e−7；渲染不运行新的物理步，不增加统计样本。所有失败和全部64例保留在CSV。')
 text('普通完整状态可能已经包含执行效果；只删除历史差分不足以证明效果记忆机制。本轮不强行宣称历史增益、跨任务能力或新RL理论。',small)
 page();title('论文范围、复现入口与下一步决策');image('Fig1_scope_and_system.png',260)
 text('一键检查：11_PAPER_FIRST_20261028/VIDEOS/index.html。采集主机打开runs/single_paper_20261028/visuals/OPEN_ME.html，四组影片和全部图可直接查看。大型录像留非Git；公开结构化数据和原始状态子集可重新渲染。')
 text('模型：16个正式文件，包括6组主模型及其等预算检查点、3组消融和公共先验；MODEL_INDEX记录SHA。RAW_SOURCE_INDEX、SCENE_SHA_INDEX、SOURCE_PHASES保留原始轨迹、场景和精确阶段源码。网络隔离仅loopback、GPU0、无串口；ARES-R挂载只读。')
 text('新论文主稿：LATEX/PAPER_SKILL_HANDOFF.tex。已有RAPS题名已按NeurIPS2021正式来源修正；STAP、外科ViSkill、Foresight、DAWN明确界定已有思想。参数化动作、下游价值、Q减名义Q均不是本轮原创。')
 text('不打开620000..620127封存场景，不扩弱Peg或饱和StackCube。接下来只围绕一个科学问题：实测支撑/抓取可行性如何进入可执行参数策略，并在强规则、评分器和同接口RL之上取得稳定增量。若做不到，应收缩为物理技能接口的系统性实证研究，不能包装已发表级方法。',small)
 def footer(c,doc):
  c.saveState();c.setStrokeColor(HexColor('#c4d1d8'));c.line(45,38,550,38);para=Paragraph(mixed(f'ARES SCI研究工作稿 | 2026-10-28 | 公开仿真，非现场/实机 | {doc.page}'),small);para.wrap(500,20);para.drawOn(c,45,18);c.restoreState()
 SimpleDocTemplate(str(o/'MENTOR_DECISION_CN.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=53,title='ARES单篇SCI：技能边界参数RL真实实验与研究判断',author='ARES Research').build(story,onFirstPage=footer,onLaterPages=footer)
 print('MENTOR_DECISION_CN.pdf built')
if __name__=='__main__':main()
