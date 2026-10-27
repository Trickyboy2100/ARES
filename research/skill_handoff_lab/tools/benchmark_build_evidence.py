"""Build figures/brief strictly from completed public-benchmark run artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--runs', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--pdf-only', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)

if not a.pdf_only:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    rows = []
    for path in sorted(a.runs.glob('*/manifest.json')):
        d = json.loads(path.read_text())
        if 'episodes' not in d:
            continue
        valid = [e for e in d['episodes'] if e['steps'] > 0 and e['error'] in (None, 'NATIVE_HORIZON_EXHAUSTED', 'PLANNER_REPORTED_FAILURE')]
        n = sum(len(e['native_success_ever']) for e in valid)
        success = sum(sum(e['native_success_ever']) for e in valid)
        final = sum(sum(e['native_final']['success']) for e in valid)
        rows.append({'run': path.parent.name, 'task': d['task'], 'controller': d['controller'],
                     'n': n, 'native_success_ever': success, 'final_success': final,
                     'scope': d.get('evaluation_scope', 'NATIVE_HORIZON'), 'source_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'invalid_episodes': len(d['episodes'])-len(valid)})
    (a.output / 'RESULTS.json').write_text(json.dumps(rows, indent=2))
    selected = [r for r in rows if r['run'] in ['ms_stack_feedback_validation64', 'ms_pick_ppo_256k_eval16',
        'ms_stack_ppo_1024k_eval64', 'ms_stack_residual_256k_eval64', 'ms_stack_residual_zero_eval64', 'ms_stack_cartesian_256k_eval64',
        'ms_stack_residual_seed37_eval64', 'ms_stack_residual_seed43_eval64']]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    labels = {'ms_stack_feedback_validation64': 'Stack: feedback', 'ms_pick_ppo_256k_eval16': 'Pick: official PPO 256k',
              'ms_stack_ppo_1024k_eval64': 'Stack: joint PPO 1.024M',
              'ms_stack_residual_256k_eval64': 'Stack: residual PPO 256k',
              'ms_stack_residual_zero_eval64': 'Stack: residual forced zero',
              'ms_stack_residual_seed37_eval64': 'Stack: residual PPO seed37',
              'ms_stack_residual_seed43_eval64': 'Stack: residual PPO seed43',
              'ms_stack_cartesian_256k_eval64': 'Stack: Cartesian PPO 256k'}
    for i, r in enumerate(selected):
        y = r['final_success']/r['n']
        ax.barh(i, y, color='#177b83' if 'feedback' in r['run'] else '#3567a0')
        ax.text(min(y+.015,.91), i, f"{r['final_success']}/{r['n']}", va='center')
    ax.set(yticks=range(len(selected)), yticklabels=[labels[r['run']] for r in selected], xlim=(0,1.1),
           xlabel='Native final success (development validation; not sealed test)',
           title='Real ManiSkill/SAPIEN executions | fixed-budget pilots, seeds 31/37/43')
    fig.savefig(a.output/'native_validation.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    training = []
    for run in sorted((a.runs/'runs').glob('*')):
        log = run/'training.log'
        if not log.exists():
            seed = re.search(r'seed(\d+)', run.name)
            suffix = '_seed'+seed[1] if seed and seed[1] != '31' else ''
            log = a.runs/(('stack_residual'+suffix+'.log') if 'residual' in run.name else 'stack_cartesian.log')
        if not log.exists():
            continue
        step, points = 0, []
        for line in log.read_text().splitlines():
            m = re.search(r'global_step=(\d+)',line)
            if m: step=int(m[1])
            m = re.match(r'eval_(reward|success_at_end)_mean=([0-9.eE+-]+)',line)
            if m: points.append({'step':step,'metric':m[1],'value':float(m[2])})
        short=run.name.replace('ms_','').replace('_seed31','')
        for ax,metric in zip(axes,['reward','success_at_end']):
            q=[x for x in points if x['metric']==metric]
            ax.plot([x['step'] for x in q],[x['value'] for x in q],'.-',label=short,linewidth=1.3)
            ax.set(xlabel='Online training interactions',ylabel=metric)
        training.append({'run':run.name,'points':points,'checkpoint_complete':(run/'final_ckpt.pt').exists()})
    axes[1].set_ylim(-.03,1.05); axes[0].legend(fontsize=6)
    fig.suptitle('Actual evaluation logs | 4 eval envs; individual training seeds, no CI claim')
    fig.savefig(a.output/'learning_curves.png',dpi=180);plt.close(fig)
    (a.output/'LEARNING_DATA.json').write_text(json.dumps(training,indent=2))
    # All 64 feedback trajectories; no best-case filtering.
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    failure=[]
    for path in sorted((a.runs/'ms_stack_feedback_validation64').glob('episode_*.json')):
        d=json.loads(path.read_text()); states=np.array([x['state']['actors']['cubeA'] for x in d['transitions']])
        for i,ok in enumerate(d['summary']['native_final']['success']):
            color='#16867d' if ok else '#c04e45'
            axes[0].plot(np.arange(len(states))/20,states[:,i,2],color=color,alpha=.25)
            axes[1].plot(np.arange(len(states))/20,np.linalg.norm(states[:,i,7:10],axis=1),color=color,alpha=.25)
            if not ok: failure.append({'batch_seed':d['summary']['seed'],'env_index':i,'final':d['summary']['native_final']})
    axes[0].set(xlabel='Physical time (s)',ylabel='Cube A height (m)')
    axes[1].set(xlabel='Physical time (s)',ylabel='Cube A linear speed (m/s)')
    fig.suptitle('Stack feedback: all 64 executions | teal=success, red=failure')
    fig.savefig(a.output/'feedback_physics.png',dpi=180);plt.close(fig)
    (a.output/'FEEDBACK_FAILURES.json').write_text(json.dumps(failure,indent=2))
    paired=[]
    base=a.runs/'ms_stack_feedback_validation64'
    for name in ['ms_stack_residual_256k_eval64','ms_stack_residual_zero_eval64','ms_stack_residual_seed37_eval64','ms_stack_residual_seed43_eval64']:
        directory=a.runs/name
        if not (directory/'manifest.json').exists(): continue
        gains=losses=both=neither=0
        action_deltas=[]
        identical_initial_states=True
        for path in sorted(directory.glob('episode_*.json')):
            d=json.loads(path.read_text()); b=json.loads((base/path.name).read_text())
            identical_initial_states &= d['initial_state'] == b['initial_state']
            x=np.array(b['summary']['native_final']['success']); y=np.array(d['summary']['native_final']['success'])
            gains+=int((~x&y).sum());losses+=int((x&~y).sum());both+=int((x&y).sum());neither+=int((~x&~y).sum())
            for t in d['transitions']:
                audit=t['info'].get('controller_action_audit')
                if audit: action_deltas.extend(np.linalg.norm(np.array(audit['executed'])[:,:3]-np.array(audit['nominal'])[:,:3],axis=1).tolist())
        paired.append({'run':name,'fixed_feedback_failures':gains,'introduced_failures':losses,
                       'both_success':both,'both_failure':neither,
                       'exact_initial_state_match_all_cases':identical_initial_states,
                       'mean_normalized_xyz_action_change':float(np.mean(action_deltas)) if action_deltas else None})
    (a.output/'PAIRED_CAUSAL_AUDIT.json').write_text(json.dumps(paired,indent=2))
    jaka=a.runs/'jaka_eg2_float_init'
    if (jaka/'manifest.json').exists():
        import xml.etree.ElementTree as ET
        jd=json.loads((jaka/'manifest.json').read_text())
        q=np.load(jaka/'physical_joint_trace.npz')['q'][:,0]
        joints={n.get('name'):n for n in ET.parse(Path(__file__).resolve().parents[3]/'robot/jaka_minicobo_gripper.urdf').getroot().findall('joint')}
        checks=[]
        for i,name in enumerate(jd['joint_names']):
            original=name.removeprefix('eg2_'); node=joints[original]; limit=node.find('limit')
            lo,hi=float(limit.get('lower')),float(limit.get('upper'))
            violation=max(0.,float((lo-q[:,i]).max()),float((q[:,i]-hi).max()))
            check={'joint':name,'max_limit_excess_rad':violation,
                   'peak_finite_difference_velocity_rad_s':float(np.abs(np.diff(q[:,i])/jd['physics_dt']).max()),
                   'urdf_velocity_limit':float(limit.get('velocity'))}
            mimic=node.find('mimic')
            if mimic is not None:
                j=jd['joint_names'].index('eg2_'+mimic.get('joint'))
                check['max_mimic_error_rad']=float(np.abs(q[:,i]-float(mimic.get('multiplier','1'))*q[:,j]-float(mimic.get('offset','0'))).max())
            checks.append(check)
        (a.output/'JAKA_INDEPENDENT_AUDIT.json').write_text(json.dumps({
            'physics_run_complete':True,'grasp_tested':False,
            'qualification':'FAIL' if any(x['max_limit_excess_rad']>.001 for x in checks) else 'NOT_FULLY_TESTED',
            'checks':checks,'note':'drive smoke only; no complete collision/contact or industrial safety validation'},indent=2))
    for label,batch,index in [('improvement',1,2),('regression',9,2)]:
        left=base/f'episode_{batch:03d}.mp4'
        right=a.runs/'ms_stack_residual_256k_eval64'/f'episode_{batch:03d}.mp4'
        if right.exists():
            seed=27101+4*batch
            filters=(f'[0:v]crop=512:512:{512*index}:0,setpts=2*PTS,drawtext=text=Feedback:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.6:x=12:y=12[l];'
                     f'[1:v]crop=512:512:{512*index}:0,setpts=2*PTS,drawtext=text=Residual_PPO_seed31:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.6:x=12:y=12[r];'
                     f'[l][r]hstack,drawtext=text=ManiSkill_SAPIEN__batch_seed_{seed}__env_{index}__half_speed:fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:x=12:y=h-30[v]')
            output=a.output/f'paired_{label}.mp4'
            subprocess.run(['ffmpeg','-loglevel','error','-y','-i',str(left),'-i',str(right),'-filter_complex',filters,'-map','[v]','-c:v','libx264','-crf','21',str(output)],check=True)
            subprocess.run(['ffmpeg','-loglevel','error','-y','-ss','4.4','-i',str(output),'-frames:v','1',str(a.output/f'paired_{label}.png')],check=True)
    drawer='isaac_drawer4_front_view' if (a.runs/'isaac_drawer4_front_view/manifest.json').exists() else 'isaac_drawer4_official_sm_v2'
    for run,name,second in [(drawer,'isaac_drawer',4),
                            ('ms_stack_feedback_validation64','stack_feedback',2),
                            ('ms_peg1_gpu_pose_diagnostic','peg_planner',4)]:
        source=a.runs/run/('actual_execution.mp4' if run.startswith('isaac') else 'episode_000.mp4')
        if source.exists():
            subprocess.run(['ffmpeg','-loglevel','error','-y','-ss',str(second),'-i',str(source),'-frames:v','1',str(a.output/(name+'.png'))],check=True)
    hashes={str(x.relative_to(a.output)):hashlib.sha256(x.read_bytes()).hexdigest() for x in a.output.iterdir() if x.is_file()}
    (a.output/'EVIDENCE_SHA256.json').write_text(json.dumps(hashes,indent=2))
else:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('CJK','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
    pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    style=ParagraphStyle('body',fontName='CJK',fontSize=10,leading=16,spaceAfter=10)
    title=ParagraphStyle('title',parent=style,fontSize=19,leading=26,spaceAfter=18)
    def para(text,st=style):
        parts=[]
        for char in text:
            font='CJK' if ord(char)>127 else 'Latin'
            parts.append(f'<font name="{font}">{char.replace("&","&amp;").replace("<","&lt;")}</font>')
        return Paragraph(''.join(parts),st)
    story=[para('ARES：公开基准与 JAKA 双轨研究',title),para('2026-10-27 | 真实物理实验阶段简报'),
           para('本轮已运行 Isaac Lab 原生 Lift、Stack、Drawer，以及独立环境中的 ManiSkill Pick、Stack、Peg。旧 P1/P2 与现场结果完整保留；以下不是实机成绩，也不是已证明的新算法优势。'),
           Image(str(a.output/'native_validation.png'),width=490,height=250),
           para('强反馈 58/64；三颗普通残差 PPO 种子为 61/64、62/64、60/64，均值 95.31%，种子标准差 1.56 个百分点。残差置零恢复 58/64 全部相同结果。共完成 7 个最终模型、256 万次在线训练交互；同一 64 案例不是 192 个独立场景，不能据此宣称未见场景泛化或新方法创新。'),
           PageBreak(),para('物理证据与训练，而非静态演示',title),
           Image(str(a.output/'learning_curves.png'),width=490,height=192),
           Image(str(a.output/'feedback_physics.png'),width=490,height=196),
           para('曲线直接来自原始训练日志。失败轨迹与成功轨迹全部保留。Stack 的剩余失败涉及释放、放置偏差和稳定；Peg 的 GPU 几何读数修复后 4/4 可完成，但多数超过官方 100 步预算，不能计入官方期限内成功率。'),
           PageBreak(),para('研究选择与尚未成立的贡献',title),
           Image(str(a.output/'isaac_drawer.png'),width=490,height=276),
           para('图为 Isaac Sim 5.1 / Isaac Lab 2.3.0 实际抽屉执行截图。原生 Lift/Drawer 配置没有统一成功终止项，报告提升高度、抽屉位移而不虚构官方成功率。'),
           para('公共主平台选择 ManiSkill3，Stack 作为快速闭环与释放稳定性任务，Peg 为不同接触条件候选。JAKA 应用轨道保持 Isaac Lab；已完成 720 物理步开合，但夹爪限位及 mimic 验收失败，尚不能称为抓取或任务迁移成功。'),
           para('待验证方法：根据当前接触、抓取及放置结果调整合法技能参数或恢复动作。普通残差控制不是创新本身。下一步必须在相同接口下完成强反馈、普通残差和结果感知参数策略的多种子消融。'),
           para('许可：ManiSkill 代码 Apache-2.0；上游资产存在 CC BY-NC 提示，商业交付范围需另行核验。本包不重新分发仿真资产。完整数值、失败与来源见 RESULTS.json、LEARNING_DATA.json 和运行清单。')]
    SimpleDocTemplate(str(a.output/'BENCHMARK_BRIEF_CN.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=44,bottomMargin=44).build(story)
