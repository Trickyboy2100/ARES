"""Scientific figures / actual simulation media / honest client and paper updates."""
import argparse,csv,hashlib,json,shutil,subprocess
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def figures(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image,ImageDraw,ImageFont
    out=args.output;figs=out/'figures';figs.mkdir(parents=True,exist_ok=True);media=out/'media';media.mkdir(exist_ok=True);runs=args.runs
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    result=load(out/'RESULTS.json');table=result['task_scores'];entries=[]
    def finish(fig,name,title,note,source):
        fig.text(.065,.96,title,fontsize=19,weight='bold',va='top',color='#203b51');fig.text(.065,.885,note,fontsize=10,va='top',color='#4a6071',linespacing=1.5);fig.text(.065,.025,'NEW SAPIEN / PHYSX SIMULATION   |   NO PHYSICAL ROBOT   |   DEVELOPMENT, NOT SEALED TEST',fontsize=9,color='#4a6071')
        fig.savefig(figs/(name+'.png'),dpi=200,facecolor='white');fig.savefig(figs/(name+'.svg'),facecolor='white');plt.close(fig);entries.append(dict(file='figures/'+name+'.png',source=source,source_kind='NEW_PHYSICAL_SIMULATION_OR_MODEL_FIT',sha256=sha(figs/(name+'.png'))))
    def chart():
        fig,ax=plt.subplots(figsize=(12,7));fig.subplots_adjust(top=.74,bottom=.18,left=.12,right=.95);return fig,ax
    for name,kind in [('Fig1_system','system'),('Fig2_method','method')]:
        fig=plt.figure(figsize=(12,7));ax=fig.add_axes([.04,.13,.92,.62]);ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
        labels=(['Native task\nphysics / reward / horizon','Versioned skills\nlegal parameters','Actual commands\nobject effects','Native terminal\noutcome','Train-only paired\ncold-reset branches','Shared PPO / CEM\npaired vs absolute model'] if kind=='system' else ['Current state\nprevious physical effect','One legal place\nboundary decision','Three legal choices\nKEEP / lower / higher','Actual task suffix\nfull native outcome','Train-only supervision\npaired outcome delta','Empirical ensemble\nmean - SD > 0.05'])
        positions=[(.25,3.8),(4.3,3.8),(8.3,3.8),(.25,.6),(4.3,.6),(8.3,.6)]
        for i,((x,y),label) in enumerate(zip(positions,labels)):
            ax.add_patch(FancyBboxPatch((x,y),3.35,1.4,boxstyle='round,pad=.15',edgecolor='#3a7184',facecolor='#e5eff1'));ax.text(x+1.675,y+.7,label,ha='center',va='center',fontsize=13)
        # Boundary-to-boundary arrows, avoiding text and crossing flows.
        for before,after in [(0,1),(1,2),(3,4),(4,5)]:
            x,y=positions[before];xx,yy=positions[after]
            ax.annotate('',xy=(xx-.15,yy+.7),xytext=(x+3.5,y+.7),arrowprops=dict(arrowstyle='->',lw=1.8,color='#7895a5'))
        ax.annotate('',xy=(1.925,2.15),xytext=(1.925,3.65),arrowprops=dict(arrowstyle='->',lw=1.8,color='#7895a5'))
        ax.text(2.15,2.8,'full physical suffix',fontsize=10,color='#4a6071')
        finish(fig,name,'Measured skill authority and outcome comparison' if kind=='system' else 'Implemented one-boundary selector (contextual bandit)', 'Implemented components only. No new PPO likelihood, no trained latent dynamics model, no claimed safety guarantee.\nThe empirical abstention mechanism is evaluated even when it yields no improvement.', ['method_stack_physics.py','method_outcome_selector.py','BENCHMARK_PROTOCOL.json'])
    # Native success uses full 0-100% axes; paired count plots carry small effects honestly.
    fig,ax=chart();labels=[r['policy'] for r in table];values=[r['success_rate']*100 for r in table];colors=['#346c86' if x.startswith('residual') else '#d2824b' if x.startswith('parameter') else '#4a8e7e' if x in ['paired','no_abstain'] else '#8796a3' for x in labels]
    ax.bar(range(len(labels)),values,color=colors);ax.set_ylim(0,108);ax.set_ylabel('Native terminal success [%]');ax.set_xticks(range(len(labels)),labels,rotation=55,ha='right',fontsize=9);fig.subplots_adjust(bottom=.30)
    for i,r in enumerate(table):ax.text(i,values[i]+1,str(r['successes']),ha='center',fontsize=9)
    finish(fig,'Fig6a_native_results','No independent method gain on the new 64 development conditions','Ordinary Residual PPO: 63 / 62 / 64; parameter PPO: 62 / 63 / 60.\nPaired abstention: 60; fixed and equal-cost CEM: 62. Counts above bars are out of 64.', ['EXPERIMENT_RESULTS.csv'])
    fig,ax=chart();selected=[r for r in table if r['policy']!='nominal'];yy=np.arange(len(selected))
    ax.barh(yy,[r['rescues'] for r in selected],color='#4a8e7e',label='rescue')
    ax.barh(yy,[-r['regressions'] for r in selected],color='#be705e',label='regression')
    ax.set_yticks(yy,[r['policy'] for r in selected],fontsize=9);ax.invert_yaxis();ax.axvline(0,color='#203b51',lw=.8)
    ax.set(xlim=(-3,5),xlabel='Paired case counts (left: regression; right: rescue)');ax.set_xticks(range(-3,6),[str(abs(x)) for x in range(-3,6)]);ax.legend(loc='lower right',fontsize=10);ax.grid(axis='x',alpha=.15);fig.subplots_adjust(left=.27)
    finish(fig,'Fig6b_paired_tradeoff','Success totals hide different rescue / regression sets','Every bar uses the SAME 64 native initial conditions. Training seeds are separate models, not extra scenes.\nCounts show both directions of effect; no oracle is used for deployment.', ['CASE_PAIRED_CONFUSION.csv'])
    curves=load(out/'LEARNING_CURVES.json')
    for mode in ['residual','parameter']:
        fig,ax=chart()
        for seed,c in zip([31,37,43],['#346c86','#d2824b','#4a8e7e']):
            rows=[r for r in curves if r['policy']==mode and r['seed']==seed and r['tag']=='train/success_at_end'];ax.plot([r['step'] for r in rows],[r['value'] for r in rows],label=f'seed {seed}',color=c,lw=1.5)
        ax.set(xlabel='Actual online native control interactions',ylabel='On-policy native terminal success',ylim=(0,1.02));ax.legend();ax.grid(alpha=.15)
        finish(fig,'Fig5_'+mode,'Actual ordinary PPO learning: '+mode,'102,400 online interactions per seed; original dense reward and 50-step horizon unchanged.\nTraining traces are not sealed-test generalization curves. Concurrent wall times are not speed comparisons.', ['LEARNING_CURVES.json','ACTOR_UPDATE_AUDIT.json'])
    # Read actual recorded trajectories from the video reruns, verified against evaluation.
    reference=load(runs/'validation64_r1/RESULTS.json');ref={(r['policy'],r['seeds'][0]):r for r in reference['rows']}
    video_receipts=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    for folder in ['stack_video_batch0','stack_video_batch1']:
        if not (runs/folder/'RESULTS.json').exists():continue
        for r in load(runs/folder/'RESULTS.json')['rows']:
            original=ref[r['policy'],r['seeds'][0]];assert r['initial_sha256']==original['initial_sha256'] and r['success']==original['success']
        for file in (runs/folder).glob('*.mp4'):
            target=media/file.name;shutil.copyfile(file,target);video_receipts.append(dict(path='media/'+target.name,source=str(file),source_sha256=sha(file),sha256=sha(target),kind='ACTUAL_PHYSICS_RERUN_NOT_NEW_STATISTICAL_SAMPLE',reproduces_primary_evaluation=True))
    def frame(file,index):
        content=subprocess.check_output(['ffmpeg','-v','error','-i',str(file),'-vf',f'select=eq(n\\,{index})','-frames:v','1','-f','image2pipe','-vcodec','png','-'])
        import io
        return Image.open(io.BytesIO(content)).convert('RGB')
    for case,policies,name in [(200003,['nominal','parameter_ppo_43'],'Fig3a_stack_rescue'),(200006,['nominal','parameter_ppo_43'],'Fig3b_stack_regression'),(200020,['nominal','no_abstain'],'Fig3c_selector_regression')]:
        if not all((media/f'{policy}_case{case}.mp4').exists() for policy in policies):continue
        canvas=Image.new('RGB',(2048,1280),'white');draw=ImageDraw.Draw(canvas);draw.text((20,10),f'ACTUAL SAPIEN / PhysX | StackCube-v1 case {case} | selected paired example, not a rate estimate',font=font,fill='#203b51')
        for row,policy in enumerate(policies):
            for col,index in enumerate([9,24,37,49]):
                image=frame(media/f'{policy}_case{case}.mp4',index);image.thumbnail((500,570));canvas.paste(image,(col*512+6,70+row*600))
        canvas.save(figs/(name+'.png'));entries.append(dict(file='figures/'+name+'.png',source=[f'media/{x}_case{case}.mp4' for x in policies],source_kind='ACTUAL_SIMULATION_FRAMES',sha256=sha(figs/(name+'.png'))))
    for case,batch in [(200003,0),(200006,0)]:
        folder=runs/'stack_video_batch0';index=case-200000
        for feature in ['object_height','gripper_opening']:
            fig,ax=chart()
            for label,c in [('nominal','#346c86'),('parameter_ppo_43','#d2824b'),('residual_ppo_43','#4a8e7e')]:
                records=load(folder/f'{label}_{batch:03d}.json')['transitions'];t=np.array([r['step'] for r in records])*.05
                y=np.array([r['cubeA'][index][2]-r['cubeB'][index][2] for r in records])*1000 if feature=='object_height' else np.array([sum(r['q'][index][-2:]) for r in records])*1000
                ax.plot(t,y,label=label,color=c,lw=1.8)
            ax.set(xlabel='Actual native simulation time [s]',ylabel='A minus B center height [mm]' if feature=='object_height' else 'Measured two-finger opening [mm]');ax.legend(fontsize=10);ax.grid(alpha=.15)
            finish(fig,f'Fig4_{feature}_{case}',f'Measured physical response: case {case}', 'Actual object and gripper states; curves are not regenerated desired TCP paths.\nNative success also requires geometry, staticness and release; a height curve alone is not success.', ['stack_video_batch0 native transition JSON hashes in RAW_SOURCE_INDEX'])
    # Peg failure has measured position error; do not turn the trace into a positive baseline.
    fig,ax=chart()
    for label,c in [('tracked_nominal','#346c86'),('reactive','#d2824b'),('align_then_advance','#4a8e7e')]:
        records=load(runs/'peg_feedback_r3'/f'case000_{label}.json')['transitions'];ax.plot(np.arange(1,len(records)+1)*.05,[r['info']['peg_head_pos_at_hole'][0][0]*1000 for r in records],label=label,color=c)
    ax.axhline(-15,color='#9a5d63',ls='--',label='native X requirement (Y/Z also required)');ax.set(xlabel='Native simulation time [s]',ylabel='Peg head axial error in hole frame [mm]');ax.legend(fontsize=9);ax.grid(alpha=.15)
    finish(fig,'Fig7_peg_failure','Peg: physical insertion remains a weak baseline','8 design conditions: official/tuned first round 0/8; tracked / reactive round 1/8.\nNo Peg RL improvement is claimed. All runs respect the original 100-step cap.', ['peg_feedback_r3 raw episodes','CONTROL_AUTHORITY_MAP.csv'])
    # Multi-stage authority table and actual successor geometry.
    p=load(runs/'pyramid_paired_parameters/RESULTS.json');policies=['tracked_nominal','first_dx_minus','first_dx_plus'];success=[sum(r['success'] for r in p['rows'] if r['policy']==label) for label in policies]
    fig,ax=chart();ax.bar(policies,np.array(success)/16*100,color=['#346c86','#4a8e7e','#d2824b']);ax.set(ylabel='Native full pyramid success [%]',ylim=(0,108));
    for i,value in enumerate(success):ax.text(i,value/16*100+2,f'{value}/16',ha='center')
    finish(fig,'Fig8a_pyramid_authority','First-skill target changes the downstream physical task','Same cold resets, original 250-step sparse task, native planner controller.\nMinus12mm: one rescue AND one regression; plus12mm: five regressions. Not an RL gain.', ['pyramid_paired_parameters raw episodes and hashes'])
    for seed,name in [(122007,'rescue'),(122005,'regression')]:
        case=seed-122000;fig,ax=chart()
        for policy,c in zip(policies,['#346c86','#4a8e7e','#d2824b']):
            records=load(runs/'pyramid_paired_parameters'/f'{case:03d}_{policy}.json')['transitions'];distance=[np.linalg.norm(np.asarray(r['state']['actors']['cubeA'][0][:2])-np.asarray(r['state']['actors']['cubeB'][0][:2]))*1000 for r in records];ax.plot(np.arange(1,len(records)+1)*.05,distance,label=policy,color=c)
        ax.set(xlabel='Actual simulation time [s]',ylabel='Observed base A-B XY center distance [mm]');ax.legend(fontsize=10);ax.grid(alpha=.15)
        finish(fig,'Fig8_'+name,f'Multi-stage paired {name}: Pyramid case {seed}', 'Measured successor geometry changes after the first skill. Curves end when the actual program ends.\nRemaining C placement uses observed object poses; branches are not forced back onto the same suffix state.', ['pyramid_paired_parameters actual native trajectory JSON'])
    if (runs/'pyramid_release_control/RESULTS.json').exists():
        release=load(runs/'pyramid_release_control/RESULTS.json');pol=['released_nominal','released_dx_minus','released_dx_plus'];counts=[sum(r['success'] for r in release['rows'] if r['policy']==label) for label in pol]
        fig,ax=chart();xx=np.arange(3);ax.bar(xx-.18,np.array(success)/16*100,width=.36,label='official release order',color='#346c86');ax.bar(xx+.18,np.array(counts)/16*100,width=.36,label='open immediately after A goal',color='#d2824b');ax.set_xticks(xx,['nominal','first X -12mm','first X +12mm']);ax.set(ylabel='Native full pyramid success [%]',ylim=(0,110));ax.legend(fontsize=10)
        for i,(x,y) in enumerate(zip(success,counts)):
            ax.text(i-.18,x/16*100+2,f'{x}/16',ha='center',fontsize=10);ax.text(i+.18,y/16*100+2,f'{y}/16',ha='center',fontsize=10)
        finish(fig,'Fig9_release_control','Strong-control explanation: test release timing before crediting learning','Same16 DESIGN conditions, same initial hashes, native250-step cap; no extra RL or sealed-test selection.\nOnly a native gripper-open command is added after the first placement goal.', ['PYRAMID_RELEASE_CASES.csv','PYRAMID_HANDOFF_DRIFT.csv','PYRAMID_RELEASE_HYPOTHESIS_PRE.json'])
    # Annotate all non-Stack raw media before publishing; no site imagery is included.
    for folder in ['peg_video_round1','peg_video_round2','peg_feedback_r3','pyramid_native_planner','pyramid_paired_parameters','pyramid_video_rescue','pyramid_video_regression','pyramid_release_control']:
        for file in sorted((runs/folder).glob('*.mp4')):
            target=media/(folder+'_'+file.name)
            raw_json=file.with_suffix('.json');episode=load(raw_json);summary=episode['summary'];summary.setdefault('task','PegInsertionSide-v1' if folder.startswith('peg') else 'StackPyramid-v1');case=summary['seed'];policy=summary.get('policy',summary.get('variant'));initial_sha=summary['initial_sha256']
            title=f'SAPIEN PhysX | {summary["task"]} | case {case} | {policy} | success={int(summary["success"])}'
            if folder.startswith('pyramid_video_'):
                reference=load(runs/'pyramid_paired_parameters/RESULTS.json');original=next(r for r in reference['rows'] if r['seed']==case and r['policy']==policy)
                assert initial_sha==original['initial_sha256'] and summary['success']==original['success']
            vf="drawbox=x=0:y=0:w=iw:h=35:color=white:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='"+title+"':fontcolor=black:fontsize=12:x=6:y=8"
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(file),'-vf',vf,'-an','-c:v','libx264','-crf','20',str(target)],check=True)
            video_receipts.append(dict(path='media/'+target.name,source=str(file),source_sha256=sha(file),sha256=sha(target),kind='ACTUAL_SAPIEN_EPISODE_WITH_DISPLAY_LABEL',physical_robot=False,task=summary['task'],case=case,policy=policy,initial_sha256=initial_sha,trajectory_source=str(raw_json),trajectory_sha256=sha(raw_json),native_terminal_success=summary['success']))
    comparisons=[('stack_rescue_200003','nominal_case200003.mp4','parameter_ppo_43_case200003.mp4'),('stack_regression_200006','nominal_case200006.mp4','parameter_ppo_43_case200006.mp4'),('selector_regression_200020','nominal_case200020.mp4','no_abstain_case200020.mp4'),('pyramid_release_control_122000','pyramid_paired_parameters_000_tracked_nominal.mp4','pyramid_release_control_000_released_nominal.mp4'),('pyramid_rescue_122007','pyramid_video_rescue_000_tracked_nominal.mp4','pyramid_video_rescue_000_first_dx_minus.mp4'),('pyramid_regression_122005','pyramid_video_regression_000_tracked_nominal.mp4','pyramid_video_regression_000_first_dx_minus.mp4')]
    for name,left,right in comparisons:
        if not (media/left).exists() or not (media/right).exists():continue
        target=media/('PAIRED_'+name+'.mp4')
        # hstack repeats the final frame of a shorter program; the header discloses this.
        filt="[0:v]setpts=PTS-STARTPTS[l];[1:v]setpts=PTS-STARTPTS[r];[l][r]hstack=inputs=2:shortest=0,pad=iw:ih+36:0:36:white,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Same initial state - native 20Hz timing - shorter stream holds final frame':fontsize=18:fontcolor=black:x=8:y=8[v]"
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(media/left),'-i',str(media/right),'-filter_complex',filt,'-map','[v]','-an','-c:v','libx264','-crf','20',str(target)],check=True)
        video_receipts.append(dict(path='media/'+target.name,source=[left,right],sha256=sha(target),kind='SYNCHRONIZED_SAME_CASE_ACTUAL_EPISODES',timing='20Hz native control; shorter completed program final frame held, not extra physics'))
    # Store exact plot sources and playback metadata, not fabricated image-generation assets.
    (out/'FIGURE_MANIFEST.json').write_text(json.dumps(dict(figures=entries,videos=video_receipts,renderer='Matplotlib / actual SAPIEN render frames; no generated robot imagery',source_kind='NEW_PHYSICAL_SIMULATION',sealed_test_opened=False),indent=2)+'\n')
    html='<!doctype html><meta charset="utf-8"><title>ARES方法演进证据</title><style>body{font:17px/1.7 system-ui;max-width:1400px;margin:35px auto;color:#203b51}img{width:100%}video{width:49%;max-height:550px}a{color:#237986}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>ARES甲方SCI：方法演进真实证据</h1><p>本轮没有独立算法增量。普通Residual PPO仍是强基线；差分弃权选择未提高当前开发集成功率。封存测试未打开。所有视频为真实SAPIEN/PhysX运行，不是实机或状态动画。</p><p><a href="CLIENT_METHOD_UPDATE_CN.pdf">中文方案PDF</a> · <a href="PAPER_METHOD_EVOLUTION.pdf">英文论文草稿</a> · <a href="RESULTS.json">指标</a> · <a href="EXPERIMENT_RESULTS.csv">主表CSV</a> · <a href="CASE_PAIRED_CONFUSION.csv">逐案例配对</a> · <a href="REPRODUCE.md">复现</a></p>'
    for entry in entries:html+=f'<h2>{Path(entry["file"]).stem}</h2><a href="{entry["file"]}"><img loading="lazy" src="{entry["file"]}"></a>'
    html+='<h2>实际模拟视频：完整单条素材</h2>'
    for entry in video_receipts:html+=f'<p>{entry["path"]}</p><video controls preload="none" src="{entry["path"]}"></video>'
    (out/'index.html').write_text(html)
    print(json.dumps(dict(figures=len(entries),videos=len(video_receipts))))

def pdf(args):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    import html,itertools
    def mixed(s):
        return ''.join(('<font name="Latin">'+html.escape(''.join(chars))+'</font>') if latin else html.escape(''.join(chars)) for latin,chars in itertools.groupby(s,lambda x:ord(x)<0x2e80))
    pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    out=args.output;result=load(out/'RESULTS.json');pdfmetrics.registerFont(TTFont('CN','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'));W,H=A4;c=canvas.Canvas(str(out/'CLIENT_METHOD_UPDATE_CN.pdf'),pagesize=A4);c.setTitle('ARES甲方SCI方法演进：真实实验与负结果')
    style=ParagraphStyle('body',fontName='CN',fontSize=10,leading=16,wordWrap='CJK',textColor=HexColor('#203b51'))
    def text(s,y,size=10):
        st=ParagraphStyle('x',parent=style,fontSize=size,leading=size*1.6);p=Paragraph(mixed(s),st);_,height=p.wrap(W-90,H);p.drawOn(c,45,y-height);return y-height-10
    def head(title,page):
        c.setFillColor(HexColor('#203b51'));c.setFont('CN',19);c.drawString(45,H-52,title);foot=Paragraph(mixed(f'ARES研究工作稿 | 2026-10-28 | SAPIEN/PhysX模拟，不是实机 | 第{page}页'),ParagraphStyle('foot',parent=style,fontSize=8,leading=10));foot.wrap(W-90,30);foot.drawOn(c,45,25)
    head('一页结论：物理权限成立，独立算法收益未成立',1);y=H-83
    for paragraph in [
        '研究问题：哪些合法技能参数能改变后续物理结果，如何利用当前状态与真实执行反馈决定何时、修正多少，并减少普通残差学习的退化？',
        f'本轮实做：6组普通PPO模型，3个训练种子×2种动作接口，共614,400次在线控制交互；512训练条件×3条冷重置后缀共76,800次交互；另运行同76,800成本的CEM。所有实际试验与评价合计{result["total_control_interactions"]:,}次控制交互，计数范围见成本CSV。',
        '新64例开发评价：名义控制60/64，固定高度与CEM 62/64；普通Residual PPO为63、62、64/64，参数化PPO为62、63、60/64。参数化接口并未稳定优于普通残差。三种子共用案例，不能视为192个独立场景。',
        '差分模型结果：保守弃权60/64，去掉弃权62/64，去掉执行效果60/64、去掉后续目标59/64、同容量绝对结果模型60/64。单一预测/门控构件没有独立增量，不能包装成SCI新算法。',
        '物理新证据：Pyramid原生250步任务中，官方规划器在4例完成4例。16例同初态参数对照为12/16、12/16、7/16；−12mm包含1次救回与1次退化，+12mm造成5次退化。第一层动作确实改变后续任务，而不是只改变TCP图。',
        '负结果与转向：Peg三轮原生100步试验仍仅1/8，接触/跟踪基线偏弱；手写Pyramid控制0/16，不能拿它当论文弱基线。下一研究对象已实测转为官方Pyramid技能后的状态/收益选择，尚未宣称其RL增益。',
        '决策：不再扩大原R3似然、不扩五种子、不打开封存测试。本轮state-based观察含仿真状态真值，非privileged感知、多模态迁移、JAKA硬件成功均未验证。ARES-R保持只读，未使用受限现场资料。']:
        y=text(paragraph,y)
    c.showPage();head('控制权限：从请求到真实任务后果',2)
    c.drawImage(str(out/'figures/Fig8a_pyramid_authority.png'),42,390,width=W-84,height=315,preserveAspectRatio=True)
    y=text('第一层参数对后续整任务具有双向效应；相同总成功率不能说明相同失败机制。小样本仅用于物理设计，不是新算法验证集。',370)
    c.drawImage(str(out/'figures/Fig8_rescue.png'),42,67,width=W-84,height=265,preserveAspectRatio=True)
    c.showPage();head('真实学习曲线与配对退化',3)
    c.drawImage(str(out/'figures/Fig5_residual.png'),42,420,width=W-84,height=300,preserveAspectRatio=True)
    c.drawImage(str(out/'figures/Fig6a_native_results.png'),42,96,width=W-84,height=300,preserveAspectRatio=True)
    text('收益须超过强固定控制、同容量绝对价值与普通PPO；本轮尚未达到。图表不把开发集称为未见封存测试。',82,size=9)
    c.showPage();head('同案例实际视频帧：救回与退化都保留',4)
    c.drawImage(str(out/'figures/Fig3a_stack_rescue.png'),42,390,width=W-84,height=315,preserveAspectRatio=True)
    c.drawImage(str(out/'figures/Fig3b_stack_regression.png'),42,65,width=W-84,height=315,preserveAspectRatio=True)
    text('帧来自新的原生物理运行，与完整评价的初态哈希和终态逐项一致；不是状态动画，也不是实机。',H-88,size=9)
    c.showPage();head('可复现性、成本与下一轮收缩条件',5);y=H-88
    for paragraph in [
        '代码与协议：独立分支codex/ares-sci-method-evolution-20261028；pinned ManiSkill 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3。所有训练/验证scene seed池和SHA在BENCHMARK_PROTOCOL.json；封存测试没有执行。',
        '模型：6组最终PPO、初始权重、4种选择器、真实轨迹保存在非Git运行目录。MODEL_INDEX.json列出SHA与loader；ACTOR_UPDATE_AUDIT.json证明Actor真实更新且已参与新物理评价。公开仓库仅含允许的模拟素材、源码、标量数据、哈希和复现脚本。',
        '公平性：本轮模型输入为显式state-based条件；差分与绝对模型使用相同容量、相同真实后缀。配对采集3条分支成本都单独计入；CEM以同等成本在训练条件选固定参数，不使用测试案例oracle。部分独立任务共用GPU0，壁钟时间不用于算法效率优劣判断。',
        '科学边界：新参数接口不是新PPO损失；M1为一次边界的contextual bandit，不是长程SMDP。ensemble标准差是经验不确定性，不是在线安全保证。只保留可反证机制，不因失败而放宽物理、horizon、reward或成功谓词。',
        '下一轮只做：基于Pyramid已有双向可控证据，将第一层实际执行后果与下一技能状态绑定，比较同权限规则、普通参数PPO、绝对价值、配对差分与搜索。先确认有独立收益，再扩5种子与完全封存测试；Peg先完成可靠抓持/姿态/入口跟踪基线，JAKA暂不进入成功迁移主表。']:
        y=text(paragraph,y)
    c.save();print('CLIENT_METHOD_UPDATE_CN.pdf generated')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--pdf-only',action='store_true');a=p.parse_args();pdf(a) if a.pdf_only else figures(a)
