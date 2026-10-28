"""One evidence-synchronized manuscript, client decision and public provenance."""
import argparse,csv,gzip,hashlib,json,shutil,re
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root,out=a.root,a.output;base=out.parent;metrics=json.loads((out/'STATISTICAL_SUMMARY.json').read_text());op=json.loads((out/'OPPORTUNITY_SUMMARY.json').read_text());ledger=json.loads((out/'COST_LEDGER.json').read_text());stats={x['method']:x for x in metrics['methods']};groups=metrics['seed_groups'];cmp=metrics['candidate_minus_ordinary'];total=ledger['total_external_control_interactions'];ordinary=groups[0]['success_counts'];candidate=groups[1]['success_counts'];ci=cmp['reset_batch_and_seed_bootstrap_ci_pp'];robust=cmp['mean_difference_pp']>0 and ci[0]>0 and min(cmp['seed_differences'])>0
 decision='PROMISING PILOT; additional independent validation required' if robust else 'NO RELIABLE INDEPENDENT METHOD INCREMENT; not ready for SCI method submission'
 # Preserve original current docs before replacing their entry point. Originals
 # remain local history; dated evidence directories are not overwritten.
 history=root/'preserved_current_documents';history.mkdir(exist_ok=True)
 for f in [base/'01_CLIENT_PROPOSAL_CN.md',base/'01_CLIENT_PROPOSAL_CN.pdf',base/'02_PAPER_SEED'/'main.tex',base/'02_PAPER_SEED'/'main.pdf',base/'02_PAPER_SEED'/'references.bib']:
  if f.exists():
   target=history/(f.parent.name+'_'+f.name)
   if not target.exists():shutil.copyfile(f,target)
 cn=f'''# 面向多阶段机器人操作的任务条件化技能边界参数强化学习

2026-10-28｜唯一论文主线；全部新数字来自公开Panda / 原生StackPyramid-v1物理仿真，不是JAKA或现场结果。

## 当前结论

已实现“PickA → PlaceA/实际松爪/退让/稳定验收 → 重新观测决策 → PickC → PlaceC/实际松爪/验收”的连续物理执行器，并训练两个真实边界上的参数RL。新64个训练设计场景中，{len(op['current_verified_downstream_different_cases'])}例在前段均合法完成时，仅A参数不同就改变后续结果；另有{len(op['identical_posterior_C_choice_different_outcome'])}组在完全相同的前段后验下，C参数改变成功与否。物理可控性已成立，算法原创性与独立增益尚须由公平结果决定。

**独立64开发场景：**普通同接口PPO三种子{ordinary}；风险约束候选{candidate}；候选相对普通PPO平均{cmp['mean_difference_pp']:+.2f}个百分点，按训练种子与16环境重置批次交叉bootstrap95%区间[{ci[0]:.2f},{ci[1]:.2f}]个百分点。区间是小样本描述性不确定性，不是显著性保证。名义{stats['nominal']['native_success']}/64，几何规则{stats['rule']['native_success']}/64，训练集固定搜索{stats['fixed']['native_success']}/64，绝对结果评分器{stats['absolute_q']['native_success']}/64，成对风险评分器{stats['paired_ranker']['native_success']}/64。不能选最佳种子或用已见候选并集冒充策略分数。

判断：{decision}。共{total:,}次新外部控制交互，包含工程失败、数据采集、在线RL、消融、开发评估及250步中的终态保持。在线主训练288,000、消融48,000；共同物理设计数据144,000；独立开发240,000。原生重置内部settling不计为外部控制交互。旧903,584交互属于上一轮，不能累加成当前训练量。

## 一篇论文只解决什么

局部技能合格并不保证下一个技能可接续。本文只研究：在几个有界、实际可执行的抓放参数之间，如何在松爪前选前段参数，并在真实释放稳定后利用当前物体状态选择第二段参数。不是重学关节运动，不是VLA、多模态大系统或实机迁移成果。第一决策选择A支撑放置方向；第二决策选择C真实抓取朝向。第二段的物理程序只在第一段独立验收合格后启动，没有中间reset、物体瞬移或虚拟成功标签。

## 公平方法与科学解释

所有方法共用公开Panda名义planner、真实PhysX、57维特权状态、同样三选一参数和原生250步。普通PPO与候选使用完全相同网络、初始化、训练数据和48,000在线交互/种子。候选仅加训练域成对退化风险正则；下游价值、技能参数化和Q减名义Q均是已有思想，不把这个组合先命名为原创。监督评分器与固定搜索也使用同一物理数据。三个信息/重决策消融用16,000在线交互/模型，对应比较完整策略的16,000交互检查点。

原生成功和两段程序验收完成分别报告。有程序验收完但原生最终条件失败的案例；也有原生条件成立但程序验收不完整的案例。所有试验物理运行满250步，即12.5秒原生仿真时间；提前逻辑完成不等于节省物理执行时间。规划缓存仅重用起点/物体状态核验匹配的候选关节计划，所有对象效果重新运行物理；缓存与求解时间单列，不能把后运行模型受益于缓存解释成算法加速。

## 可直接检查的证据

`11_PAPER_FIRST_20261028/VIDEOS/index.html`为统一入口；视频大文件位于非Git的runs/single_paper_20261028/visuals，打开本机OPEN_ME.html即可播放。影片以原始保存的关节、物体状态渲染；标为原始物理状态回放，不把渲染当新实验。Case420010：A两种放置都完成实际松爪和稳定，但同一C后缀执行后，一种AB距离越过原生阈值，另一种成功；Case420002：相同A后验，仅C抓取参数不同导致抓取失败/成功。

逐案例见CASEWISE_EFFECTS.csv、OPPORTUNITY_STUDY.csv；全部种子、验收/条件成功率、救回/退化及区间见DEVELOPMENT_RESULTS.csv与STATISTICAL_SUMMARY.json；论文表格在LATEX。模型、训练域结构化数据、源码及场景哈希均可复现。未公开任何受限现场素材，ARES-R保持只读。

## 下一步与可投稿差距

先保留这个单一科学对象，不再扩已饱和StackCube或弱Peg。若候选收益不稳，应把本轮视为“真实物理参数机会 + 强同接口对照”的实证研究，而不能宣称RL新方法成立。后续应针对失败原因和风险模型校准提出一个具体、预声明的机制，证明它超越几何规则/评分器/普通PPO；取得可靠增量再扩5种子、只开一次封存测试及一个不同物理机制任务。当前没有新任务泛化、感知噪声/点云输入或真实JAKA验证；完整状态下历史差分可冗余，不能强行宣传执行历史增益。

RAPS正式题名已更正。STAP、外科ViSkill、Foresight和DAWN均按原始来源限定引用范围；详细核对见REFERENCE_AUDIT.md。本文主线没有发明后续价值、参数化动作或ResidualRL。
'''
 (out/'MENTOR_DECISION_CN.md').write_text(cn);(base/'01_CLIENT_PROPOSAL_CN.md').write_text(cn)
 def short(n):return n.replace('ppo_s','s').replace('_b0_full',' PPO').replace('_b2_full',' risk').replace('_',' ')
 tex=out/'LATEX';tex.mkdir(exist_ok=True);bibs=r'''\begin{thebibliography}{9}
\bibitem{raps} M. Dalal, D. Pathak, and R. Salakhutdinov. Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives. NeurIPS34,2021. \url{https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html}.
\bibitem{stap} C. Agia, T. Migimatsu, J. Wu, and J. Bohg. STAP: Sequencing Task-Agnostic Policies. ICRA2023, pp.7951--7958. doi:10.1109/ICRA48891.2023.10160220.
\bibitem{viskill} Tao Huang, Kai Chen, Wang Wei, Jianan Li, Yonghao Long, and Qi Dou. Value-Informed Skill Chaining for Policy Learning of Long-Horizon Tasks with Surgical Robot. IROS2023. \url{https://arxiv.org/abs/2307.16503}.
\bibitem{foresight} Y. Liu, X. Zhang, L. Liu, and A. Boularias. Foresight Residual RL for Long-Horizon Robot Manipulation with Vision-Language-Action Models. arXiv:2607.16506,2026 (author-listed IROS2026 acceptance). doi:10.48550/arXiv.2607.16506.
\bibitem{dawn} G. Ma, L. Li, H. Wang, Z. Liu, P.-L. Bacon, and D. Tao. What Makes Value Learning Efficient in Residual Reinforcement Learning? ICML2026, PMLR306:84569--84601. \url{https://proceedings.mlr.press/v306/ma26s.html}.
\end{thebibliography}
'''
 paper=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,array,float}
\usepackage{hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}
\IfFileExists{../FIGURES/Fig1_scope_and_system.pdf}{\def\assetpath{../FIGURES}\def\tablepath{.}}{\def\assetpath{../11_PAPER_FIRST_20261028/FIGURES}\def\tablepath{../11_PAPER_FIRST_20261028/LATEX}}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\title{Task-Conditioned Skill-Boundary Parameter Reinforcement Learning\\for Reliable Multi-Stage Manipulation\\\large A controlled native-physics study and falsifiable method pilot}
\author{ARES Research Working Draft}\date{28 October 2026 --- not a submission-ready method claim}
\begin{document}\maketitle
\begin{abstract}
Local completion of a manipulation primitive does not guarantee that the resulting physical state permits its successor to succeed. We study executable parameter adaptation at two physically verified boundaries in native StackPyramid-v1, using a common Panda planner and actual grasp, release, support and stability readbacks. A cold-reset training-design study finds nine cases with different downstream outcomes despite successful predecessor verification, and18 matched-prefix contexts where the second parameter changes the outcome. We train matched finite-horizon PPO and a paired-regression-risk candidate with equal data, initialization and action authority across three seeds. This draft reports all outcomes, strong fixed/geometry and supervised baselines, matched-budget information ablations and physical cost. The empirical tables below determine the scope of the method claim: opportunity and engineering integration alone are not algorithmic novelty. Experiments use privileged simulator state, not vision or a physical robot.
\end{abstract}
\section{One scientific question}
We ask whether a policy can select a physically executable predecessor parameter before release, observe the actual stable released state, and adapt the successor parameter to improve whole-task reliability beyond strong nominal control and ordinary RL with identical authority. The industrial motivation is the separation of perception, Skill Contract/Runtime, parameter policy and robot control in ARES-R. Only the public Panda simulation layer is evaluated here; no JAKA, AMR, point-cloud policy or site deployment is tested.

A waypoint is not a completed skill. A valid handoff must include actual gripper release, retreat, stable objects, support contact and robot state. We therefore separate three claims: physically controllable terminal states exist; a shared executable interface is correct; and a learning mechanism improves independent outcomes. The first two do not establish the third.
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{\assetpath/Fig1_scope_and_system.pdf}\caption{Evaluated simulation components versus industrial context. Dashed components are not experimental achievements.}\end{figure}
\section{Relation to prior methods}
RAPS already learns parameters of manually specified primitives~\cite{raps}. STAP coordinates geometric dependencies using skill values~\cite{stap}; surgical ViSkill learns chaining behavior from whole-task value~\cite{viskill}. Foresight Residual RL explicitly optimizes terminal handoff quality with downstream-success predictions~\cite{foresight}. DAWN studies anchored residual value learning and normalization~\cite{dawn}; it is not reproduced by this PPO pilot. We do not claim that downstream value, parameterization, nominal-value subtraction or adding a predictor is new. Any eventual contribution must be a specific independently validated mechanism operating through physically executable contracts.
\section{Physical task and common skill contract}
We pin public ManiSkill commit62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3, StackPyramid-v1 and its original sparse reward, success predicate, physics and250-control-step horizon. Native control frequency is20Hz. A red cube A is placed adjacent to green B; blue C must then be placed on the supports. The upstream predicate is used unchanged and evaluated independently at the native horizon.

The common program is: pick A; place, open, retreat and verify A; observe and decide; pick C; place, open, retreat and verify C. No resets, object setters, teleports or cached outcomes occur between skills. The public MPLib planner generates joint paths; the servo consumes every third path sample with a bounded readback correction. Screw planning has two attempts and a0.5s RRT fallback. Symmetry-equivalent gripper orientation avoids gratuitous wrist rotations. These are shared engineering controls, not learned gains. Cube obstacles are not encoded in a certified scene-aware motion planner: actual contact and failures occur in native PhysX. This benchmark is not industrial collision certification.

Decision0 selects an A support-placement direction offset in $\{-\pi/2,0,+\pi/2\}$ relative to the initial A--B axis. The support clearance accounts for observed B orientation. Decision1 is requested only after verified release of A and selects a C grasp-yaw offset in $\{-0.35,0,+0.35\}$rad. The desired C center remains the measured A/B midpoint; actual held-object-to-TCP offset is used for retargeting. Requested controller angles are not treated as measured object rotation.

The boundary requires three consecutive control readbacks with finger opening $>65$mm, object not grasped, all three object linear/angular speeds $<0.01$ m/s and $<0.5$ rad/s, robot joint speed $<0.15$ rad/s and XY target error $<15$mm. A additionally needs table-height error $<8$mm and actual table contact. C needs target-height error $<8$mm and measured contact with A or B. Measured forces and projected footprint overlaps are recorded; projection is not contact area. Native success and verified program completion are reported separately.

All batched episodes physically run all250 control steps, including terminal holds. Logical completion is a diagnostic, not saved physical time. Candidate joint plans may be reused only when the complete measured per-environment object pose/velocity, robot q/qvel and target match to $2\times10^{-6}$ max absolute, with the same seed, label and chosen-action prefix. Physical servo and effects always run again. Cache reuse and actual planning costs are separated; no algorithm speedup is inferred from cache order.
\section{Finite-horizon two-boundary learning}
At a visited boundary, $o_k$ contains57 privileged features: proprioception, current relative object/goal geometry, object orientations, observed position changes, support-contract margins, stage and remaining budget. The actor never receives future outcome labels or case IDs. Full state may make history features redundant; we do not assume a history benefit.

The physical transition is $s_{k+1}=\operatorname{Execute}(\mathcal S_k,a_k,s_k)$ followed by independent verification. A failed predecessor terminates the program and creates no virtual second decision token. For finite native horizon $T=250$, option duration is the difference between actual decision steps, with the final option including terminal hold until $T$. We explicitly use $\gamma=1$. Whole-task native terminal success $Y$ is the shared return for each visited decision. There is no reward change favoring the candidate and no claim of physical-time optimization.

Ordinary PPO and the candidate share actor/critic64-by64 tanh networks, train-only normalization, initialization, sampled scenes, terminal-return advantages and a common supervised warm start. PPO uses clip0.2, Adam $3\times10^{-4}$, eight epochs per16-episode update, value coefficient0.5 and entropy coefficient0.02. Each main model receives12 updates,192 episodes and48,000 physical control interactions. Seeds31,37,43 are all retained.

The conventional candidate tests a frozen paired-regression estimator $h_\phi(o,a)$. Training-only cold-reset labels are $d=\mathbb1[Y_{\rm nom}=1,Y_a=0]$: at boundary0 the suffix is nominal; at boundary1 nominal is the C choice under the identical A prefix. Binary cross-entropy fits $h_\phi$. Its actor objective adds
\[
 L_{\rm cand}=L_{\rm PPO}+\beta\,\mathbb E_{o}\sum_a\pi_\theta(a\mid o)h_\phi(o,a),\quad\beta=2.
\]
Ordinary PPO uses $\beta=0$ with exactly the same data access and warm start. This heuristic trades predicted downside against return; it is not a calibrated risk guarantee or new policy-gradient theorem. Both actor and critic are trained with actual terminal native outcomes. Predicted risk is never substituted for a physical outcome.
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{\assetpath/Fig2_two_boundary_method.pdf}\caption{The second parameter is selected after actual release/stability, unlike a one-shot contextual classifier. Failed A has no successor decision.}\end{figure}
\section{Protocol and fair comparisons}
The64 new training-design scenes420000--420063 form an exhaustive3-by3 cold-reset intervention grid (144,000 control interactions). The first48 cases, in three reset batches, fit the shared absolute-outcome and paired-risk networks; the last16 cases, in a separate batch, report calibration without selecting epochs. Both networks use fixed400-epoch fits. Posthoc audit finds only9 positive regression labels among546 eligible risk-fit tokens. On the separate189-token calibration batch, risk Brier is0.0233 versus0.0207 for a constant-prevalence predictor; mean predicted risk is0.0065 versus observed prevalence0.0212. This diagnoses a sparse-label calibration failure, not a development-tuned correction. The absolute-result model has training Brier0.0104 and calibration Brier0.0779, indicating a nontrivial generalization gap rather than certainty. Online training uses new and reused cases within a separately declared420000--420511 pool; the192 actual cases per model are disclosed. Old16 Pyramid examples and old Stack64 scores are excluded.

All methods are evaluated on the same64 new development scenes520000--520063 with matching initial-state hashes, not only nominal failures. Baselines are strong nominal, a geometry rule choosing A-C and C-jaw clearance, the best fixed pair from the training grid, an absolute-result Q argmax, a paired-risk ranker, matched ordinary PPO and the candidate. Subtracting a state-dependent nominal Q alone cannot alter an argmax; the ranker additionally uses paired downside risk. Finite exhaustive search is the fixed/search baseline; no CEM reproduction is claimed.

Three exploratory ablations remove first-decision next-object geometry, replace second-decision object geometry/effects with the initial snapshot, or commit the second parameter at the initial observation. They train64 episodes/16,000 interactions, compared against the full candidate at its same-budget update4. Masked inputs can create distribution shift for the frozen teacher; these are exploratory information tests, not clean proof of a novel posterior-identification mechanism. The zero-risk baseline is ordinary PPO. No best-seed or development-selected checkpoint is reported.

Task scenes and training seeds are distinct statistical units. We report paired rescues/regressions on identical cases and descriptive bootstrap intervals both over cases and over the four16-environment reset batches, crossed with training seeds. Three seeds and four batches yield limited precision. The sealed620000--620127 scenes remain unopened pending a reliable increment.
\section{Results: physical opportunity versus method benefit}
\subsection{Interventions change actual downstream outcomes}
The nominal training-design pair succeeds52/64; the best single fixed pair59/64; the seen candidate union62/64. The union is an ex-post opportunity ceiling, not a deployable strategy score. Nine scenes have different downstream outcomes across A parameters despite verified predecessor completion. Across181 valid A-prefix contexts, the three C interventions have exactly equal saved decision1 input states (maximum absolute discrepancy0);18 contexts show different outcomes. Thus both parameter boundaries have real physical authority.
\begin{figure}[H]\centering\includegraphics[width=.75\linewidth]{\assetpath/Fig3a_physical_opportunity_grid.pdf}\caption{Actual training-design grid, not an RL result. Better training fixed parameters need not generalize.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{\assetpath/Fig3b_intervention_physical_state.pdf}\caption{Recorded native physical states in case420010. Both A releases verify; identical C parameter leads to different support separation and final native success. The first panel zooms the support-placement portion. This intervention demonstrates a total physical effect, not isolated causal mediation by margin alone.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.85\linewidth]{\assetpath/Fig3c_measured_handoff_geometry.pdf}\caption{Measured stable handoff geometry and subsequent outcomes. Projected overlap is a proxy; red/green points are not a fitted causal boundary.}\end{figure}
\subsection{All seeds and same-interface comparisons}
\begin{figure}[H]\centering\includegraphics[width=.87\linewidth]{\assetpath/Fig4_online_learning.pdf}\caption{Actual online rollouts, all three seeds; lines are not held-out scores. Shared physical design collection is additional cost.}\end{figure}
\begin{table}[H]\centering\scriptsize\input{\tablepath/development_results.tex}\caption{Same64 development cases. Native success and BOTH local boundary checks are distinct; the latter does not imply final native success. Rescue/regress compare each method to nominal on identical cases.}\end{table}
RESULT_PARAGRAPH
\begin{figure}[H]\centering\includegraphics[width=.9\linewidth]{\assetpath/Fig5a_development_all_seeds.pdf}\caption{Strong rules, search and supervised baselines are retained; all learning seeds are shown.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.9\linewidth]{\assetpath/Fig5b_paired_intervals.pdf}\caption{Descriptive paired scene-bootstrap intervals. Reset-batch and training-seed intervals are also recorded in the statistical JSON; neither treats repeated192 evaluations as192 independent task cases.}\end{figure}
\subsection{Information and physical redecision}
\begin{figure}[H]\centering\includegraphics[width=.9\linewidth]{\assetpath/Fig6_information_and_redecision_ablations.pdf}\caption{Equal16,000-online-interaction checkpoints. One-seed ablations cannot establish a robust mechanism. Full-state feedback can already contain the physical effect; deleting only effect deltas would be an invalid posterior ablation.}\end{figure}
\section{Cost, reproducibility and limitations}
COST_PARAGRAPH

Source, model and scene SHA indexes accompany the tables. Full physical raw trajectories and large videos remain in non-Git evidence storage; selected original measured trajectories and structured training data are versioned. Videos are graphics-only re-renders of saved robot/object states: no idealized motion and no new experiment outcome is inferred. The two qualitative intervention cases were selected from the training-design set, not used as independent method statistics. Isolation uses a network namespace, GPU0 only and read-only ARES-R mounts.

The study has only one native physical mechanism, privileged state, fixed cube sizes and a small online budget. It does not demonstrate visual perception, zero-shot new-task transfer, sim-to-real or industrial safety. Strong controller corrections are common to every method and are never counted as learning gains. Native fixed-horizon execution costs do not support a wall-time improvement claim. The risk estimator is a small supervised model with limited calibration, and masks may move its inputs off distribution.
\section{Research decision and publication scope}
DECISION_PARAGRAPH

The stable paper object remains executable skill-parameter adaptation across verified physical boundaries. A positive method claim would require reliable gain beyond matched ordinary RL and the best non-RL alternative, a controlled mechanism ablation, additional learning seeds, one-time sealed testing, a distinct physical task and perception robustness. Until those tests pass, this document is a coherent experimental manuscript and falsifiable research record, not a completed SCI method paper. Negative results are retained rather than replaced with an easier baseline or selected successful videos.
'''
 result=f"Ordinary same-interface PPO scores {ordinary[0]}, {ordinary[1]}, {ordinary[2]} out of64; the candidate scores {candidate[0]}, {candidate[1]}, {candidate[2]}. Candidate minus ordinary averages {cmp['mean_difference_pp']:+.2f} percentage points, with paired seed differences {cmp['seed_differences']} cases. The reset-batch-and-seed bootstrap95\\% interval is [{ci[0]:.2f},{ci[1]:.2f}] percentage points. Native nominal, geometric rule, fixed search, absolute Q and paired ranker score {stats['nominal']['native_success']}, {stats['rule']['native_success']}, {stats['fixed']['native_success']}, {stats['absolute_q']['native_success']} and {stats['paired_ranker']['native_success']} respectively. These are development results, not sealed-test evidence."
 cost=f"This round records {total:,} external control interactions:144,000 design collection,288,000 main online training,48,000 ablation training,240,000 development evaluation, plus {ledger['engineering_interactions']:,} engineering probes/failures. Internal reset settling is excluded from the external-control counter. Every evaluation episode physically executes250 steps, including terminal padding. No2x or3x algorithm speedup is inferred from batched planning or cache order. Per-batch wall time, planner solves/cache hits and logical completion remain separately inspectable in the cost ledger and raw summaries."
 decisiontext=("The candidate is promising at pilot scale but remains unvalidated as a new method. Additional independent tests are required before a publication claim." if robust else "The tested risk regularizer does not establish a reliable independent method increment. This is a negative result for this concrete mechanism, not proof that every skill-boundary policy is ineffective. The actual physical opportunity and functioning two-boundary learning interface remain valid assets. We therefore retain the manuscript as a controlled physical study and do not expand to saturated StackCube, weak Peg, five seeds or sealed testing simply to obtain a positive score.")
 paper=paper.replace('RESULT_PARAGRAPH',result).replace('COST_PARAGRAPH',cost).replace('DECISION_PARAGRAPH',decisiontext)+bibs+'\\end{document}\n';paper='\n'.join(re.sub(r'\b([A-Za-z]{2,})(?=\d)',r'\1 ',line).replace('out of64','out of 64').replace('all250','all 250').replace('a0.5s','a 0.5 s').replace('is20Hz','is 20 Hz').replace('64-by 64','64-by-64').replace('64-by64','64-by-64') if line and line[0].isalpha() else line for line in paper.split('\n'));(tex/'PAPER_SKILL_HANDOFF.tex').write_text(paper)
 current=base/'02_PAPER_SEED';(current/'main.tex').write_text('''% Current paper entry point. Historical manuscript preserved in non-Git evidence.\n\\input{../11_PAPER_FIRST_20261028/LATEX/PAPER_SKILL_HANDOFF.tex}\n''')
 (current/'README.md').write_text('''# Current single-paper manuscript\n\nCompile11_PAPER_FIRST_20261028/LATEX/PAPER_SKILL_HANDOFF.tex from its own LATEX directory; main.pdf is the synchronized current manuscript. The wrapper main.tex can also be compiled from02_PAPER_SEED; figure/table roots are resolved explicitly. Historical source/PDFs are preserved under runs/single_paper_20261028/preserved_current_documents. No old results have been relabeled as new.\n''')
 (out/'RESEARCH_DECISION.json').write_text(json.dumps(dict(decision=decision,robust_pilot_increment=bool(robust),candidate_minus_ordinary=cmp,sealed_test='UNOPENED',second_task='NOT_STARTED_PENDING_INDEPENDENT_INCREMENT',submission_ready=False),indent=2)+'\n')
 print(decision)
if __name__=='__main__':main()
