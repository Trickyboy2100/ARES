# 本轮新实验复现入口

这是ARES自建接触任务，不是ManiSkill/CBF标准benchmark成绩，也不是10月现场任务。原结果与资产不覆盖。ARES-R仅用于读取固定模型/FK，生产代码不修改。

## 文件与运行顺序

源码：`research/skill_handoff_lab/tools/`。

1. `build_contact_composition_asset.py --asset <既有asset_mesh_colliders_v5> --urdf <固定jaka_minicobo_gripper.urdf> --output <新目录>`：只在新资产中将历史研究焊接托盘换成固定研究推板；保留机械臂/夹爪碰撞，记录来源与SHA。
2. 使用已部署的Isaac Sim 5.1 Python，并设置Isaac Lab 2.3.0对应路径。将本目录加入模块路径，运行`run_contact_composition.py --asset ... --output <不存在的新目录> --num-envs 16 --gpu 0 --steps 32768 --seed 31 --controller ppo --credit-mode standard --gamma 1 --rollout-steps 512 --eval-episodes 64 --skip-capture`。这是单应用16物理环境，不是16个应用。
3. 普通PPO与boundary模式分别运行31/41/51；all模式先运行31做机制消融。`run_contact_study_batch.py`固定任务列表及单进程超时，只停止自己的进程组。正式结果禁止覆盖同名输出。
4. `run_contact_heldout_batch.py`对全部已训练模型及四种非学习控制器评估两个冻结参数组；部署路径在脚本中显式可见。`run_contact_causal_batch.py`只在开发集对预先指定的seed31做阶段动作替换和置零，不挑选最好种子、不重新训练。
5. 独立CPU验收：用包含python-fcl、SciPy和ARES-R只读FK的环境运行`validate_contact_composition.py --run <结果目录> --asset ... --urdf ...`。不能用没有FCL的训练Python误以为验收完成。
6. `summarize_contact_study.py --root <本轮结果根目录>`只读实际存在结果，生成图表、结果JSON与每文件SHA清单。缺失结果不填0。

## 固定协议

控制60Hz、PhysX120Hz、episode8秒；MLP[64,64]，PPO学习率3e-4，6epochs，batch256，rollout512×16，gamma1，entropy0.005。两个任务交替分配16环境。训练物体XY扰动±6mm，开发/测试±10mm；训练case由各seed产生，开发case seed1731，冻结测试2703。开发例已参与任务调试，不能再命名为最终测试。

冻结测试G1：槽宽50mm、相对工具起点目标X115mm、槽Y−30mm；G2：58mm、105mm、−20mm。工件40mm立方体、质量50g、静/动摩擦0.4/0.3。不是实测金属料盘。当前策略没有显式槽宽输入，测试仅表明有限未见几何鲁棒性，不是合同条件化或跨物体泛化。

原v4的16384步40/64、65536步普通PPO16/64和残差PPO15/64是不同优化配置的探索结果，全部保留，不能与正式v5当作等预算对照。v1–v3失败及控制器windup修正也保留。

## 日志、模型与证据

受控研究工作区的`runs/skill_composition_20261016_r0/`保存每次运行的manifest、精确task_source快照、run_config、training_episodes、ppo/progress.csv、credit_audits、frozen_evaluation、actual_physics_trace、policy.zip及实际视频/帧状态。新版额外保存credit源码。生成的ARTIFACT_MANIFEST逐文件记录SHA，REFERENCE_SOURCE_INDEX冻结公开参考代码commit和文件哈希。

视频为实际PhysX训练/执行采样，不能把低帧率录像当全频控制日志。v4探索视频为等时帧展示；v5按物理步差设置帧时长。终止前采样与终止判定可能差一个物理步，原始成功与独立保守确认分别报告。当前验证不包含完整机械臂网格连续碰撞、实机动态安全、接触抓取或现场泛化。
