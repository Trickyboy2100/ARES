# 文献—机制—源码对应（覆盖状态如实记录）

不整体移植旧仿真环境；下表不是已完成全部论文复现的声明。

|工作/官方代码|已有贡献与可借鉴机制|当前阅读与重叠/待验证差异|
|---|---|---|
|[RAPS](https://github.com/mihdalal/raps)|参数化机器人原子动作降低动作空间复杂性|README与动作接口机制初读；完整论文/关键源码精读待补。合法Prepare参数属于重叠，不能称首次提出|
|[STAP](https://github.com/agiachris/STAP)|技能价值与动力学联合规划，联合成功概率的价值传播|已读官方stap/planners/cem.py与utils.py：iCEM及联合价值计算。当前工业几何/预算差异尚待实验，不默认优于充分搜索|
|[ViSkill](https://github.com/med-air/ViSkill)|后续价值指导技能衔接|README初读，核心训练源码精读待补。后续条件化不是本研究首创；物理有效准备动作与真实Skill Contract差异待验证|
|[Sequential Dexterity](https://github.com/sequential-dexterity/SeqDex)|transition feasibility与技能切换|README初读；T-value训练实现待准确定位与精读。工业Option终止/持续时间机制尚未实现正式训练|
|[RecoveryChaining](https://github.com/shivamvats/recovery-chaining)|已有名义技能与学习恢复组合|已读learn_recoveries.py、recovery_skill.py及[论文方法](https://arxiv.org/html/2410.13979v2)。真实名义后缀评估、连续恢复与回交名义技能；当前尚无合格恢复状态，不能声称已移植/改进|
|[AAWR](https://github.com/penn-pal-lab/aawr)|主动感知及非对称学习|已读src/algorithm/asym_awr.py的Q/V与优势加权actor；论文全文精读待补。只有真实观测不确定性成立才采用，不简单隐藏真值|
|[Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0)|DirectRLEnv、InteractiveScene与并行物理|已实际部署固定源码SHA，使用克隆/复制物理/独立重置；阅读官方cartpole direct环境与Camera教程。当前真实1→4→16 PPO smoke，但未完成双臂底盘工业环境|

下一轮优先补RecoveryChaining/STAP可执行机制与所选主线必要论文，不无差别移植七套环境。SkiMo/SAC/MPC均未作为默认实现。
