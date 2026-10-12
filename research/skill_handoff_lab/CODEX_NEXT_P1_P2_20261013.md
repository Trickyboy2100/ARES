# Codex 下一轮单独工作单（P1+P2，只做 CPU 可复现基座）

**日期 2026-10-13。只读 ARES-R。不得移动机器人。不得安装/卸载 NVIDIA 驱动、不得 reboot、不得改变旧模拟环境。**

## 进入工作状态

1. 在 \`~/robot-research/ARES-Skilllab/repos/ARES\` 执行 \`git status -sb\`、\`git log -3 --oneline\`，确认本地 \`2b13fb9\` 包含先前 L0 审计与报告。除非用户许可，不做 reset/clean；确认远端正常后可正常 push 现有本地工作。
2. 在不破坏现场工作树的前提下，读取 \`research/skill_handoff_lab/RESEARCH_ROADMAP_20261013.md\`（规划分支）；同时完整复习 \`research/skill_handoff_lab/WORK_ORDER_CODEX_3080.md\`、\`SITE_20261012_REFERENCE.md\` 和 ARES-R \`transfer_chain.py\`。
3. 所有训练/仿真新文件在 ARES 的 \`research/skill_handoff_lab/bench\`；\`$LAB/runs\` 只放生成数据/图/缓存。
4. 今天优先交付真实数据流，**不尝试 PPO、不启动 Isaac、不等待 GPU 驱动修复完成**。

## 任务 1：程序接口（Python 3.13 NumPy/SciPy 既有环境）

实施：\`bench/{__init__,contracts,scenario,kinematics,candidates,successor,env}.py\` 及测试。

强类型合同：

- Scenario：\`task_family\`、scene/task/robot/tool revisions、BODY frames、pick/place 独立静态世界、任务剩余目标、当前 q、attachment 和 base delta。
- Candidate：前一个 skill 的真正终止状态（q/pose/branch），下个 skill 的 initiation 条件，动作合法 mask，不能仅有 TCP xyz。
- Result：明确 \`GEOMETRIC_FEASIBILITY_ONLY\` 与 \`PLANNER_VALIDATED\`、\`PHYSICS_VALIDATED\` 的分层；error/timeout/oom 标 \`UNKNOWN\`。
- \`reset(seed)\` 返回同一场景、同一候选 ID；\`step\` 只做 L0 的真实计算或读取有 provenance 的 oracle 缓存，不生成随机成功标签。

注意 ARES-R \`transfer_chain.py\` 的手写 cost/DP 是 baseline，不要删、不要训练标签使用同一 cost。复制/导入只读纯函数而非改变 ARES-R 代码。

## 任务 2：先构建能产生真实动作端点的场景

T1：料盘→凹槽；T2：料盘→烘干箱入口/70mm preland（柜体 object=4；槽位 object=3）。场景几何来自 ARES CAD/URDF 或清楚标记为工位任务参数化几何；不能假称来自现场已观测点云。不可达环境不算 skill handoff 失败对照。

每场景生成不少于 4 个候选，如 two-finger symmetry × center yaw/IK branch，明确每个候选的接触/非接触语义，维持 \`center\` 为 Cartesian target。约束链：

\`\`\`text
candidate predecessor skill terminal reachable?
   → actual joint and TCP from pinned FK/IK?
   → soft joint limits, level tolerance and attached geometry?
   → successor initialization predicate?
   → only then ready for GPU motion planning
\`\`\`

不要简化成两个手工 bool 或随机选 true/false。

## 任务 3：6×4 初始探路

- 6 个独立 seed、至少两个任务族，每个 4 候选；全部保存 scenario hash、candidate ID、goal FK/IK、是否满足下一个 skill 的**真实启动 predicate**。
- 当前 CPU 不具备 cuRobo GPU 规划时，报告标签只能是 \`IK_VALID\`, \`SOFT_LIMIT_VALID\`, \`GEOMETRY_APPROX_VALID\`，\`CUROBO_ORACLE=NOT_TESTED\`。**不要把 CPU endpoint success 叫做 L1 verified planning**。
- 报表列：\`scene,source,task,candidate,terminal_q,terminal_pose,next_skill_precondition,geometry_status,solver_failure,known_unknown\`。
- 同场景候选的 target、obstacles 不随 action 修改；将终态差异通过真正 IK/FK/约束变化体现。
- 完成后输出最少 1 张两任务族的候选/初始化集合比较图、至少 1 张 failure attribution 图（可以 CPU plot，不用仿真 GUI）。
- 单测包括：seed determinism、asset revision mismatch、scene stale、wrong frame、soft limit、unverified effect、same scene pairing、correct split grouping。

## 任务 4：旁路 GPU 驱动诊断

另行写只读 \`GPU_DIAGNOSIS.md\`，执行：

\`\`\`bash
uname -r
cat /proc/driver/nvidia/version
modinfo nvidia | head -n 20
dkms status
dpkg-query -W 'nvidia-*' 'libnvidia-*' 2>/dev/null | head -n 80
ldconfig -p | grep -E 'libnvidia-ml.so|libcuda.so' | head -n 20
df -h
free -h
swapon --show
\`\`\`

留意 Debian/Ubuntu dpkg 的工具已经可读且不需要 root 安装。不卸载，不重启，不临时 reload kernel modules。仅出 2 个或更少最小可行修复候选，说明风险/需人工确认事项。

## 任务 5：交付与下轮决定

文件：
- \`research/skill_handoff_lab/bench/*.py\`
- \`research/skill_handoff_lab/tests/test_bench_*.py\`
- \`research/skill_handoff_lab/reports/P1_P2_CPU_SMOKE.md\`
- \`$LAB/runs/smoke1_cpu/{scenarios,candidates,summary}.jsonl\`
- \`$LAB/runs/smoke1_cpu/figures/*.png\`
- \`research/skill_handoff_lab/reports/GPU_DIAGNOSIS.md\`

报告必须给出以下真实状态（PASS/FAIL/NOT_TESTED）：

\`\`\`text
DATA_CONTRACT
SCENARIO_GENERATOR_TWO_TASKS
REPRODUCIBLE_6x4
ASSET_BINDING
IK_AND_NATIVE_SOFT_LIMITS
SUCCESSOR_START_CONTRACT
CPU_GEOMETRY_APPROX
CUROBO_ORACLE
CANDIDATE_SENSITIVITY
HEURISTIC_HEADROOM
READY_FOR_LEARNING
\`\`\`

最后输出 **下次最少增量**，预估 cuRobo 解算次数/耗时。若 L0 smoke 没有候选差异，直接指出，不把其解释为 RL 失败或成功。只有 P2 合格后才启动针对 P3 的 GPU oracle 调研。
