# 新 v7 实验材料索引

本目录仅包含**自建合成 P2 接触任务**的脱敏派生表与哈希索引。没有现场点云、配置、实机读数或现场派生图。正式模型和完整原始状态保存在非 Git 研究目录，不把大文件复制进入公开仓库。

- 本地与190统一研究根：`/home/andyee/robot-research/ARES-Skilllab/runs/client_sci_pilot_20261027/`。
- 正式有界 pilot：`formal/`。只消费同时具有 `RUN_COMPLETE.json`、manifest、完整 trace 的子目录。
- `NEW_PILOT_RESULTS.csv`：每次已完成运行的原始/独立成功数、固定训练预算、源码及验收哈希。`split=train` 行仅为训练结束的8例检查，不能当成32例新验证集结果。
- `NEW_RUNS_INDEX.json`：模型、精确运行源码、场景、原始 episode/action/trace/验收的 SHA256、大小和实际路径。标明模型是否真正改变参数、优化次数和在线交互。
- `ONLINE_TRAINING_EPISODES.csv`：真实训练 episode；奖励/成功来自物理执行，不是人工标签。
- `HANDOFF_PHYSICAL_STATES.csv`：真实阶段交接时的位置、姿态、速度、角速度和推板接触相对状态。
- `PHYSICAL_CAUSAL_INTERVENTIONS.{csv,json}`：同 checkpoint 输入干预及固定参数连续物理对照。相同冷启动初态，后续状态真实演化；不是恢复不完整的接触快照。

## 精确证据边界

64D Actor=原56D当前仿真状态+8D短期执行历史。B1/B2将新增历史置零，B3读取它。8维依次为任务坐标的物体有限差分速度2、实际参考命令速度2、物体/命令差2、物体角速度Z、请求的接触参数/4mm。

历史最后一维及 `contact_parameter_m` 字段指**目标命令参数**，不是物体实际位移。后方接近和防积分饱和可抑制参数的瞬时作用。实际动作证据是 `applied_xy`、真实关节状态、工件轨迹以及参数配对干预；不可仅按非零网络输出宣称物理影响。

B1/B2/B3均有相同3输出、64-64网络、约束和回报。B3不是新增预测头；普通PPO按完整物理任务回报更新Actor。局部坐标、残差PPO和历史拼接不是单独原创算法。去历史不等于去全部物理反馈：公共56维仍含当前速度/角速度/接触。去next-contract只遮蔽阶段0已知下游目标/几何，合法动作映射仍使用已知任务方向，属于部分信息干预。

B2/B3的cos/sin规范为1/0，而q/qd与固定工具/工作空间仍BODY；不得把B1/B2称为完全信息等价或完整等变控制。向量往返测试依赖外部已知n。二者共享底层状态来源和容量，但Actor表示/信息并非已证明可逆；主要B3/B2对照在同一规范化基座上成立。

独立验收是120Hz采样 OBB/FK/软限位/精确post-step状态/0.5s稳定保持，不是连续CCD、完整机械臂碰撞认证或硬件安全认证。对照参数−4/0/+4mm分别真实执行32例，3次预算不能作为免费的一次搜索。

## 源码权威与软件修正

每次实际执行的 `task_source.py`、`frame_source.py`、`boundary_credit_source.py`、`scene.json` 和 `run_config.json` 是权威。当前Git源码增加了可选测量历史干预，不能据此更改旧运行来源SHA。

最早两次16,384交互训练发生在 `formal/` 外：`B1_seed31`、`B3_seed31`。源审查发现任务局部观测把cos/sin写入19/20，误覆盖dwell20；应为18/19。这不修改物理任务，但会污染归纳偏置比较。两次训练及各自validation均完整保存；额外**32,768交互**计入调试成本，不能混入正式主表。只停止两个自有队列父进程，已启动子任务正常完成，无孤儿任务继续运行。

修正后建立新的 `formal/engine/`，通过观测布局/相位/dwell/旋转往返测试，重新配对训练。没有根据测试分数改参数，没有覆盖旧源文件/结果。最终测试配置的SHA在训练之前已冻结，未用于方法选择。

复现队列：

```bash
python research/skill_handoff_lab/tools/run_client_v7_queue.py \
  --root /home/andyee/robot-research/ARES-Skilllab --gpu 0
# GPU1同理；正式目录必须尚不存在，不能覆盖既有证据。
```

请先恢复对应精确引擎/场景快照和60MB自建模型资产；`batch_gpu0.json`、`batch_gpu1.json` 保留每个实际命令与独立超时/完成判据。推理干预不增加训练交互，但需计入额外物理评估预算。
