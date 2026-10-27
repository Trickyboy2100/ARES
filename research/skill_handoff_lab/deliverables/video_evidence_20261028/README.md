# 两段选用视频：完整说明、原始记录与复现入口

本目录对应研究者在 2026-10-28 选定的两个 MP4。所有数据来自 ARES 自建接触任务的真实 Isaac PhysX 执行。它们不是公开 benchmark 成绩或实机实验；不包含受限现场点云、生产配置或现场状态文件。

基础代码提交：`e94ace76ea7151edf99950afce7384a0cf09b4d1`。本目录随后的提交补齐原始记录、研究资产、模型和便携复现入口。

| 需要的内容 | 本目录中的位置 |
|---|---|
| 紧凑 16 环境最终视频与说明 | [R3/video](R3_online_training_12288/video/) · [逐项说明](R3_online_training_12288/README.md) |
| 紧凑版原始物理状态、动作、训练日志、模型、源码快照 | [R3/run](R3_online_training_12288/run/) |
| R0 左右对比最终视频、轨迹图、CSV 与配对来源 | [R0/comparison](R0_case_001/comparison/) · [逐项说明](R0_case_001/README.md) |
| R0 左侧反馈控制原始记录与相机流 | [R0_baseline](R0_case_001/R0_baseline/) |
| R0 右侧 Residual PPO 原始记录与相机流 | [R0_residual41](R0_case_001/R0_residual41/) |
| R0 右侧原始训练、模型及学习日志 | [original_training_seed41](R0_case_001/original_training_seed41/) |
| 实际机器人研究 USD、URDF、运行所需资产元数据 | [assets](assets/) |
| 原始训练/测试场景 | [configs](configs/) |
| 相机采集、逐帧合成、独立验收代码 | [code](code/) |
| 完整运行步骤 | [REPRODUCING.md](REPRODUCING.md) |
| 所有归档数据哈希及解压后原始哈希 | [ARTIFACT_SHA256.json](ARTIFACT_SHA256.json) |
| 软件版本与运行说明 | [ENVIRONMENT.json](ENVIRONMENT.json) |

大 JSON、USD 和逐帧记录采用 gzip 无损压缩。`ARTIFACT_SHA256.json` 同时保存压缩文件哈希和原始字节哈希；`.npz`、`.zip`、MP4 原样保存。完整动作记录没有用 CSV 降采样替代。视频中的接触力是 PhysX net-force proxy；独立验收采用采样 OBB/状态检查，不能解释为全机械臂连续碰撞或工业安全认证。

R0 保留两侧全部 32 个冻结案例的数值记录及原始训练结果，供检查案例选择与失败；仅选中 case001 的原始相机流和关键帧随本包发布。R3 保存本次紧凑复录的完整数值与相机记录。没有通过重复相同案例增加独立样本量。

`code/vendor/ares_r/perception/robot_collision.py` 是用于离线 FK 验收的只读代码快照，来自 ARES-R `2619cb4bb6156d9146820d2f1e28df6fa81a0c2a`，未修改 ARES-R 工作区。USD 来自 ARES 既有机器人网格和研究推板；保留上游资产条款，不对第三方模型另行授予许可。NVIDIA 运行时及上游 Isaac Lab 请按各自许可安装，不包含在数据包内。

原始 JSON/源码快照中的绝对路径保留用于追溯；复现脚本生成单独的路径重定位配置，不要求具备 `/home/andyee` 目录。资产清单只保留仿真运行必需字段，去掉不参与这两个任务的历史现场来源链。
