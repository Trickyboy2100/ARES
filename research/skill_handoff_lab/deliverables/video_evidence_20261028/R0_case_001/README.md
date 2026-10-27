# Case001：历史反馈控制与 Residual PPO 的配对执行

使用文件：`comparison/paired_actual_physx.mp4`，SHA256：
`78f40ae24cde3e19cb9d595b05a67f4df097d5d3b3332267fe4a7ba4c454d553`。

1920×1080、30 fps、15.483 秒。左侧 `Feedback (historical)` 为 damped 状态反馈控制；右侧 `Residual PPO (historical seed41)` 为在该控制基础上加入学习残差的策略。两侧为独立冷启动后的实际 PhysX 闭环执行，相机流按物理时间并列合成。

这是 P2 槽位任务的冻结 case001（`deep_geo0`）。开口 gap 0.05 m、目标纵向参数 0.148 m、横向偏移 −0.028 m、槽方向 0.12 rad、工件初始 yaw 0.06 rad、质量 0.05 kg、摩擦参数 0.3；完整参数见 `configs/r0_test_geometry.json`。自由工件边长 0.04 m，研究推板 0.03×0.03×0.04 m。单位均采用 SI。

本案例左侧原始/独立验收均失败，在约 11.983 s 终止；右侧原始/独立验收均成功，约 8.767 s 终止。两侧可检查的初始关节、速度、工件位置/四元数/速度及推板位置最大差为零；不声称隐藏的 PhysX solver/cache 状态相同。成功要求包含位置 8 mm、线速度 0.01 m/s、角速度 0.2 rad/s、yaw 8°、upright 5°，稳定保持 30 个控制步（0.5 s），另有采样碰撞/限位验收。

该案例是实际单案例正向差异，不能替代多种子总体比较，也不是新提出机制的独立消融证据。选择协议与全部 32 案例结果均保留。右侧加载 checkpoint SHA256：`b0597000fa35ad72f0f0dfecb518d2191388a5bcc7cd7d21bdfb5d4f762a5198`，原训练 seed41、65,536 在线交互。配对视频执行阶段 `steps=0`，没有在测试案例继续训练。

| 证据 | 相对位置 |
|---|---|
| 两侧来源、精确初始比较、结果、时间、资产/模型哈希 | `comparison/PAIR_PROVENANCE.json` |
| 原始相机流和精确帧物理时钟 | `R0_baseline/video_raw/case_001.mp4`、`R0_residual41/video_raw/case_001.mp4` 及各自 `case_001_frames.json.gz` |
| 两侧完整 120 Hz 状态 | 各自 `actual_physics_trace.npz`，包含该运行全部 32 案例 |
| 两侧原始动作 | 各自 `action_log.json.gz` |
| 原始结果与独立验收 | 各自 `frozen_evaluation.json`、`INDEPENDENT_TASK_AND_CONTACT.json` |
| 动作期间状态重写计数、原始帧/轨迹/动作哈希 | 各自 `VIDEO_CAPTURE_PROVENANCE.json` |
| 执行源码与配置 | 各自 `base_task_source.py`、`recorded_task.py`、`run_config.json` |
| 原始训练、模型、学习记录 | `original_training_seed41/`，含 `policy.zip`、`training_episodes.json.gz`、`ppo/progress.csv` |
| 展示用轨迹图、CSV、视频与物理时间映射 | `comparison/case_001_trajectories.*`、`case_001_states.csv`、`video_physics_timeline.csv` |
| 精确的视频合成脚本 | `comparison/composition_source.py` |

展示 CSV 每四个物理采样取一条，不能代替 NPZ 原始数据。接触为仿真代理，独立碰撞是采样 OBB，非硬件安全保证；没有接触抓取或现场任务完成声明。
