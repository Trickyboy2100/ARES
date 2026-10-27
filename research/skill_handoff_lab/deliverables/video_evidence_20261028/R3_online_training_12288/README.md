# 16 环境紧凑阵列：真实在线训练录像

使用文件：`video/actual_online_training_16env.mp4`，SHA256：
`78a36b25901f1e53615500f9d65798bab844e891c2e600234af2d7d1afa5200e`。

1920×1080、12 fps、190 帧、15.834 秒。单个 Isaac 应用中 16 个独立 PhysX 环境，4×4 排列，环境间距 0.85 m，近景正向相机。机器人、自由工件、槽位与任务控制沿用原训练配置。该视频是 seed31 的新在线复录；相对旧 2.5 m 阵列改变了全局间距及相机，不能声称逐位相同的原轨迹。

任务为自建 P2 导向槽接触操作：JAKA 研究推板通过接触推动自由工件完成两阶段任务。Actor 使用仿真工件状态；控制器为 `residual`，策略模式 `qualified_gaussian`，87 维观测、5 维策略动作，实际有效动作由执行接口约束。它不是双臂抓取、真实金属料盘放置或公开 benchmark。

总计 12,288 次训练环境交互，16 环境每个 768 个控制步；控制频率 60 Hz、物理频率 120 Hz，训练经过 1,536 个共享物理 tick（12.8 s）。PPO 每 4,096 个环境交互执行一次 rollout 优化，三次调用后参数哈希均改变；每次 6 epochs。视频是短预算训练过程展示，不据此主张收敛或方法性能优势。视频帧头显示当前物理时间、实际交互数和已完成 rollout 更新。

末尾明确显示 3 秒停止帧。原始相机流为按标称 6 Hz 加优化事件采集，最终视频按真实时间戳进行零阶保持；不生成中间运动。精确帧时间以记录为准。

| 证据 | 相对位置 |
|---|---|
| 本视频专属来源、变化与输出哈希 | `video/COMPACT_VIDEO_PROVENANCE.json` |
| 原始相机 MP4 | `run/video_raw/matrix_live_training.mp4` |
| 每个原始帧与物理时钟对应 | `run/MATRIX_LIVE_TRAINING_FRAMES.json.gz` |
| 120 Hz 训练状态与字段 | `run/TRAINING_120HZ_STATE.npz` |
| 完整原始策略动作日志 | `run/action_log.json.gz` |
| rollout 参数前后哈希、优化事件 | `run/LIVE_OPTIMIZER_EVENTS.json` |
| 第一次真实优化的梯度审计与样本 | `run/FIRST_REAL_ROLLOUT_GRADIENT.json`、`run/first_real_rollout_gradient_samples.npz` |
| 训练 episode、CSV 学习记录 | `run/training_episodes.json`、`run/ppo/progress.csv` |
| 最终策略、首次更新前策略 | `run/policy.zip`、`run/policy_before_first_update.zip` |
| 训练后 4 案例数值评估 | `run/frozen_evaluation.json`（不是正式独立主结果） |
| 精确执行任务和依赖 | `run/recorded_task.py`、`run/base_task_source.py`、同目录 helper 快照 |
| 原始配置、场景、运行统计 | `run/run_config.json`、`run/scene.json`、`run/manifest.json` |
| 源码注入及物理方法 AST 一致性 | `run/VIDEO_SOURCE_PATCH.json` |
| 单次运行控制台日志 | `run.log.gz` |

`actual_physics_trace.npz` 和 `post_step_contact_history.npz` 对应训练后的评估记录；训练期间的高频记录请使用 `TRAINING_120HZ_STATE.npz`，不要混用二者。`LIVE_TRAINING_MONTAGES.json` 对应的关键帧另存于 `run/video_keyframes/`，这些 montage 与最终单相机阵列视频分别标注。
