# 最短复现：使用实际执行快照，不默认当前HEAD

正式R3的权威源为非Git目录`runs/client_sci_closed_loop_20261027/r3/engine_training/`；每run的`task_source.py`、实际helper、`run_config.json`、`scene.json`与manifest SHA是归档依据。

已核对`event_qualified_seed31/task_source.py`与冻结engine中的`contact_composition_v8.py`一致：

`e31be3c1bc94be7935b19a750057739c97c9339c34fb8ee661109d6eaef9a414`

冻结launcher `run_contact_v8.py`：

`6c4c58f9ce160326c7eeeeff2932f58dcb3ebf943e27d3ef62b7ec28ee693823`

共用研究asset：`runs/contact_handoff_narrow_asset`，不是实测料盘或现场CAD。`jaka_paddle.usda`预期SHA：

`30d2c7250f2e09fdf7da889a536165e5a89bfe5e6ecd4f63bbb45f9b6495fde0`

仿真与Python仍使用190已部署的Isaac Sim5.1/Lab2.3.0和`envs/isaac51-check/bin/python`；不安装其他版本，不改驱动。运行前先检查目标GPU空闲，不与已有worker重叠。以下输出路径必须不存在，原训练目录不覆盖。

```bash
ARES_RESEARCH_ROOT=/home/andyee/robot-research/ARES-Skilllab
OMNI_KIT_ACCEPT_EULA=YES \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
"$ARES_RESEARCH_ROOT/envs/isaac51-check/bin/python" \
  "$ARES_RESEARCH_ROOT/runs/client_sci_closed_loop_20261027/r3/engine_training/run_contact_v8.py" \
  --asset "$ARES_RESEARCH_ROOT/runs/contact_handoff_narrow_asset" \
  --output "$ARES_RESEARCH_ROOT/runs/client_sci_closed_loop_20261027/reproductions/qualified31_dev_repeat_001" \
  --gpu 1 --num-envs 16 --family slot --steps 0 --controller residual --seed 31 \
  --frame task --task-version v6b --gamma 1 --rollout-steps 256 --episode-seconds 24 \
  --scenario-file "$ARES_RESEARCH_ROOT/runs/client_sci_closed_loop_20261027/configs/development.json" \
  --eval-seed 102841 --eval-episodes 32 --skip-capture --contact-following --log-contact-history \
  --contact-steering 0 --exact-footprint --guarded-alignment --yaw-enabled --yaw-reference zero \
  --action-cost executed --recontact-enabled --parameter-hold 32 --event-refresh \
  --policy-mode qualified_gaussian --exact-recontact --feedback-kp 2 \
  --load-policy "$ARES_RESEARCH_ROOT/runs/client_sci_closed_loop_20261027/r3/event_qualified_seed31/policy.zip"
```

这是32例开发评估重执行，0额外训练，不是训练视频。使用原case seed/场景/模型。复现普通对照，将checkpoint改`event_ordinary_seed31/policy.zip`、policy-mode改`gaussian`，其他条件相同。tool合同/时间上限不要用R1/R2旧值替代。

从model/source bundle恢复时保持`engine_training`的实际模块文件名。若只有单run快照，需将`task_source.py`恢复为`contact_composition_v8.py`、`boundary_credit_source.py`为`contact_credit_v2.py`、`frame_source.py`为`contact_task_frame_v7.py`、`geometry_source.py`为`contact_geometry_v8.py`，并保留原名的masked-policy、两种recontact、真实gradient-audit模块，配套使用归档launcher。不要混入当前HEAD中不匹配的helper。

独立验收使用本轮冻结的`validate_closed_loop_physics.py`及其旧OBB/FK依赖，资产目录需沿用原研究asset。先核查`RUN_COMPLETE.json`、无`exception.txt`，再核对raw success、`INDEPENDENT_CLOSED_LOOP.json`与终态保持；cuRobo或单帧到达不是该任务的成功标签。

严格同GPU/版本冷启动可以检查物理轨迹重现；不保证跨驱动/GPU求解器缓存逐bit相等。模型参数hash应按tensor字节验证，不把ZIP时间戳变化当成网络更新。
