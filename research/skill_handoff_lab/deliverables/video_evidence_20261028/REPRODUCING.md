# 复现与核验

在 GitHub 克隆 ARES 的 `codex/ares-sci-autonomous-research-visual-20261027` 分支后，进入本目录。此包以普通 Git 文件保存，不需要 Git LFS 或访问 NAS。最终 MP4、NPZ 与模型 ZIP 均在仓库内；大文本为 gzip。

## 1. 先校验并解压

只需 Python 标准库，无需 GPU：

```bash
python3 reproduce.py verify
python3 reproduce.py extract --stage /tmp/ares-selected-video-evidence
```

解压目录必须是新目录。脚本逐项验证仓库中的文件哈希，并检查 gzip 解压后与原始字节哈希一致。解压目录保留原始归档数据与源码，不覆盖本仓库。原始绝对路径留在 `run_config.json`/manifest 中供追溯。

## 2. 不运行物理，只从原始相机与时钟重建两段视频

使用带有 NumPy、Pillow、Matplotlib 的 Python，系统应提供 `/usr/bin/ffmpeg`（libx264）和 DejaVu Sans 字体：

```bash
python3 reproduce.py encode-r3 --stage /tmp/ares-selected-video-evidence --execute
python3 reproduce.py compose-r0 --stage /tmp/ares-selected-video-evidence --execute
```

输出在解压目录的 `new_runs/reencoded_r3/` 与 `new_runs/recomposed_r0/R0_case_001/`。这验证视频来自归档的原始相机流和物理时间；此步骤不运行 Isaac、不产生新的物理数据。相同 FFmpeg/字体环境下可以比较 MP4 哈希；跨编码器版本以帧内容和时钟为准。

## 3. 独立重跑 PhysX 与训练

使用已安装 Isaac Sim 5.1、Isaac Lab v2.3.0 及 `ENVIRONMENT.json` 所列 RL 依赖的 Python。运行时必须满足 NVIDIA 的兼容要求；本包不安装驱动或变更宿主机。示例中的 `/path/to/isaac/python` 替换为自己的解释器。先运行命令计划确认路径，再添加 `--execute`。每次单 GPU，默认 GPU0；可加 `--gpu N`。

紧凑 16 环境、seed31、12,288 交互的真实在线训练：

```bash
python3 reproduce.py r3 --stage /tmp/ares-selected-video-evidence \
  --python /path/to/isaac/python --execute
```

新运行在 `new_runs/r3_compact/`，保留配置、训练状态、梯度审计、模型和相机流。再次编码新录像：

```bash
/path/to/isaac/python /tmp/ares-selected-video-evidence/code/export_compact_training_video.py \
  --run /tmp/ares-selected-video-evidence/new_runs/r3_compact \
  --output /tmp/ares-selected-video-evidence/new_runs/r3_new_video
```

R0 两侧实际执行（使用归档的 seed41 checkpoint，测试期间不训练）：

```bash
python3 reproduce.py r0 --stage /tmp/ares-selected-video-evidence \
  --python /path/to/isaac/python --execute
```

输出 `new_runs/R0_baseline/` 和 `new_runs/R0_residual41/`。环境仍为原配置 16 个，全部 32 案例数值评估，选中 case001 录制相机流。原始两侧执行配置分别保留，不能与后续 R3 配置混用。

重新训练 R0 右侧原模型，65,536 交互、seed41：

```bash
python3 reproduce.py r0-train --stage /tmp/ares-selected-video-evidence \
  --python /path/to/isaac/python --execute
```

此项运行保存的原始训练任务源码，只重定位资产、配置和输出路径；新模型位于 `new_runs/r0_train_seed41/`。它不是自动挑选更好 checkpoint 的流程。

所有输出路径已存在时会拒绝覆盖。重跑会真实推进 PhysX；CUDA/PhysX 求解与渲染具有版本和数值依赖，不承诺新轨迹、新参数或新 MP4 与历史逐位相同。归档原始数据及其哈希才是这两段已发布视频的证据。

## 4. 独立采样物理验收

存档中的 `INDEPENDENT_TASK_AND_CONTACT.json` 保留实际验收结论。重新检查提取出来的原始记录：

```bash
python3 reproduce.py audit --stage /tmp/ares-selected-video-evidence \
  --python /path/to/python-with-numpy-scipy-python-fcl --execute
```

该命令在解压副本中重新生成验收报告；仓库存档不变。使用包内只读 FK helper、URDF 和资产元数据，不依赖 ARES-R 生产配置。要验收新运行，可将打印命令中的 `--run` 换为 `new_runs/...`。R3 另有工具姿态合同，完整工具验收入口为 `code/validate_closed_loop_physics.py`，参数相同。

独立检查覆盖采样 OBB 穿透、软限位、FK 推板误差及稳定终态，保留源码中的阈值。不将它描述为机械臂全网格 CCD 或实机安全认证。

## 5. 文件字段与时钟

NPZ 自带 `columns`/字段数组，以实际字段为准，避免硬编码旧版列位置。训练阶段使用 `TRAINING_120HZ_STATE.npz`；训练后冻结评估使用 `actual_physics_trace.npz`、`frozen_evaluation.json` 与 post-step 记录。视频映射依据原始 frame JSON 的物理时间，不能用 MP4 帧号直接推断仿真步。

`ARTIFACT_SHA256.json` 覆盖原始证据与快照，`PUBLICATION_SHA256.json` 额外覆盖本包说明、便携复现脚本与环境记录。校验器和本包 Git commit 一起提供可追溯入口。
