# EXP-001：可重复的接触失败与强控制器修正

**决定：继续 S2 / EXP-001，先独立验证最终强控制器，不训练新网络。** 已取得有实际物理依据的控制改进；独立 RL 方法增益仍为 UNVERIFIED。新工作树 `repos/ARES-SCI-program`，分支 `research-sci-program`，起点 `eb8a93f`；既有研究工作树和 ARES-R 未修改。

## 结果与反证

| 协议 | 独立初态/执行数 | 原积分控制 | 有界反馈 | 解释 |
|---|---:|---:|---:|---|
| 冻结控制器跨进程复跑，3参数 | 6初态×3组合×2进程 | 每进程5/18 | 未比较 | 18/18成败一致，17/18记录状态一致；不把36次当36场景 |
| 新设计组，两者均有当前原生到位保持 | 4初态，8执行 | 1/4 | 4/4 | 救回3、退化0；4/4实际ALIGN边界记录一致 |
| 冻结后新确认组，同到位保持 | 8初态，16执行 | 1/8 | 5/8 | 救回5、退化1；7/7到达ALIGN的边界记录一致，另1例共同抓持失败 |
| 单因素“原地轴向保持”修正，事后诊断 | 4已看初态，4执行 | 前版反馈2/4 | 新版4/4 | 两个失败救回、两个成功回归保持；不是新的独立确认组 |

- 历史诊断初态721000、723002；设计724000–724003；确认725000–725007。确认组在运行前已冻结控制器，看过结果后现在属于开发证据，不再是 sealed test。
- 725002是明确退化：原积分成功，前版反馈停在约19.19 mm；725005前版反馈停在约23.54 mm。记录显示对齐门限反复触发整段撤回，而非单纯“后验价值不准”。仅把失配时的轴向目标改为当前实际深度，两例分别达到3.42、3.92 mm，原生成功；另外725000、725006成功保持。
- 725004在两种控制器中均未通过抓持检查，后段修正不能解释或救回该前段失败。
- 8例是小样本：5救回/1退化的双侧精确符号检验 p=0.21875。不能宣称统计可靠泛化，更不能宣称新RL算法。最终新版尚未独立确认，**不能拼接为7/8**。

## 修正了什么

共同保留原生 `PlugCharger-v1`、200步、位置≤5 mm且角度≤0.2 rad；没有状态设定、瞬移、改动力学或成功阈值。所有实际结果来自 `env.step`，同一固定抓持15°、接近25 mm用于冷进程控制器比较。

新控制器在相同抓取/抬起/ALIGN后，以实测工具位姿进行小步校正。横向误差<0.4 mm、角误差<0.03 rad时推进2 mm；目标TCP位移≤4 mm、旋转≤0.04 rad、每关节指令相对当前q裁剪到±0.05 rad。位置积分逐轴限制±8 mm，抵消真实静差。单初值IK接受阈值1e-4；不是新学习机制，也不声称真实机器人动态/力安全认证。

初版反馈失配时撤回原接近深度，造成循环；候选新版 `servo_local_hold_stop` 只保持当前深度，继续横向/姿态校正。两种被比较的完整控制器都可观测当前任务目标判据，一旦达到原生目标便保持当前关节位置到第200步；终态仍独立检查，短暂成功不计终态成功。

调试负结果均保留：极短路径TOPP异常；无积分反馈在约0.47 mm横向静差停滞；过严IK接受阈值导致失败；前版推进后重新对齐可离开目标。未删除失败或用渲染代替物理。

## 成本与证据边界

- **14个不同初态，74个完整episode，14,800个完整记录控制步；另一次TOPP崩溃105步，总计14,905物理控制步。优化器更新0。**
- 确认8例总成本：原积分85次规划调用、0.380 s规划墙钟、43.967 s整段墙钟；反馈457次规划/IK调用、0.253 s规划/IK墙钟、45.024 s整段墙钟。IK和完整路径规划不是同等操作；墙钟包含记录I/O，不能解释为GPU kernel时间。
- Python/NumPy/Torch seed固定31031；MPLib0.1.1没有使用可验证的全局seed接口。第一协议只保证每套批次冷进程；后续每个“场景×控制器”均独立冷进程，避免前一episode的额外IK消耗影响后一episode的RRT。
- 17/18重复一致与边界一致检查覆盖记录的q/qvel、TCP/工具/插座位姿、线角速度、控制动作及接触力标量。未直接验证PhysX内部接触缓存，不称完整内部仿真状态逐位相同。唯一重复不同的组合为失败→失败。
- 强基线修正后仍有前段抓持与后段停滞问题；尚未证明这些剩余差异必须由RL解决。候选新版的4例事后改善可能继续缩小RL空间，下一步应先证伪这一可能。

## 文件与精确来源

- [METRICS.json](METRICS.json)：全部运行统计、配对、完整轨迹SHA256、崩溃精确步数。
- [EPISODES.csv](EPISODES.csv)：每条真实执行的原生结果、首次成功、末10步成功数、失败分类、跟踪/接触及计算成本。
- [RUN_INDEX.json](RUN_INDEX.json)：每次运行的源码与依赖SHA、种子、配置、原始路径；每个运行都保留当时源码快照。
- [matched_controls.png](matched_controls.png)：设计/冻结确认分开比较；[contact_effect.png](contact_effect.png)：同初态真实目标误差与接触曲线。
- 源码：`research/skill_handoff_lab/tools/program_charger.py`、`program_charger_feedback.py`、`program_charger_analyze.py`、`program_charger_render.py`。
- 原始数据仅保留本机非Git：`/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-001/`，未声称全部大文件已上传GitHub。视频位于该目录`visuals/controller_pair.mp4`；仅显示记录姿态、零新增物理步，源与渲染校验见`visuals/receipt.json`。Git提交精简结果、图和大文件索引。
- ManiSkill固定`62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`，已核对Apache-2.0许可证。相同既有Python/Torch/SAPIEN环境，单张RTX3080，断网、ARES-R只读、无实机/现场受限数据。

## 复现与下一轮

从此工作树，使用现有隔离器；每条命令输出目录必须为新目录。每次仅运行一个GPU进程：

```bash
PY=/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python
bash research/skill_handoff_lab/tools/handoff_offline.sh "$PY" \
 research/skill_handoff_lab/tools/program_charger.py \
 --out /home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-001/reproduction_new \
 --variant servo_local_hold_stop --seeds 725002 --single-angle 15 --single-distance .025
```

复现旧控制器用`--variant frozen_stop`，冻结确认前版用`servo_stop`。未给单一参数时运行15°/25 mm、45°/50 mm、45°/75 mm三组合；同进程多个episode的规划随机流不能视作独立冷启动。完整分析：

```bash
python3 research/skill_handoff_lab/tools/program_charger_analyze.py \
 --root /home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-001 \
 --out /tmp/ares-exp001-review
python3 research/skill_handoff_lab/research_program/scripts/program_status.py check
```

下一次仍执行 **EXP-001**：先冻结`servo_local_hold_stop`，在未查看的新初态小组与`servo_stop`及`frozen_stop`冷进程配对，验证退化是否被消除；再对剩余真实接触失败定义EXP-002动作探针。若强规则继续消除差异，应换任务或操作条件，而非恢复小数据FQ/风险头训练。本轮没有重编译论文。
