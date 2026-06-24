# PG-JY — 金榆双臂机器人开发平台

双臂 JAKA MiniCobo + Inspire EG2-4C2 夹爪系统的完整开发仓库，涵盖仿真、运动规划、视觉位姿估计三个模块。

---

## 仓库结构

```
PG-JY/
├── isaac_sim/
│   ├── simforge/          ← 主仿真实验室（见下文）
│   ├── playground_dual_arm_control/     ← 旧版 playground（勿删）
│   ├── playground_dual_arm_control_v2/  ← v2 playground（当前开发参考）
│   └── cad_assets/        ← STEP/STL 零件文件（托盘、实验台等）
│
├── jaka_ros2/             ← JAKA 官方 ROS2 包
│   └── src/jaka_description/urdf/
│       ├── jaka_minicobo.urdf           ← 纯臂 URDF（规划用）
│       └── jaka_minicobo_gripper.urdf   ← 含夹爪 URDF（cuRobo 用）
│
├── jinyu_ros_pkg/         ← 金榆自研 ROS2 包
│   └── nodes/simulation/
│       ├── jaka_minicobo_curobo.yml     ← cuRobo 机器人配置
│       └── jaka_minicobo_gripper.urdf   ← cuRobo URDF（含夹爪）
│
├── cuRobo/                ← cuRobo 运动规划库（本地 fork/安装）
├── pose_est/              ← 视觉位姿估计（FoundationPose 等）
├── vision_pose_benchmark/ ← 视觉算法 benchmark 实验
└── docs/                  ← 技术文档
```

---

## 核心模块：SimForge

**`isaac_sim/simforge/`** 是本项目的仿真主体，git 版本管理，模块化组织。

### 快速上手

**第一步：配置路径**（加入 `~/.bashrc`）

```bash
# Isaac Sim 安装根目录
export ISAACSIM_ROOT=~/isaacsim

# 主仿真场景 USD 文件（需要机器上有对应资产文件）
export SIMFORGE_SCENE=~/isaacsim/playground/2026061100_main.usd

# JAKA minicobo URDF 所在目录（含 jaka_minicobo.urdf）
export SIMFORGE_URDF_DIR=~/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf

# cuRobo 机器人 YAML 配置文件路径
export SIMFORGE_CUROBO_CFG=~/Developer/PG-JY/jinyu_ros_pkg/nodes/simulation/jaka_minicobo_curobo.yml
```

**第二步：验证配置**

```bash
python3 isaac_sim/simforge/config.py
```

输出 `All paths OK.` 则配置完成。

**第三步：启动演示**

> **⚠️ 使用 `LD_LIBRARY_PATH`，不要用 `LD_PRELOAD`**（会导致 ld.so assertion failure）

```bash
# 每次启动前先清理旧进程
pkill -f "isaacsim/kit/kit" 2>/dev/null; sleep 2

# 设置所有 nvidia 库路径（每次新终端执行一次）
ISAAC_NVLIBS=$(find ~/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia -type d -name "lib" | tr '\n' ':')
export LD_LIBRARY_PATH="${ISAAC_NVLIBS}${LD_LIBRARY_PATH:-}"
```

---

## 演示目录

### ★ tray_grasp_cycle — 双臂托盘搬运循环（主 Demo）

**2026-06-25 验证通过** · 左臂 FixedJoint 夹取耳片 → 抬升 30cm → 交接右臂 → 右臂 carrier 送至烘干机 → 循环，7+ cycles 稳定。

```bash
# 一键启动（自动 kill 旧进程）
bash isaac_sim/simforge/demos/tray_grasp_cycle/launch.sh

# 带完整日志捕获 + 自动摘要报告
bash isaac_sim/simforge/demos/tray_grasp_cycle/record.sh
```

**技术要点：**
- **FixedJoint 物理夹取**：L 臂 `right_pad`（kinematic rigid body）→ tray 的 FixedJoint 实现 6DOF 跟随
- **carrier 交接机制**：R 臂 pad 无 RigidBodyAPI；用隐形 kinematic carrier cube 作为中间体
- **cuRobo 路径规划**：R 臂 home→handoff、handoff→dryer 用 cuRobo persistent subprocess worker
- **branch A linspace**：L 臂 home→preL 用 `np.linspace(q_zero, q_pre_L, 120)`（必须，cuRobo 落在 branch B 会导致 FixedJoint 失效）
- **GUI**：双臂实时面板（力历史、pad-Y 曲线、carrier 位置、抬升高度）

**验证数据（2026-06-25）：**

| 指标 | 数值 |
|------|------|
| IK pre-grasp 误差 | 0.0 mm |
| 力停止触发 | ≥ 2 N |
| 最大抬升高度 | ~30 cm |
| tray 交接位置 | [-0.45, 0.76, 1.64] m |
| 最大关节跳变 | ≤ 0.039 rad（L/R） |
| 循环稳定性 | 7+ cycles 确认 |

---

### gripper_force_demo — 夹爪力实时可视化

```bash
bash isaac_sim/simforge/demos/gripper_force_demo/launch.sh
```

双臂同时抓取球体；`omni.ui` 面板实时显示法向力、摩擦力、压缩量。设计模板：`tray_grasp_cycle` 的接触力模型和 GUI 架构均源自此 demo。

---

### 其他演示

```bash
# 双臂画图（圆 + 方块）
~/isaacsim/isaac-sim.sh --exec isaac_sim/simforge/demos/dual_arm_draw.py

# 仅打开场景（无运动）
~/isaacsim/isaac-sim.sh --exec isaac_sim/simforge/demos/open_scene.py
```

---

## SimForge 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **配置** | `config.py` | 集中管理所有外部路径，支持环境变量覆盖 |
| **运动学** | `core/kinematics.py` | URDF 加载、正运动学 FK、关节链构建、`get_world_pose` |
| **规划** | `core/planning.py` | IK 求解、笛卡尔路径规划、`selected_pad_midpoint` |
| **夹爪** | `core/gripper.py` | EG2-4C2 xform 控制、pad 几何、`pad_separation_m` |
| **场景工具** | `core/scene_utils.py` | USD xform 操作、矩阵转换 |
| **关节限位** | `core/ik_sanity.py` | `joint_limits(chain)` 从 URDF 提取 |
| **托盘耳片抓取** | `demos/tray_grasp_cycle/` | 完整循环 demo（主体） |
| **力 Demo** | `demos/gripper_force_demo/` | 球体抓取力可视化（设计模板） |

---

## 场景快照更新

修改 Isaac Sim 场景后，保存快照并 commit：

```bash
cd isaac_sim/simforge/scenes
./checkpoint.sh "feat(scene): 描述本次变更"
```

当前快照：`isaac_sim/simforge/scenes/main.usd`（源自 `~/isaacsim/playground/2026061100_main.usd`）

---

## 依赖环境

| 依赖 | 版本 | 说明 |
|------|------|------|
| Isaac Sim | 5.1.0-rc.19 | 安装到 `~/isaacsim` |
| CUDA | 12.x | 仅需 `nvidia/nvjitlink/lib`（通过 `LD_LIBRARY_PATH`）|
| cuRobo | — | 安装到 Isaac Sim 的 Python 环境中 |
| numpy / scipy | — | 随 Isaac Sim Python 自带 |
| JAKA minicobo URDF | — | 位于 `jaka_ros2/src/jaka_description/urdf/` |
| cuRobo YAML | — | 位于 `jinyu_ros_pkg/nodes/simulation/` |
| 场景 USD | — | `2026061100_main.usd` 含 `/World/Tray/tray_grasp_point` 标定 prim |

### 场景 USD 资产依赖

`2026061100_main.usd` 内部引用以下外部资产（需与 USD 文件保持相对路径关系）：

```
~/Developer/robot_usds/grippers/Inspire_EG2_4C2/Inspire_EG2_4C2.usd
~/Developer/PG-JY/isaac_sim/cad_assets/CAM_Mount.usd
~/Developer/PG-JY/isaac_sim/cad_assets/force_sensor.usd
~/Developer/PG-JY/isaac_sim/cad_assets/gripper_flange.usd
```

---

## 关键设计原则

1. **FixedJoint 物理夹取** — L 臂通过 kinematic `right_pad` → dynamic tray 的 FixedJoint 实现 6DOF 跟随；接触力由 PhysX 处理
2. **IK branch A 必须严守** — L 臂 home→preL 用 linspace（不用 cuRobo）；cuRobo 落在 branch B，FixedJoint 在 branch B 静默失效
3. **cuRobo 用 subprocess** — R 臂路径规划通过 `_curobo_worker.py` 独立子进程运行，避免与 Isaac Sim Python 环境冲突
4. **双臂每帧维持 FK** — 右臂必须每帧设置 joint 角度，否则 PhysX 会随机驱动它
5. **UI 在 `timeline.play()` 之前创建** — 否则第一帧挂死
6. **只开一个 Isaac Sim** — 启动前先 kill 旧进程（`pkill -f "isaacsim/kit/kit"`）
7. **禁止 `LD_PRELOAD`** — 只用 `LD_LIBRARY_PATH` 注入所有 nvidia/\*/lib 路径
8. **里程碑只读** — `milestones/` 目录下内容禁止修改

---

## API 速查

### 正运动学
```python
from kinematics import load_joints, chain_to_link, fk, ARM_JOINTS, DEFAULT_ARM_URDF, get_world_pose
arm_joints  = load_joints(Path(DEFAULT_ARM_URDF))
link6_chain = chain_to_link(arm_joints, "Link_0", "Link_6")
T = base_world @ fk(link6_chain, dict(zip(ARM_JOINTS, q.tolist())))

# 读取 USD prim 的世界位姿（4×4 column-major numpy）
T_prim = get_world_pose(stage, cache, "/World/Tray/tray_grasp_point")
```

### IK 求解
```python
from planning import selected_pad_midpoint, solve_pad_pose_ik, pad_world_transform
base_world, pad_world, link6_to_pad = selected_pad_midpoint(stage, cache, "left")
q, pos_err, up_err, fwd_err, ok, msg = solve_pad_pose_ik(
    link6_chain, lower, upper, base_world, link6_to_pad,
    target_xyz, up_world, forward_world, seeds)
T_pad = pad_world_transform(link6_chain, base_world, link6_to_pad, q)
```

### 夹爪控制
```python
from gripper import gripper_link_transform, pad_separation_m, setup_gripper_xform_ops
gap_m = pad_separation_m(joint_angle_rad)          # 当前间距（m）
T     = gripper_link_transform("left_pad", angle)  # FK 变换矩阵
ops   = setup_gripper_xform_ops(stage, gripper_root, "my_suffix")
```

### USD xform
```python
from scene_utils import gf_matrix_from_column_transform
op.Set(gf_matrix_from_column_transform(T_4x4_numpy))
```
