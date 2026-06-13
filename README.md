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

```bash
# 找到 libcusparse.so.12 的路径（只需执行一次）
CUSPARSE=$(find ~/isaacsim -name "libcusparse.so.12" 2>/dev/null | head -1)
# 如找不到，用系统 CUDA：
# CUSPARSE=/usr/local/cuda-12.1/targets/x86_64-linux/lib/libcusparse.so.12

# 双臂画图演示（右臂画圆，左臂画方块）
LD_PRELOAD=$CUSPARSE ~/isaacsim/isaac-sim.sh --exec \
  isaac_sim/simforge/demos/dual_arm_draw.py

# 左臂抓托盘耳片并提升
LD_PRELOAD=$CUSPARSE ~/isaacsim/isaac-sim.sh --exec \
  isaac_sim/simforge/demos/ear_grasp_lift.py

# 夹爪物理抓取力实时可视化（双臂 + omni.ui 力显示面板）
LD_PRELOAD=$CUSPARSE ~/isaacsim/isaac-sim.sh --exec \
  isaac_sim/simforge/demos/gripper_force_demo.py

# 仅打开场景（不运动）
~/isaacsim/isaac-sim.sh --exec \
  isaac_sim/simforge/demos/open_scene.py
```

场景加载完成、IK 预计算结束后，控制台打印 `Ready — press Play`，在 GUI 按 **▶ Play** 开始运动。

---

### SimForge 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **配置** | `config.py` | 集中管理所有外部路径，支持环境变量覆盖 |
| **运动学** | `core/kinematics.py` | URDF 加载、正运动学 FK、关节链构建 |
| **规划** | `core/planning.py` | cuRobo IK 求解、笛卡尔路径规划 |
| **夹爪** | `core/gripper.py` | EG2-4C2 xform/物理驱动控制、pad 几何 |
| **场景工具** | `core/scene_utils.py` | USD xform 操作、摩擦材质、关节约束 |
| **演示** | `demos/` | 可直接运行的完整演示脚本 |
| **工具** | `tools/` | 一次性 USD 设置工具（碰撞、相机、物理等）|
| **场景** | `scenes/main.usd` | 追踪的场景快照 |
| **里程碑** | `milestones/INDEX.md` | 已验收的历史里程碑存档（只读）|

### 场景快照更新

修改 Isaac Sim 场景后，用以下命令保存快照并 commit：

```bash
cd isaac_sim/simforge/scenes
./checkpoint.sh "feat(scene): 描述本次变更"
```

---

## 依赖环境

| 依赖 | 版本 | 说明 |
|------|------|------|
| Isaac Sim | 5.1.0-rc.19 | 安装到 `~/isaacsim` |
| CUDA | 12.1 | `LD_PRELOAD` 需要 `libcusparse.so.12` |
| cuRobo | — | 安装到 Isaac Sim 的 Python 环境中 |
| numpy / scipy | — | 随 Isaac Sim Python 自带 |
| JAKA minicobo URDF | — | 位于 `jaka_ros2/src/jaka_description/urdf/` |
| cuRobo YAML | — | 位于 `jinyu_ros_pkg/nodes/simulation/` |
| 场景 USD | — | 需要 `2026061100_main.usd` 及其引用的资产 |

### 场景 USD 资产依赖

`2026061100_main.usd` 内部引用以下外部资产（需与 USD 文件保持相对路径关系）：

```
~/Developer/robot_usds/grippers/Inspire_EG2_4C2/Inspire_EG2_4C2.usd
~/Developer/PG-JY/isaac_sim/cad_assets/CAM_Mount.usd
~/Developer/PG-JY/isaac_sim/cad_assets/force_sensor.usd
~/Developer/PG-JY/isaac_sim/cad_assets/gripper_flange.usd
```

克隆仓库后，将 `robot_usds/` 放到 `~/Developer/robot_usds/`，CAD 文件已在仓库内 (`isaac_sim/cad_assets/`)。

---

## 关键设计原则

1. **不可以假抓** — 只有物理接触确认（`force_stop_step is not None`）后才创建 FixedJoint
2. **夹爪垂直地面** — EG2 X 轴 = 世界 -Z，夹爪在 Z 方向闭合
3. **里程碑只读** — `milestones/` 目录下内容禁止修改
4. **同时只开一个 Isaac Sim** — 启动前先检查并 kill 旧进程
5. **用 `isaac-sim.sh --exec` 启动** — 不用 `python.sh`（会触发 Storm 渲染器）

---

## 各模块 API 速查

### 正运动学
```python
from kinematics import load_joints, chain_to_link, fk, ARM_JOINTS, DEFAULT_ARM_URDF
arm_joints  = load_joints(Path(DEFAULT_ARM_URDF))
link6_chain = chain_to_link(arm_joints, "Link_0", "Link_6")
T = base_world @ fk(link6_chain, dict(zip(ARM_JOINTS, q.tolist())))
```

### IK 求解
```python
from planning import selected_pad_midpoint, solve_pad_pose_ik
base_world, pad_world, link6_to_pad = selected_pad_midpoint(stage, cache, "left")
q, pos_err, up_err, fwd_err, ok, msg = solve_pad_pose_ik(
    link6_chain, lower, upper, base_world, link6_to_pad,
    target_xyz, up_world, forward_world, seeds)
```

### 夹爪控制
```python
from gripper import gripper_link_transform, pad_separation_m, setup_gripper_xform_ops
gap_m = pad_separation_m(joint_angle_rad)          # 当前间距
T     = gripper_link_transform("left_pad", angle)  # FK 变换矩阵
ops   = setup_gripper_xform_ops(stage, gripper_root, "my_suffix")
```

### USD xform
```python
from scene_utils import gf_matrix_from_column_transform
op.Set(gf_matrix_from_column_transform(T_4x4_numpy))
```
