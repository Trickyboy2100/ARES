# SimForge — JAKA MiniCobo 双臂仿真平台

基于 **NVIDIA Isaac Sim 5.1.0** 的 JAKA MiniCobo 双臂机器人 + Inspire EG2-4C2 夹爪仿真框架。

## 最新 Demo：双臂完整交接循环（tray_grasp_cycle）

> **新同事请先阅读 [SETUP.md](SETUP.md) — 包含从零到运行 demo 的完整步骤（英文）。**

> **一键启动：**
> ```bash
> bash ~/simforge/demos/tray_grasp_cycle/launch.sh
> ```

当前最新版本实现了 **左臂夹取 → 举升 → 交接右臂 → 送入烘箱** 的完整端到端循环，主要特性：

| 特性 | 说明 |
|------|------|
| 夹取方式 | FixedJoint 刚性夹持（力传感器触发，F ≥ 3 N 锁定） |
| 末端姿态 | 80 步 Y 轴线性逼近，全程 jaw=0°、forward=0°（`ORIENT_WEIGHT_STRONG=0.30`） |
| 运动规划 | cuRobo（位姿感知，带缓存）；失败自动降级为单帧跳转 |
| 双臂交接 | 左臂携盘到交接点 → 右臂 FixedJoint 锁定 → 左臂释放 → 右臂送烘箱 |
| 场景重置 | 托盘释放后原位置闪现新托盘，循环不停机 |
| 场景资产 | 完全自包含（`assets/` 目录，无外部路径依赖，clone 即可运行） |

验证指标（2026-06-15）：
- IK 精度：pos_err = **0.0 mm**，jaw_err = **0.0°**，fw_err = **0.0°**（全 80 步）
- 夹取力：**F = 3.28 N**（pad Y = 0.4207 m）
- 场景：`scenes/main.usd`（全依赖本地 `assets/`）

---

## 硬件 / 系统要求

| 项目 | 要求 |
|------|------|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA RTX 3090 / 4090 或同级（显存 ≥ 20 GB） |
| 驱动 | NVIDIA Driver ≥ 525.85，支持 CUDA 12.x |
| RAM | ≥ 32 GB |
| 磁盘 | Isaac Sim 安装约 30 GB；本 repo 约 60 MB |

---

## 一、驱动与 CUDA 安装

```bash
# 查看当前驱动
nvidia-smi

# 若驱动版本 < 525，安装最新驱动（以 535 为例）
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot

# 安装 CUDA Toolkit 12（仅需 nvcc 工具，Isaac Sim 自带 runtime）
sudo apt-get install -y cuda-toolkit-12-3
```

> **注意**：Isaac Sim 5.1.0 自带 CUDA runtime，不需要单独安装 cuDNN 或 TensorRT。

---

## 二、Isaac Sim 安装

### 方法 A：Omniverse Launcher（推荐）

1. 前往 [https://developer.nvidia.com/isaac/sim](https://developer.nvidia.com/isaac/sim) 下载 Omniverse Launcher
2. 在 Launcher 中搜索 **Isaac Sim 5.1.0** 并安装
3. 默认安装到 `~/isaacsim/`

### 方法 B：命令行静默安装

```bash
# 下载 Isaac Sim package（替换为实际下载链接）
# 解压到 ~/isaacsim/
tar -xzf isaac-sim-5.1.0-linux-x86_64.tar.gz -C ~/
mv isaac-sim-5.1.0 ~/isaacsim
```

验证安装：

```bash
ls ~/isaacsim/isaac-sim.sh   # 应存在
~/isaacsim/isaac-sim.sh --help 2>&1 | head -3
```

---

## 三、Python 依赖安装

SimForge 使用两个 Python 环境：

| 环境 | 用途 |
|------|------|
| Isaac Sim 内置 Python 3.11 (`~/isaacsim/kit/python/bin/python3`) | 主仿真脚本（demo.py） |
| 系统 / Conda Python 3.13 | cuRobo 路径规划子进程 |

### 3.1 安装 cuRobo（路径规划）

```bash
# 推荐使用 Miniconda
conda create -n curobo python=3.13 -y
conda activate curobo

# 安装 cuRobo（需要 CUDA 12）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install curobo

# 验证
python -c "import curobo; print('cuRobo OK')"
```

### 3.2 安装 Isaac Sim 侧依赖

```bash
# 使用 Isaac Sim 自带 Python
~/isaacsim/kit/python/bin/python3 -m pip install scipy numpy
```

---

## 四、Clone 仓库

```bash
git clone git@github.com:Trickyboy2100/ARES.git ~/simforge
cd ~/simforge
```

克隆后目录结构：

```
simforge/
├── config.py                   ← 路径配置（通常只需设 ISAACSIM_ROOT）
├── robot/                      ← 机器人文件（URDF、Mesh、cuRobo 配置）
│   ├── jaka_minicobo.urdf      ← 关节链 URDF（仿真 IK 用）
│   ├── jaka_minicobo_gripper.urdf  ← 带夹爪 URDF（cuRobo 用）
│   ├── jaka_minicobo_curobo.yml   ← cuRobo 碰撞 / 运动学配置
│   ├── jaka_minicobo_meshes/  ← 手臂 STL mesh
│   └── eg2_4c2_meshes/        ← 夹爪 STL mesh
├── assets/                     ← 场景依赖资产（场景自包含，无外部路径依赖）
│   ├── cad/                   ← 台架、夹持法兰、力传感器等 CAD 模型
│   ├── jaka_minicobo/         ← JAKA MiniCobo USD + 材质/物理配置
│   ├── Inspire_EG2_4C2/       ← EG2-4C2 夹爪 USD + 材质/物理配置
│   ├── playground_config/     ← 场景环境光、物理参数 USD（原 playground 配置）
│   └── textures/              ← 地板纹理、ArUco 标记等贴图
├── scenes/
│   ├── main.usd               ← 主场景文件（双臂 + 托盘 + 烘箱；路径全部指向 ../assets/）
│   └── rewrite_paths.py       ← 维护工具：场景另存后重写外部路径
├── core/                       ← 共享模块
│   ├── kinematics.py
│   ├── planning.py             ← IK 求解器（支持 jaw+forward 双约束）
│   ├── gripper.py              ← EG2-4C2 夹爪控制
│   └── scene_utils.py
├── demos/
│   └── tray_grasp_cycle/       ← ★ 当前主 Demo
│       ├── demo.py
│       ├── launch.sh           ← 一键启动
│       └── record.sh           ← 启动 + 全日志录制
└── milestones/
    └── INDEX.md                ← 各版本验证指标记录
```

---

## 五、环境配置

### 5.1 设置环境变量

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
# Isaac Sim 安装路径（如不在 ~/isaacsim/ 则需要修改）
export ISAACSIM_ROOT=~/isaacsim

# cuRobo 子进程使用的 Python（Conda 环境 or 系统 Python 3.13）
# 如果使用 Conda：
export CUROBO_PYTHON=~/miniconda3/envs/curobo/bin/python3
# 如果使用系统 Python：
# export CUROBO_PYTHON=/usr/bin/python3
```

执行 `source ~/.bashrc` 生效。

> **其他路径变量**（若使用默认布局则无需设置）：
> ```bash
> export SIMFORGE_SCENE=<自定义场景路径>        # 默认用 scenes/main.usd
> export SIMFORGE_URDF_DIR=<URDF 目录>          # 默认用 robot/
> export SIMFORGE_CUROBO_CFG=<cuRobo yml 路径>  # 默认用 robot/jaka_minicobo_curobo.yml
> ```

### 5.2 验证配置

```bash
cd ~/simforge
~/isaacsim/kit/python/bin/python3 config.py
```

预期输出：

```
SimForge config — all paths OK:
  ISAACSIM_ROOT : /home/<user>/isaacsim
  SCENE_USD     : /home/<user>/simforge/scenes/main.usd
  ARM_URDF      : /home/<user>/simforge/robot/jaka_minicobo.urdf
  CUROBO_URDF   : /home/<user>/simforge/robot/jaka_minicobo_gripper.urdf
  CUROBO_CFG    : /home/<user>/simforge/robot/jaka_minicobo_curobo.yml
```

---

## 六、启动 Demo（tray_grasp_cycle）

### 6.1 一键启动

```bash
cd ~/simforge
bash demos/tray_grasp_cycle/launch.sh
```

`launch.sh` 会自动：
1. Kill 任何已运行的 Isaac Sim 实例
2. 设置必要的 `LD_LIBRARY_PATH`
3. 调用 `isaac-sim.sh --exec demos/tray_grasp_cycle/demo.py`

> **首次启动慢**：Isaac Sim 第一次运行需要编译 shader 缓存，约需 3–5 分钟。后续启动约 30–60 秒。

### 6.2 启动 + 录制日志

```bash
bash demos/tray_grasp_cycle/record.sh
```

日志保存到 `logs/tray_grasp_cycle/YYYYMMDD_HHMMSS.log`，自动生成报告到 `reports/`。

### 6.3 启动参数

`launch.sh` 可接受额外参数传给 Isaac Sim：

```bash
# 无头模式（无 GUI，服务器上运行）
bash demos/tray_grasp_cycle/launch.sh --no-window

# 指定不同场景
SIMFORGE_SCENE=~/my_custom_scene.usd bash demos/tray_grasp_cycle/launch.sh
```

---

## 七、Demo 运行说明

启动后 Isaac Sim GUI 会打开，右侧出现 **Tray Grasp Cycle** 监控面板（实时力/位移曲线）。

### 阶段流程

```
PAUSE（初始化）
  ↓ 自动触发
TO_PRE_L         左臂运动到托盘前方预位（Y+125mm）
  ↓
APPROACH_L       左臂沿世界Y轴线性逼近托盘耳部（80步，pos=0mm，fw=0°全程）
  ↓ 力传感器F ≥ 3N
CLOSE_GRIP_L     建立 FixedJoint 夹持，托盘切换为 Dynamic Body
  ↓
LIFT_L           举升托盘（+300mm Z）
  ↓
CARRY_L          左臂携托盘移动到交接位
  ↓
HANDOFF          右臂接收，建立 FixedJoint，左臂松手
  ↓
CARRY_DRYER      右臂送入烘箱
  ↓
RESET_SCENE      托盘自由落体后在原位置闪现新托盘
  ↓ 重复循环
```

### GUI 监控面板

- **L Force / R Force**：左右夹爪接触力（N）
- **Pad Y**：夹爪末端 Y 轴位置（m）
- **Approach Chart**：逼近阶段力–位移实时曲线

---

## 八、停止

**方法 1：关闭 Isaac Sim 窗口**（GUI 模式）

**方法 2：终端强制终止**

```bash
pkill -f "isaacsim/kit/kit"
# 若进程未退出：
ps aux | grep "isaacsim/kit/kit" | grep -v grep | awk '{print $2}' | xargs kill -9
```

**方法 3：Ctrl+C**（在 `record.sh` 终端中）

> **重要**：启动新 Demo 前务必先终止旧进程，Isaac Sim 同时只能运行一个实例。

---

## 九、常见问题

### 9.1 启动时报 `sem.carbonite-sharedmemory` 卡死

```bash
rm -f /dev/shm/sem.carbonite-sharedmemory
# 然后重新启动
```

### 9.2 启动时 `ld.so Assertion` 崩溃

原因：使用了错误的启动命令（直接调 `kit/kit` 而非 `isaac-sim.sh`）。  
解决：务必通过 `launch.sh` 或 `isaac-sim.sh` 启动，会自动设置必要的 `LD_LIBRARY_PATH`。

### 9.3 IK 求解报 workspace 错误

检查托盘位置是否在机械臂工作空间内：
- 托盘耳部 Y 坐标应在 `0.387–0.520m`（相对机器人基座）
- 检查 `scenes/main.usd` 中 `/World/Tray` 的初始位置

### 9.4 cuRobo 子进程超时

```bash
# 验证 cuRobo Python 路径
python3 -c "import curobo; print('OK')"
# 若 Conda 环境未激活，检查 _curobo_worker.py 顶部的 Python 路径
head -5 demos/tray_grasp_cycle/_curobo_worker.py
```

### 9.5 首次运行 cuRobo 很慢

cuRobo 首次运行会编译 CUDA kernel（约 2–5 分钟），后续运行有缓存，约 10–30 秒。

---

## 十、场景快照 / Checkpoint

更新 `scenes/main.usd` 以保存当前 Isaac Sim 场景状态：

```bash
cd ~/simforge
bash scenes/checkpoint.sh "描述本次改动的消息"
```

脚本会把 `~/isaacsim/playground/2026061100_main.usd` 拷贝到 `scenes/main.usd` 并 git commit。

---

## 十一、代码结构说明

| 文件 | 作用 |
|------|------|
| `config.py` | 路径解析入口，所有 demo 都 import 它 |
| `core/kinematics.py` | URDF 解析、FK、关节链工具 |
| `core/planning.py` | `solve_pad_pose_ik`：支持 jaw+forward 双轴约束的 IK 求解器 |
| `core/gripper.py` | EG2-4C2 夹爪 xform 驱动、pad 分离距离计算 |
| `core/scene_utils.py` | USD prim 操作工具（坐标变换、FixedJoint 创建等） |
| `demos/tray_grasp_cycle/demo.py` | 主循环状态机（~1300行） |
| `demos/tray_grasp_cycle/_curobo_worker.py` | cuRobo 子进程（Python 3.13/Warp 1.13.0） |

---

## 十二、里程碑记录

见 `milestones/INDEX.md`。
