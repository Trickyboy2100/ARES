# Gripper Force Demo

双臂 EG2-4C2 夹爪夹球力实时可视化 demo。

两个夹爪从全开位自动合拢，接触橙色球体后以软接触弹簧模型计算夹紧力，Isaac Sim 内弹出实时力显示面板。

---

## 目录结构

```
gripper_force_demo/
├── demo.py      — 主脚本（运行入口）
├── launch.sh    — 一键启动脚本（含进程清理）
├── scene.usd    — 完整仿真场景（双臂 JAKA + EG2 + 托盘 + 实验台）
└── README.md    — 本文档
```

---

## 快速启动

```bash
cd /path/to/PG-JY
bash isaac_sim/simforge/demos/gripper_force_demo/launch.sh
```

`launch.sh` 会自动：
1. Kill 已有的 Isaac Sim 进程
2. 设置 `LD_LIBRARY_PATH`（nvjitlink，避免 ld.so 崩溃）
3. 以 `--exec` 模式启动 Isaac Sim 并运行 `demo.py`

手动启动等价命令：

```bash
pkill -f "isaacsim/kit/kit" 2>/dev/null; sleep 2
CUDALIB=~/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle
LD_LIBRARY_PATH=$CUDALIB/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
  ~/isaacsim/isaac-sim.sh --exec \
  isaac_sim/simforge/demos/gripper_force_demo/demo.py
```

---

## Demo 运行流程

| 阶段 | 帧数 | 时长 | 内容 |
|------|------|------|------|
| **SETTLE** | 0–60 | 1 s | 双臂 FK 归零，夹爪全开 35.65°，橙色球体生成于 pad 中点 |
| **CLOSING** | 60–137 | ~1.3 s | 夹爪线性合拢，球体未接触，F = 0 |
| **CONTACT** | 137–210 | ~1.2 s | 首次接触，力从 0.8 N 线性升至 28.4 N，轻微视觉穿模 |
| **HOLD** | 210+ | ∞ | 保持全夹紧，pen ≈ 2.1 mm，F ≈ 28.4 N，窗口不退出 |

---

## 实时 UI 面板（Force Monitor）

Isaac Sim 窗口内弹出 "Gripper Force Monitor" 面板，包含：

```
┌─ GRIPPER FORCE MONITOR ──────────────── ● CONTACT ─┐
│ LEFT  GRIPPER   14.9°  │  F = 28.4 N  ↓2.10mm      │
│ ████████████████████░░░░░░░░░░░░░░░░░░  force bar   │
│ ██████░░  penetration bar (cyan→yellow)             │
│ ___/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  force history (350 frames) │
│                                                     │
│ RIGHT GRIPPER   14.9°  │  F = 28.4 N  ↓2.10mm      │
│  (same layout)                                      │
│                                                     │
│ sphere ø=24mm │ contact≈17.9° │ K_drive=3 K_sphere=15 │
└─────────────────────────────────────────────────────┘
```

- **力条**：绿→黄→红，满量程 50 N
- **穿模条**（仅 CONTACT 阶段显示）：青→黄，满量程 5 mm
- **历史曲线**：最近 350 帧的力值

---

## 物理模型

### 两弹簧平衡

夹爪驱动弹簧（K_drive）和球体接触弹簧（K_sphere）在 joint 处共享平衡：

```
actual = (K_drive × drive + K_sphere × contact) / (K_drive + K_sphere)
torque = K_drive × (actual − drive)
force  = 2 × torque / lever_arm
```

- `drive < contact_angle` 时 actual 被球体推开，产生夹紧力
- 视觉穿模深度 ≈ `K_drive / (K_drive + K_sphere) × Δangle_m`
- 此模型不依赖 PhysX 接触，纯解析计算，帧率稳定

### 接触几何（EG2-4C2）

```
pad_separation(a) = 2 × |−0.04 + 0.0223·cos(a) − 0.035591·sin(a)|   [m]
PAD_INNER_FACE    = 17.7 mm   (pad mesh 接触面到 prim origin 的距离)
contact_angle     = angle_for_pad_separation(2 × (R + 17.7 mm))
```

行程范围：`a = 0` → 35.4 mm 间距（全闭）；`a = 35.65°` → 85.4 mm（全开）

最大可夹球直径 = 85.4 − 35.4 = **50 mm**（demo 使用 24 mm）

### 可调参数（`demo.py` 顶部）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `SPHERE_RADIUS_M` | 0.012 m | 球半径，24 mm；最大 25 mm |
| `GRIPPER_DRIVE_STIFFNESS` | 3.0 N·m/rad | 驱动弹簧刚度 |
| `SPHERE_STIFFNESS` | 15.0 N·m/rad | 接触弹簧刚度（越大穿模越少、力越大）|
| `LEVER_ARM_M` | 0.055 m | 力矩臂，影响 torque→force 换算 |
| `SETTLE_FRAMES` | 60 | 静置帧数 |
| `CLOSE_RAMP_FRAMES` | 150 | 合拢用时帧数 |

---

## 场景说明

`scene.usd`：双臂 JAKA MiniCobo + Inspire EG2-4C2 + 托盘 + 实验台  
原始来源：`~/isaacsim/playground/2026061100_main.usd`（快照于 2026-06-13）

EG2 关节在场景中以 `active=False` 禁用了 PhysX drive，demo 全程使用 xform FK 控制，力/穿模由解析模型给出。

---

## 依赖

- Isaac Sim 5.1.0-rc.19，安装于 `~/isaacsim`
- `simforge/core/`：`kinematics.py`、`gripper.py`、`planning.py`、`scene_utils.py`
- `jaka_ros2/src/jaka_description/urdf/jaka_minicobo.urdf`（arm FK 用）
- Python 包：`numpy`、`scipy`（Isaac Sim 内置）
