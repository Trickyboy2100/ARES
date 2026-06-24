# tray_grasp_cycle — 双臂托盘搬运循环 Demo

左臂从货架夹取托盘耳片 → 抬升 → 交给右臂 → 右臂送至烘干机 → 两臂回 home → 循环。  
基于 **FixedJoint 物理夹取 + cuRobo 运动规划 + carrier 交接机制**。

---

## 快速启动

```bash
# 从仓库根目录（自动 kill 旧进程）
bash isaac_sim/simforge/demos/tray_grasp_cycle/launch.sh

# 带日志捕获 + 自动摘要报告
bash isaac_sim/simforge/demos/tray_grasp_cycle/record.sh
```

或手动启动：

```bash
pkill -f "isaacsim/kit/kit" 2>/dev/null; sleep 2
ISAAC_NVLIBS=$(find ~/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia -type d -name "lib" | tr '\n' ':')
export LD_LIBRARY_PATH="${ISAAC_NVLIBS}${LD_LIBRARY_PATH:-}"
cd ~/isaacsim && nohup ./isaac-sim.sh --exec /path/to/demo.py > /tmp/tgc.log 2>&1 &
```

---

## 状态机

```
SETTLE (120 fr)
  └─ 物理稳定；双臂保持 q=0

READ_TRAY / IK_PLAN
  └─ 读取托盘 AABB，解算 pre/approach/pick 路径
  └─ cuRobo 规划 R 臂：home→handoff、handoff→dryer

CYCLE（无限循环）:
  TO_PRE_L   → L 臂到达预抓取位（joint-space linspace，branch A）
  APPROACH_L → L 臂 Y 轴线性推进；接触力 ≥ 2N → FORCE STOP
  CLOSE_GRIP_L → 夹爪闭合
  LIFT_L     → L 臂抬升 ~30cm；FixedJoint 牵引托盘 6DOF 跟随
  CARRY_L    → L 臂携盘运动至交接位（cuRobo runtime replan）
  R_APPROACH → R 臂 carrier 靠近托盘侧面
  CLOSE_GRIP_R → R carrier FixedJoint 创建
  RELEASE_L  → L FixedJoint 释放，R 接手
  CARRY_DRYER → R 臂送托盘至烘干机方向（cuRobo）
  HOLD_DRYER → 等待
  RELEASE_R  → R FixedJoint 释放，托盘自由落体
  HOME_R     → R 臂回 home（runtime linspace）
  RETRACT_L  → L 臂回 home（runtime linspace）
  RESET_SCENE → 托盘复位到货架
  PAUSE      → 90 fr 等待，进入下一 cycle
```

---

## 夹取机制

### L 臂 — FixedJoint（直接）

`LEFT_GR/right_pad` 有 `PhysicsRigidBodyAPI + kinematicEnabled=True`。

```
APPROACH_L 力停止
  → _set_tray_kinematic(True)          # 防止夹爪关闭时弹飞
CLOSE_GRIP_L 结束
  → _create_grasp_joint(pad, tray)     # FixedJoint: pad → tray
  → _set_tray_kinematic(False)         # tray 变 dynamic
LIFT_L / CARRY_L
  → arm FK 驱动 pad（kinematic）
  → FixedJoint 牵引 tray 做 6DOF 跟随
```

**关键：home→preL 必须用 linspace，不能用 cuRobo。**  
cuRobo 落在 IK branch B；approach_L 从 branch B 出发会导致 q_contact_L 也在 branch B，  
FixedJoint 在 branch B 静默失效（pad 世界坐标相同，但 PhysX 无法满足约束）。

### R 臂 — carrier 机制

`RIGHT_GR/right_pad` 无 `PhysicsRigidBodyAPI`，直接 FixedJoint 失效。

```
每帧: carrier_cube 位置 = R 臂 FK pad 位置（carrier 是 kinematic rigid body）
CLOSE_GRIP_R
  → _set_tray_kinematic(True)
  → FixedJoint: carrier → tray
  → _set_tray_kinematic(False)
CARRY_DRYER
  → carrier 持续跟随 R 臂 FK → tray 跟随
```

---

## 路径规划

| 段 | 方法 |
|----|------|
| L 臂 home→preL | `np.linspace(q_zero, q_pre_L, 120)`（branch A） |
| L 臂 approach | Y 轴线性 IK（`_linear_y_path`，从 q_pre_L 出发） |
| L 臂 carry | cuRobo async replan（从 q_lift[-1] → handoff FK） |
| R 臂 home→handoff | cuRobo 离线（启动时规划） |
| R 臂 handoff→dryer | cuRobo 离线（启动时规划） |
| L 臂 retract→home | `np.linspace(q_L_now, q_zero, 120)` |
| R 臂 home（dryer后）| `np.linspace(q_R_now, q_zero, 120)` |

cuRobo 使用 **persistent subprocess worker**（`_curobo_worker.py`），通过 stdin/stdout JSON 通信，避免在 Isaac Sim Python 环境中直接加载 PyTorch。

---

## GUI — 双臂监控面板

窗口：**"Dual Arm Grasp Monitor"**（800 × 720 px）

| 区域 | 内容 |
|------|------|
| 左臂 | pad EE XYZ · 接触力 · 夹爪角度/间距 · 力历史曲线 · pad-Y 逼近曲线 · 抬升高度 |
| 右臂 | pad EE XYZ · carrier 位置 · R_pad force · 状态标签 |
| 底部 | 当前阶段 · cycle 计数 · 关键常量 |

---

## 关键 USD Prim 路径

| 路径 | 作用 |
|------|------|
| `/World/robot/jaka_minicobo_left` | L 臂根节点 |
| `/World/robot/jaka_minicobo_right` | R 臂根节点 |
| `/World/Tray` | 托盘刚体（抓取期间 dynamic） |
| `LEFT_GR/right_pad` | L 夹爪 pad（有 RigidBodyAPI） |
| `/World/_GraspLock_L` | L 臂 FixedJoint |
| `/World/_GraspCarrier_R` | R 臂 carrier cube（1cm kinematic） |
| `/World/_GraspLock_R` | R 臂 FixedJoint（carrier→tray） |

---

## 验证结果（2026-06-25，7+ cycles）

| 指标 | 数值 |
|------|------|
| IK L_pre 位置误差 | 0.0 mm |
| L 臂力停止触发 | ≥ 2 N（pad_y ≈ 0.4207 m） |
| 最大抬升高度 | ~30 cm |
| tray 交接位置 | [-0.45, 0.76, 1.64] m |
| tray 烘干机方向 | free-fall from ~[-1.1, 2.3, -15.2] m |
| 最大关节跳变 L | ≤ 0.039 rad |
| 最大关节跳变 R | ≤ 0.038 rad |
| 循环稳定性 | 7+ cycles 连续验证 |

---

## 调试技巧

```bash
# 实时查看阶段转换
tail -f /tmp/tgc.log | grep "→ "

# 查看关节配置（每30帧）
tail -f /tmp/tgc.log | grep "TGC-JOINT"

# 查看关节跳变
tail -f /tmp/tgc.log | grep "TGC-JUMP"

# 查看托盘位置 + 抬升高度
tail -f /tmp/tgc.log | grep "tray_pos\|lift="
```

---

## 已知限制

- 托盘复位是硬编码位置（非 ArUco 视觉定位）
- 烘干机「放入」仅靠 free-fall，无精确定位
- L 臂夹爪深度偏深约 1-2cm（`PICK_Y_OFFSET=-0.008`，不影响抓取）
