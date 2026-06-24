# Milestone: FixedJoint Dual-Arm Handoff — 2026-06-25

Git commit: `1e313b4` — fix(simforge): restore FixedJoint grasp with branch-A linspace for home→preL

---

## 功能状态

**完全工作的双臂托盘搬运循环（7+ cycles 验证）：**

1. **L 臂夹取** — 耳片力停止（≥2N）→ FixedJoint 锁定 → 抬升最高 30cm
2. **双臂交接** — L 释放 FixedJoint，R carrier 夹住，托盘无缝转移
3. **R 臂运送** — cuRobo 规划路径送至烘干机方向 → 释放 → 复位
4. **循环** — 两臂回 home → PAUSE → 下一 cycle

---

## 关键验证数据

| 指标 | 数值 |
|------|------|
| 循环稳定性 | 7+ cycles 连续验证 |
| L 臂最大抬升 | 29.5~30.0 cm |
| tray_pos @ RELEASE_L | ~[-0.45, 0.76, 1.64] m（交接位）|
| tray free-fall from | ~[-1.1, 2.3, -15.2] m（烘干机方向）|
| 最大关节跳变 L | ≤ 0.039 rad |
| 最大关节跳变 R | ≤ 0.038 rad |

---

## 核心技术方案

### L 臂夹取（FixedJoint）
- `right_pad` 有 `PhysicsRigidBodyAPI + kinematicEnabled=True`
- 力停止后创建 `_GraspLock_L` FixedJoint（pad→tray）
- 托盘变 dynamic，FixedJoint 牵引跟随 arm FK 运动
- **关键**：home→preL 用 `np.linspace(q_zero, q_pre_L, 120)`（branch A），不用 cuRobo（cuRobo 落在 branch B，FixedJoint 在 B 分支静默失效）

### R 臂夹取（carrier 机制）
- `right_pad` 无 `PhysicsRigidBodyAPI`，无法直接用 FixedJoint
- 每帧移动隐形 kinematic carrier cube（`_GraspCarrier_R`）跟随 R 臂 FK
- 交接：carrier→tray FixedJoint（`_GraspLock_R`）

### 路径规划
- L 臂 home→preL：joint-space linspace（120 步）
- L 臂 approach：Y 轴线性 IK（从 q_pre_L 出发，branch A）
- R 臂 home→handoff、handoff→dryer：cuRobo 离线规划（persistent worker）
- 所有 retract/home 段：runtime linspace（从当前 q 到 q_zero）

---

## 文件列表

- `demo.py` — 主 demo 脚本（含完整状态机）
- `core/` — kinematics_probe, planning, gripper, scene_utils
- `main.usd` — 场景快照
