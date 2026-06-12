# PG-JY Project Instructions

## 原则性指令

1. **不可以假抓** — 只有 `force_stop_step is not None`（物理接触已确认）后才允许建立 FixedJoint。
2. **夹爪必须垂直于地面** — EG2 X 轴 = 世界 -Z 方向，夹爪在 Z 方向闭合。
3. **严禁修改任何 milestone 目录** — `isaac_sim/playground_dual_arm_control/milestones/` 下所有内容只读。
4. **同时只保持一个 Isaac Sim 窗口** — 启动新场景前先 kill 旧进程。
5. **测试尽量用 GUI** — 优先 `isaac-sim.sh --exec` 启动，不用 `python.sh`（后者会触发 Storm 渲染器）。

---

## 工作目录结构

| 路径 | 说明 |
|------|------|
| `isaac_sim/playground_dual_arm_control/` | 原 playground（勿删）|
| `isaac_sim/playground_dual_arm_control_v2/` | 新 playground（当前开发主线）|
| `isaac_sim/playground_dual_arm_control/milestones/` | 只读里程碑存档 |
| `isaac_sim/cad_assets/` | 零件 CAD 文件 |

**当前主场景**：`/home/andyee/isaacsim/playground/2026061100_main.usd`
（来源：`~/Documents/isc/2026061000.usd` 的副本，已隐藏 RgbCamera / DepthCamera）

---

## Tray（托盘）零件分析

### 物理尺寸（来自 CAD/STL，`isaac_sim/cad_assets/tray.STEP / tray.STL`）

| 维度 | 原始 STL 坐标系 | 场景世界坐标系（settled） |
|------|-----------------|---------------------------|
| 长边（long） | X = 179.5 mm | Y 方向 ≈ 179.5 mm |
| 短边（wide） | Z = 99.5 mm | X 方向 ≈ 99.5 mm |
| 体高（body） | Y = 30 mm | Z 方向 ≈ 30 mm |

- STL 三角面数：2722
- CAD 文件别名：`tray_for_foundationpose.STEP/STL`（尺寸相同，同一零件）

### 抓取耳结构

- **两侧**（+Y 侧 / -Y 侧）各有一片 **2 mm 薄翼**，作为抓取耳（Ear）
- **抓取方向**：沿长边（世界 Y 方向，180 mm 方向）；夹爪在 Z 方向（垂直地面）闭合夹持耳片
- GraspFrames（USD 中的抓取参考帧）位于主体顶面（Z≈1.011 m）**上方约 130 mm（Z≈1.140 m）**
  - ⚠️ **该 Z 坐标不适用于垂直夹爪（EG2 X=world -Z）+ 从+Y接近的抓取方式**
  - 垂直夹爪时，EE 应位于 **托盘主体中心 Z ≈ 0.996 m**（而非 GraspFrame Z）
  - 原因：EG2 X=world -Z 时两片 pad 在世界 Z 方向上下各偏 17.7 mm；EE 在 Z=0.996 m 时 pad 恰好夹住 30 mm 高托盘体
  - GraspFrame Z=1.140 m 可能为其他抓取方式（如水平夹爪向下伸手）预留的参考帧

### USD Prim 层级

```
/World/Tray                          (Xform, RigidBodyAPI)
  /World/Tray/Mesh                   (Mesh,  CollisionAPI, approximation=none)
  /World/Tray/CollisionProxies/
    Body                             (Cube 碰撞代理)
    YPlusEar                         (Cube 碰撞代理)
    YMinusEar                        (Cube 碰撞代理)
  /World/Tray/GraspFrames/
    YPlusEar                         (Xform 抓取参考坐标系，+Y 侧)
    YMinusEar                        (Xform 抓取参考坐标系，-Y 侧)
```

### 场景摆放位置（`2026061100_main.usd`，物理稳定后）

**托盘整体 Bounding Box（世界坐标）**

| | min (m) | max (m) |
|--|---------|---------|
| X | 0.023 | 0.1225 |
| Y | 0.2295 | 0.409 |
| Z | 0.9811 | 1.0111 |

- 质心：`[0.0728, 0.3193, 0.9961]` m
- 托盘 Xform 初始平移（未落地）：`(0.1133, 0.2870, 1.4441)` m
- 初始旋转：rotateZ=90°, rotateY=0°, rotateX=90°

**GraspFrame 世界坐标（物理稳定后，Z 高于主体）**

| 抓取点 | 世界坐标 (m) |
|--------|-------------|
| YPlusEar  | `[0.0778, 0.3812, 1.1403]` |
| YMinusEar | `[0.0778, 0.2397, 1.1403]` |

两帧 Z 相同（1.1403 m），高于主体约 130 mm。

### 物理配置

- `/World/Tray`：`RigidBodyAPI` = True（动力学刚体）
- `/World/Tray/Mesh`：`CollisionAPI` = True，approximation = none（精确碰撞）
- 碰撞代理（Body / YPlusEar / YMinusEar）为 Cube 替代几何体

---

## 抓取物理设计（垂直夹爪 + 从+Y接近）

### 夹爪开合分析（EG2-4C2）

| 动作 | 关节角 (rad) | pad 中心间距 |
|------|-------------|-------------|
| 全闭 | 0.000 | 35.4 mm（物理最小值，**不可闭合到 2 mm**）|
| 50mm 开 | 0.1945 | 50.0 mm |
| 全开 | 0.6241 | 85.4 mm |

**关节角 = 间距对应角度公式**（来自 EG2 FK 链）：

```python
x_pad(a) = -0.04 + 0.0223*cos(a) - 0.035591*sin(a)
separation_m(a) = 2 * |x_pad(a)|
```

用 `angle_for_pad_separation_m(gap_m)` 函数（见 `gui_left_arm_ear_grasp_lift_demo.py`）计算任意间距对应的角度。

### 接触物理机制

**2mm 薄翼的接触方向是 Y（进入方向），不是 Z（开合方向）**：
- pad 从 +Y 侧压入耳面（法向力 ∥ Y 方向）
- 摩擦力 ∥ Z 方向 → 抵抗重力，托起托盘
- 夹爪闭合（Z 方向）的作用：把两片 pad 分别定位在托盘体**上方和下方**，使两片 pad 均可压到耳面

### contact box 参数（pad 局部坐标系）

| 参数 | 值 | 含义 |
|------|-----|------|
| `cx` | ±0.0035 m | 向夹心偏移（EG2 X = world -Z） |
| `cz` | 0.012 m | 沿手指方向（world -Y）定位，与 pick Y 偏移一致 |
| `hx` | 0.020 m | **Z 方向半跨度 40mm，覆盖托盘全高 30mm** |
| `hy` | 0.006 m | X 方向半跨度 12mm |
| `hz` | 0.008 m | Y 方向半跨度 16mm，覆盖 2mm 耳面有余量 |

### EE 目标位置（垂直夹爪抓取时）

```
EE_X ≈ 0.0778 m（托盘 X 中心）
EE_Y = ear_Y + pick_offset（pick 时偏移 0.012 m）
EE_Z = tray_body_center_Z ≈ 0.996 m  ← 关键！不用 GraspFrame Z=1.140
```

### force_stop + FixedJoint 约束

1. 接近时监测托盘 Y 位移（> 1 mm → 确认接触，设 `force_stop_step`）
2. 确认后关闭夹爪（30 帧，0.5 s）
3. 创建 FixedJoint（仅在 `force_stop_step is not None` 后执行）
4. 沿 +Z 提升

主脚本：`isaac_sim/playground_dual_arm_control_v2/scripts/gui_left_arm_ear_grasp_lift_demo.py`
