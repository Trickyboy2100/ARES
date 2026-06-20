# perception/mechmind — Mech-Eye Log S 相机接入

用于连接梅卡曼德 Mech-Eye Log S 相机，通过 Mech-Vision 获取抓取位姿，引导机器人执行抓取。

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `connect.py` | 相机连接、发现、单帧采集（彩色图 + 深度图 + 点云） |
| `grasp_client.py` | 从 Mech-Vision Robot Interface 接收抓取位姿，转换到机器人基坐标系 |
| `hand_eye_calib.py` | 手眼标定辅助（采集数据 + 求解 AX=XB） |
| `debug.py` | 一键诊断：网络 ping → 端口 → SDK → 采集帧率 → 点云质量 |

---

## 快速开始

### 1. 安装 SDK

```bash
pip install mecheye
# 文档：https://github.com/MechMindRobotics/mecheye_python_samples
```

### 2. 网络配置

- 相机默认 IP：`192.168.x.x`（出厂标签或 Mech-Eye Viewer 查看）
- 主机与相机需在**同一子网**
- 防火墙放行：UDP 5787（相机发现），TCP 5577（数据传输）

### 3. 连接与诊断

```bash
# 全面诊断
python3 debug.py --ip 192.168.x.x --vision-ip 192.168.x.x

# 仅连接并采集一帧
python3 connect.py --ip 192.168.x.x --save /tmp/mechmind_frame

# 列出网络中所有相机
python3 connect.py --list
```

### 4. 获取抓取位姿

前提：Mech-Vision 已启动对应工程，开启 Robot Interface（默认端口 50004）。

```bash
# 请求一次抓取位姿（相机坐标系）
python3 grasp_client.py --host 192.168.x.x

# 带手眼矩阵，转换到机器人基坐标系
python3 grasp_client.py --host 192.168.x.x --hand-eye hand_eye_result.json

# 持续轮询
python3 grasp_client.py --host 192.168.x.x --hand-eye hand_eye_result.json --loop
```

### 5. 手眼标定

```bash
python3 hand_eye_calib.py --board 7x6 --square 25
# 按提示移动机器人，采集 ≥10 个姿态，结果保存到 hand_eye_result.json

# 验证标定结果
python3 hand_eye_calib.py --verify hand_eye_result.json
```

---

## 与 demo 集成

`grasp_client.py` 返回的 `GraspPose` 对象携带机器人基坐标系下的 `position_m()` 和 `quaternion`，可直接传给 `core/planning.py` 的 IK 求解器：

```python
from perception.mechmind.grasp_client import MechVisionClient, load_hand_eye_matrix
import numpy as np

cam_to_base = load_hand_eye_matrix("perception/mechmind/hand_eye_result.json")
client = MechVisionClient(host="192.168.x.x")
client.connect()
poses = client.get_grasp_poses()
pick_xyz = poses[0].to_robot_base(cam_to_base).position_m()  # → numpy [x,y,z] in meters
```

---

## 常见问题

**发现不了相机**
- 检查子网：`ip addr` 确认主机 IP 与相机同段
- `ping 192.168.x.x` 通了再试 SDK

**采集超时**
- Mech-Eye Log S 曝光时间可能较长，增大 `--timeout`（`connect.py` 里 `Device` 构造参数）

**Mech-Vision 端口拒绝连接**
- 确认 Mech-Vision 主界面已点击"开始服务"并显示 Robot Interface 端口
- 检查端口号（Mech-Vision 2.x 默认 50004，旧版可能不同）
