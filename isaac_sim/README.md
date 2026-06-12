# Isaac Sim 仿真环境

> 辅助仿真方案 — 物理仿真、合成数据生成、USD 场景渲染

## 目录

```
isaac_sim/
├── cad_assets/                    # SolidWorks 导出的 STEP/STL 文件
│   ├── jinyu_lab_0526_full_visual.step          # 完整实验台+场景 (52MB)
│   ├── jinyu_lab_0526_full_visual-removed.step  # 移除目标物体后的实验台 (52MB)
│   ├── tray_for_foundationpose.STEP             # 被抓取托盘 (229KB)
│   └── tray_for_foundationpose.STL              # 托盘STL版 (133KB)
├── setup_lab_scene.py             # Isaac Sim 场景载入脚本
├── launch_lab.sh                  # 一键启动脚本
└── README.md
```

## 快速启动

```bash
# 空白场景 (大地平面)
bash /home/andyee/Developer/PG-JY/isaac_sim/launch_lab.sh

# 仓库场景 (带墙壁)
bash /home/andyee/Developer/PG-JY/isaac_sim/launch_lab.sh warehouse

# 或直接:
cd /home/andyee/isaacsim
SCENE_MODE=blank ./isaac-sim.sh --exec /home/andyee/Developer/PG-JY/isaac_sim/setup_lab_scene.py
```

## 场景说明

- **jinyu_lab_0526_full_visual-removed.step**: 90°实验台 + 桌面设备 (不含被抓取的 tray)
- **tray_for_foundationpose.STEP**: 被抓取目标，抓取点为两侧薄耳位置
- 桌面在 world 坐标系中位于 `(0, -0.5, 0.80)`, 与 cuRobo 规划场景一致

## CAD 导入流程

```
SolidWorks 装配体
  → 另存为 STEP (.step/.stp)
  → 放入 cad_assets/
  → Isaac Sim 启动时自动导入 (通过 setup_lab_scene.py)
  → 或在 GUI 中 File → Import → 选择 STEP
```

## 下一步

1. 在 Isaac Sim 中验证实验台正确显示
2. 导入 JAKA Minicobo 双臂 URDF
3. 关联 cuRobo 进行运动规划
4. 用 FoundationPose 估计托盘位姿 → cuRobo 规划抓取
