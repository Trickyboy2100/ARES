#!/bin/bash
# 启动 Isaac Sim 并加载金宇实验室场景
#
# 用法:
#   bash launch_lab.sh              # 空白场景
#   bash launch_lab.sh warehouse    # 仓库场景
#
set -e

ISAAC_DIR="/home/andyee/isaacsim"
SCRIPT="/home/andyee/Developer/PG-JY/isaac_sim/setup_lab_scene.py"

MODE="${1:-blank}"
echo "🚀 启动 Isaac Sim (场景: $MODE) ..."
echo "   实验台: jinyu_lab_0526_full_visual-removed.step"
echo "   托盘:   tray_for_foundationpose.STEP"
echo ""

# 修复 CUDA 库冲突: Isaac Sim 5.1 自带 CUDA 12.8+ 库,
# 需要优先于系统的 CUDA 12.1 被加载
CUDALIB="$ISAAC_DIR/exts/omni.isaac.ml_archive/pip_prebundle"
export LD_LIBRARY_PATH="$CUDALIB/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

cd "$ISAAC_DIR"
SCENE_MODE="$MODE" ./isaac-sim.sh --exec "$SCRIPT"
