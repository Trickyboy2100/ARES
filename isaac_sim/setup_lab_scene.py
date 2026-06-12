#!/usr/bin/env python3
"""
Isaac Sim — 金宇实验室场景载入脚本
=====================================

用法:
  # 方法1: GUI 中 Script Editor 打开运行
  # 方法2: 命令行启动时自动执行
  cd /home/andyee/isaacsim
  ./isaac-sim.sh --exec /home/andyee/Developer/PG-JY/isaac_sim/setup_lab_scene.py

  # 可选参数 (通过环境变量):
  SCENE_MODE=blank   ./isaac-sim.sh --exec ...   # 空白场景 (默认)
  SCENE_MODE=warehouse ./isaac-sim.sh --exec ... # 仓库场景
"""

import os, sys, time
import numpy as np

# ============================================================
#  配置
# ============================================================
STEP_LAB = "/home/andyee/Developer/PG-JY/isaac_sim/cad_assets/jinyu_lab_0526_full_visual-removed.step"
STEP_TRAY = "/home/andyee/Developer/PG-JY/isaac_sim/cad_assets/tray_for_foundationpose.STEP"
SCENE_MODE = os.environ.get("SCENE_MODE", "blank")  # "blank" | "warehouse"

# ============================================================
#  等待 Isaac Sim 就绪
# ============================================================
import omni
from omni.isaac.kit import SimulationApp

# 如果已经通过 --exec 启动, SimulationApp 已存在
# 否则创建新的
try:
    kit = omni.kit.app.get_app()
except:
    kit = None

print("[setup_lab] 等待 Isaac Sim 完全初始化...", flush=True)
import omni.timeline
import omni.usd
from pxr import UsdGeom, Gf, Sdf, UsdLux, Usd

# 等待至少 3 秒让所有扩展加载
time.sleep(3)

stage = omni.usd.get_context().get_stage()
if not stage:
    # 创建新场景
    stage = Usd.Stage.CreateNew("/tmp/lab_scene.usd")
    omni.usd.get_context().attach_stage(stage)

print(f"[setup_lab] 场景模式: {SCENE_MODE}", flush=True)


# ============================================================
#  辅助函数
# ============================================================
def make_box(path, pos, scale, color=(0.5, 0.5, 0.5)):
    """创建 Cube prim"""
    g = UsdGeom.Cube.Define(stage, path)
    g.AddScaleOp().Set(Gf.Vec3f(*scale))
    g.AddTranslateOp().Set(Gf.Vec3f(*pos))
    g.GetPrim().CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    ).Set([Gf.Vec3f(*color)])
    return g


def make_cylinder(path, pos, height, radius, color=(0.5, 0.5, 0.5)):
    """创建 Cylinder prim"""
    g = UsdGeom.Cylinder.Define(stage, path)
    g.AddScaleOp().Set(Gf.Vec3f(radius, radius, height))
    g.AddTranslateOp().Set(Gf.Vec3f(*pos))
    g.GetPrim().CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    ).Set([Gf.Vec3f(*color)])
    return g


def add_ground():
    """添加大地平面"""
    make_box("/World/Ground", (0, 0, -0.01), (10, 10, 0.01), (0.6, 0.6, 0.6))
    print("[setup_lab] 地面已添加", flush=True)


def add_lighting():
    """添加基础灯光"""
    dl = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
    dl.CreateIntensityAttr(800)
    dl.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))

    dl2 = UsdLux.DistantLight.Define(stage, "/World/Sun")
    dl2.CreateIntensityAttr(3000)
    dl2.CreateAngleAttr(1.5)
    dl2.AddRotateYOp().Set(45)
    dl2.AddRotateXOp().Set(-30)
    print("[setup_lab] 灯光已添加", flush=True)


# ============================================================
#  场景构建
# ============================================================
if SCENE_MODE == "warehouse":
    # ── 简易仓库 ──
    S, H = 4.0, 3.0  # 房间半边长, 墙高
    make_box("/World/Floor", (0, 0, -0.02), (S, S, 0.02), (0.55, 0.55, 0.55))
    walls = [
        ("WallN", (0, -S, H / 2), (S, 0.02, H / 2)),
        ("WallS", (0, S, H / 2), (S, 0.02, H / 2)),
        ("WallW", (-S, 0, H / 2), (0.02, S, H / 2)),
        ("WallE", (S, 0, H / 2), (0.02, S, H / 2)),
    ]
    for name, pos, scale in walls:
        make_box(f"/World/{name}", pos, scale, (0.45, 0.47, 0.50))
    print("[setup_lab] 仓库场景已构建", flush=True)
else:
    # ── 空白场景 + 大地平面 ──
    add_ground()

add_lighting()


# ============================================================
#  导入实验台 STEP 文件
# ============================================================
print(f"[setup_lab] 导入实验台: {STEP_LAB}", flush=True)

try:
    import omni.kit.commands

    # Isaac Sim 5.1 使用 CreatePayloadCommand 或直接操作 USD stage
    # 方法1: 直接用 USD stage 的 sublayer 引用（仅 USD 文件）
    # 方法2: CreateReferenceCommand 的参数是 usd_path 不是 url
    omni.kit.commands.execute(
        "CreateReferenceCommand",
        usd_path=STEP_LAB,
        prim_path="/World/LabBench",
    )
    print("[setup_lab] ✅ 实验台 STEP 已导入 (CreateReference)", flush=True)

except Exception as e1:
    print(f"[setup_lab] ⚠️ CreateReference(usd_path) 失败: {e1}", flush=True)

    try:
        # 方法2: 尝试 asset_path 参数
        omni.kit.commands.execute(
            "CreateReferenceCommand",
            asset_path=STEP_LAB,
            prim_path="/World/LabBench",
        )
        print("[setup_lab] ✅ 实验台 STEP 已导入 (asset_path)", flush=True)
    except Exception as e2:
        print(f"[setup_lab] ⚠️ CreateReference(asset_path) 也失败: {e2}", flush=True)

        try:
            # 方法3: 使用 omni.client 直接复制并引用
            import omni.client

            result, src_path = omni.client.copy(
                STEP_LAB, "/World/LabBench"
            )
            print(f"[setup_lab] ⚡ omni.client.copy: {result}", flush=True)
        except Exception as e3:
            print(f"[setup_lab] ❌ 所有自动导入方法都失败", flush=True)
            print(
                "[setup_lab] 💡 请通过 GUI 手动导入: File → Import → 选择 STEP",
                flush=True,
            )
            print(
                f"[setup_lab] 💡 文件: {STEP_LAB}",
                flush=True,
            )


# ============================================================
#  导入托盘 (被抓取目标)
# ============================================================
print(f"[setup_lab] 导入托盘: {STEP_TRAY}", flush=True)

# 先检查托盘 STL 版本 (更易导入)
TRAY_STL = STEP_TRAY.replace(".STEP", ".STL")
tray_file = TRAY_STL if os.path.exists(TRAY_STL) else STEP_TRAY

try:
    omni.kit.commands.execute(
        "CreateReferenceCommand",
        url=tray_file,
        default_prim_path="/World/Tray",
    )

    # 托盘放置位置: 桌面中心附近, 与 cuRobo 场景一致
    # cuRobo 中桌面在 world (0, -0.5, 0.80)
    tray_prim = stage.GetPrimAtPath("/World/Tray")
    if tray_prim:
        xform = UsdGeom.Xformable(tray_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.50, 0.83))

    print("[setup_lab] ✅ 托盘已导入", flush=True)
except Exception as e:
    print(f"[setup_lab] ⚠️ 托盘导入失败 (可手动添加): {e}", flush=True)


# ============================================================
#  调整视角
# ============================================================
try:
    # 设置默认相机位置看向实验台
    import omni.kit.viewport_legacy as vp

    viewport = vp.get_viewport_interface()
    if viewport:
        cam = viewport.get_viewport_window()
        if cam:
            # 面向桌面: 前方偏高位置
            cam.set_camera_position((2.0, -1.5, 1.8))
            cam.set_camera_target((0.0, -0.50, 0.80))
    print("[setup_lab] 视角已调整", flush=True)
except Exception as e:
    print(f"[setup_lab] 视角调整跳过: {e}", flush=True)


# ============================================================
#  设置物理场景 (可选, 如需要)
# ============================================================
# ============================================================
#  设置物理场景 (可选, 如需要)
# ============================================================
try:
    from pxr import UsdPhysics
    physx_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physx_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    physx_scene.CreateGravityMagnitudeAttr().Set(9.81)
    print("[setup_lab] 物理场景已配置", flush=True)
except Exception as e:
    print(f"[setup_lab] 物理配置跳过: {e}", flush=True)

print("\n" + "=" * 50, flush=True)
print("[setup_lab] ✅ 场景初始化完成!", flush=True)
print(f"[setup_lab]  实验台: {STEP_LAB}", flush=True)
print(f"[setup_lab]  托盘:   {STEP_TRAY}", flush=True)
print(f"[setup_lab]  场景模式: {SCENE_MODE}", flush=True)
print(
    "[setup_lab]  💡 如果实验台未显示, File → Import → 手动选择 STEP",
    flush=True,
)
print("=" * 50 + "\n", flush=True)
