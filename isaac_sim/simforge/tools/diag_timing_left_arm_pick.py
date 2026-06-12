#!/usr/bin/env python3
"""
DIAGNOSTIC WRAPPER — 不改业务代码，只加精确计时
================================================
对 gui_left_arm_tray_pick_to_chest_demo.py 的所有规划步骤做毫秒级计时，
分析 1 分钟延迟到底花在哪里。

用法:
  cd /home/andyee/Developer/PG-JY/isaac_sim/playground_dual_arm_control
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV ... \
    /home/andyee/isaacsim/python.sh scripts/diag_timing_left_arm_pick.py
"""


# ── SimForge path injection ───────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_HERE = str(_Path(__file__).resolve().parent)
_SIMFORGE = str(_Path(__file__).resolve().parents[1])
_CORE = str(_Path(__file__).resolve().parents[1] / "core")
for _p in (_HERE, _SIMFORGE, _CORE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import time
import sys

# ═══════════════════════════════════════════════════════════════════
# Step 1: Monkey-patch scipy.optimize.least_squares 计时
# ═══════════════════════════════════════════════════════════════════
_original_least_squares = None
_timing_log: list[dict] = []


def _patched_least_squares(fun, x0, **kwargs):
    """包装 scipy.optimize.least_squares，记录每次调用的耗时和迭代次数"""
    global _original_least_squares, _timing_log
    t0 = time.perf_counter()
    result = _original_least_squares(fun, x0, **kwargs)
    elapsed = time.perf_counter() - t0
    x0_str = f"[{x0[0]:.2f},{x0[1]:.2f},...]" if not hasattr(x0, '__iter__') or len(x0) > 0 else "unknown"
    _timing_log.append({
        "x0_first": float(x0[0]) if hasattr(x0, '__iter__') and len(x0) > 0 else 0.0,
        "nfev": result.nfev if hasattr(result, 'nfev') else -1,
        "njev": result.njev if hasattr(result, 'njev') else -1,
        "status": result.status if hasattr(result, 'status') else -1,
        "cost": float(result.cost) if hasattr(result, 'cost') else -1,
        "success": bool(result.success) if hasattr(result, 'success') else False,
        "elapsed_ms": elapsed * 1000,
    })
    return result


# 在 import 任何业务代码之前完成 patch
from scipy.optimize import least_squares
_original_least_squares = least_squares

import scipy.optimize
scipy.optimize.least_squares = _patched_least_squares
# 也 patch 可能被直接引用的 _minpack
if hasattr(scipy.optimize, '_minpack'):
    import scipy.optimize._minpack as _mp


# ═══════════════════════════════════════════════════════════════════
# Step 2: 导入业务代码（现在所有 least_squares 调用都会被计时）
# ═══════════════════════════════════════════════════════════════════
print("[DIAG] ╔═══════════════════════════════════════╗", flush=True)
print("[DIAG] ║  LEFT-ARM-PICK 规划性能诊断            ║", flush=True)
print("[DIAG] ╚═══════════════════════════════════════╝", flush=True)
print("[DIAG] scipy.optimize.least_squares 已 patch", flush=True)

# kinematics_probe 可以在 pxr 不可用时导入（有 try/except 保护）
from kinematics_probe import (
    ARM_JOINTS, DEFAULT_ARM_URDF, chain_to_link, fk, load_joints,
)
# make_tray_handoff_curobo_demo 依赖 pxr，必须在 SimulationApp 之后导入
# 延迟到 run_diagnostics() 内部 import

# ═══════════════════════════════════════════════════════════════════
# Step 3: 复制 build_left_arm_path 逻辑，加计时
# ═══════════════════════════════════════════════════════════════════
from pathlib import Path
import numpy as np
import json

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_URDF = Path(DEFAULT_ARM_URDF)

TARGET_UP_WORLD = np.array([1.0, 0.0, 0.0], dtype=float)
TARGET_FORWARD_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)
CHEST_UP_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)
CHEST_FORWARD_WORLD = np.array([-1.0, 0.0, 0.0], dtype=float)

LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]


def joint_limits(chain):
    lower = []
    upper = []
    for joint in chain:
        if joint.name in ARM_JOINTS:
            lower.append(-np.pi if joint.lower is None else float(joint.lower))
            upper.append(np.pi if joint.upper is None else float(joint.upper))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def run_diagnostics():
    """启动 SimulationApp 后，纯测 IK 规划耗时"""
    # ── 先启动 Isaac Sim（需要 pxr 模块）──
    print("[DIAG] 启动 SimulationApp...", flush=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 640, "height": 480})

    # ── 延迟导入依赖 pxr 的模块 ──
    from make_tray_handoff_curobo_demo import (
        constrained_pose_path,
        constrained_pose_ramp_path,
        fallback_path,
        solve_pad_pose_ik,
    )

    import omni.usd
    print("[DIAG] ═══ 开始诊断 ═══", flush=True)

    # ── 加载运动学 ──
    t0 = time.perf_counter()
    arm_joints = load_joints(DEFAULT_URDF)
    chains = {name: chain_to_link(arm_joints, "Link_0", name) for name in LINK_NAMES}
    link6_chain = chains["Link_6"]
    lower, upper = joint_limits(link6_chain)
    print(f"[DIAG] URDF 加载: {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    # ── 模拟 USD 中读取的变换（硬编码典型值，跳过 Isaac Sim 初始化）──
    # 这些值来自实际运行时 USD stage 的 XformCache 查询
    base_world = np.eye(4)
    base_world[:3, 3] = [-0.1414, 0.0, 1.30]  # 左肩位置
    link6_to_pad = np.eye(4)
    link6_to_pad[:3, 3] = [0.0, 0.0, 0.1]  # Link_6 → pad midpoint 偏移

    ear_center = np.array([0.0775, 0.50, 0.83], dtype=float)
    settled_center = ear_center.copy()
    chest_target = np.array([0.07, 0.55, 1.182], dtype=float)
    lift_m = 0.07

    seed_bank = [
        np.zeros(6),
        np.array([1.0,  0.5,  0.4, -1.1, -0.2, -0.4], dtype=float),
        np.array([1.4,  0.6,  0.4, -1.1, -0.2, -0.4], dtype=float),
        np.array([0.8,  0.8,  0.2, -1.2,  0.1, -0.2], dtype=float),
        np.array([0.9,  0.5,  0.8, -0.8,  1.4,  1.5], dtype=float),
        np.array([1.1,  0.4,  0.7, -0.9,  1.5,  1.6], dtype=float),
        np.array([0.7,  0.7,  0.6, -1.0,  1.3,  1.4], dtype=float),
    ]

    pre  = ear_center + np.array([0.0,  0.125, 0.0],    dtype=float)
    pick = ear_center + np.array([0.0,  0.012, 0.0],    dtype=float)
    lift = pick       + np.array([0.0,  0.0,   lift_m], dtype=float)

    # ════════════════════════════════════════════════════════════
    # Phase 1: q_pre IK (1次 solve_pad_pose_ik)
    # ════════════════════════════════════════════════════════════
    _timing_log.clear()
    t_phase = time.perf_counter()
    print("[DIAG] ── Phase 1: 求解 pregrasp IK ──", flush=True)
    q0 = np.zeros(6)
    q_pre, *_ = solve_pad_pose_ik(
        link6_chain, lower, upper, base_world, link6_to_pad, pre,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD, [q0] + seed_bank,
    )
    elapsed_phase1 = (time.perf_counter() - t_phase) * 1000
    n_calls_phase1 = len(_timing_log)
    total_nfev_phase1 = sum(e["nfev"] for e in _timing_log)
    print(f"[DIAG]   Phase 1 完成: {elapsed_phase1:.0f} ms | {n_calls_phase1} 次 least_squares | {total_nfev_phase1} 总函数评估", flush=True)

    # ════════════════════════════════════════════════════════════
    # Phase 2: fallback_path (纯插值, 无 IK)
    # ════════════════════════════════════════════════════════════
    _timing_log.clear()
    t_phase = time.perf_counter()
    print("[DIAG] ── Phase 2: fallback_path (q0→q_pre 插值) ──", flush=True)
    path_to_pre = fallback_path(q0, q_pre, count=91)
    elapsed_phase2 = (time.perf_counter() - t_phase) * 1000
    n_calls_phase2 = len(_timing_log)
    print(f"[DIAG]   Phase 2 完成: {elapsed_phase2:.0f} ms | {n_calls_phase2} 次 least_squares (应为0)", flush=True)

    # ════════════════════════════════════════════════════════════
    # Phase 3: constrained_pose_path (pre→pick, 81 waypoints)
    # ════════════════════════════════════════════════════════════
    _timing_log.clear()
    t_phase = time.perf_counter()
    print("[DIAG] ── Phase 3: constrained_pose_path (pre→pick, 81帧) ──", flush=True)
    path_to_pick, report_pick = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_to_pre[-1], pre, pick,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [q_pre] + seed_bank, count=81,
    )
    elapsed_phase3 = (time.perf_counter() - t_phase) * 1000
    n_calls_phase3 = len(_timing_log)
    total_nfev_phase3 = sum(e["nfev"] for e in _timing_log)
    n_success_phase3 = sum(1 for e in _timing_log if e["success"])
    n_fail_phase3 = n_calls_phase3 - n_success_phase3
    avg_ms_per_call = elapsed_phase3 / max(n_calls_phase3, 1)
    # 找最慢的调用
    if _timing_log:
        slowest = max(_timing_log, key=lambda e: e["elapsed_ms"])
    else:
        slowest = {"elapsed_ms": 0}
    print(f"[DIAG]   Phase 3 完成: {elapsed_phase3:.0f} ms | {n_calls_phase3} 次 IK | {total_nfev_phase3} 总函数评估", flush=True)
    print(f"[DIAG]     成功: {n_success_phase3} | 失败: {n_fail_phase3} | 平均: {avg_ms_per_call:.0f} ms/次 | 最慢: {slowest['elapsed_ms']:.0f} ms", flush=True)

    # ════════════════════════════════════════════════════════════
    # Phase 4: constrained_pose_path (pick→lift, 81 waypoints)
    # ════════════════════════════════════════════════════════════
    _timing_log.clear()
    t_phase = time.perf_counter()
    print("[DIAG] ── Phase 4: constrained_pose_path (pick→lift, 81帧) ──", flush=True)
    _ORIENT_WEIGHT = 0.22
    path_lift, report_lift = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_to_pick[-1], pick, lift,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [path_to_pick[-1]] + seed_bank, count=81,
        axis_weight=_ORIENT_WEIGHT, forward_weight=_ORIENT_WEIGHT,
    )
    elapsed_phase4 = (time.perf_counter() - t_phase) * 1000
    n_calls_phase4 = len(_timing_log)
    total_nfev_phase4 = sum(e["nfev"] for e in _timing_log)
    n_success_phase4 = sum(1 for e in _timing_log if e["success"])
    n_fail_phase4 = n_calls_phase4 - n_success_phase4
    avg_ms_per_call4 = elapsed_phase4 / max(n_calls_phase4, 1)
    if _timing_log:
        slowest4 = max(_timing_log, key=lambda e: e["elapsed_ms"])
    else:
        slowest4 = {"elapsed_ms": 0}
    print(f"[DIAG]   Phase 4 完成: {elapsed_phase4:.0f} ms | {n_calls_phase4} 次 IK | {total_nfev_phase4} 总函数评估", flush=True)
    print(f"[DIAG]     成功: {n_success_phase4} | 失败: {n_fail_phase4} | 平均: {avg_ms_per_call4:.0f} ms/次 | 最慢: {slowest4['elapsed_ms']:.0f} ms", flush=True)

    # ════════════════════════════════════════════════════════════
    # Phase 5: constrained_pose_ramp_path (lift→chest, 151 waypoints)
    # ════════════════════════════════════════════════════════════
    # 先算 corrected_chest_pad
    partial = np.vstack([path_to_pre, path_to_pick[1:], path_lift[1:]])
    approach_end = len(path_to_pre) + len(path_to_pick) - 1
    lock_idx = approach_end + 8

    def _pad_world_xyz(base_world, link6_chain, link6_to_pad, q):
        q_map = dict(zip(ARM_JOINTS, np.asarray(q, dtype=float).tolist()))
        return (base_world @ fk(link6_chain, q_map) @ link6_to_pad)[:3, 3]

    pad_at_lock = _pad_world_xyz(base_world, link6_chain, link6_to_pad, partial[lock_idx])
    tray_from_pad = settled_center - pad_at_lock
    corrected_chest_pad = chest_target - tray_from_pad

    _timing_log.clear()
    t_phase = time.perf_counter()
    print("[DIAG] ── Phase 5: constrained_pose_ramp_path (lift→chest, 151帧, 含旋转渐变) ──", flush=True)
    path_chest, report_chest = constrained_pose_ramp_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_lift[-1], lift, corrected_chest_pad,
        TARGET_FORWARD_WORLD, CHEST_FORWARD_WORLD,
        [path_lift[-1]] + seed_bank, count=151,
        start_up_world=TARGET_UP_WORLD, end_up_world=CHEST_UP_WORLD,
        axis_weight=_ORIENT_WEIGHT, forward_weight=_ORIENT_WEIGHT,
    )
    elapsed_phase5 = (time.perf_counter() - t_phase) * 1000
    n_calls_phase5 = len(_timing_log)
    total_nfev_phase5 = sum(e["nfev"] for e in _timing_log)
    n_success_phase5 = sum(1 for e in _timing_log if e["success"])
    avg_ms_per_call5 = elapsed_phase5 / max(n_calls_phase5, 1)
    if _timing_log:
        slowest5 = max(_timing_log, key=lambda e: e["elapsed_ms"])
    else:
        slowest5 = {"elapsed_ms": 0}
    print(f"[DIAG]   Phase 5 完成: {elapsed_phase5:.0f} ms | {n_calls_phase5} 次 IK | {total_nfev_phase5} 总函数评估", flush=True)
    print(f"[DIAG]     成功: {n_success_phase5} | 失败: {n_fail_phase5} | 平均: {avg_ms_per_call5:.0f} ms/次 | 最慢: {slowest5['elapsed_ms']:.0f} ms", flush=True)

    # ════════════════════════════════════════════════════════════
    # 总结
    # ════════════════════════════════════════════════════════════
    total_ms = elapsed_phase1 + elapsed_phase2 + elapsed_phase3 + elapsed_phase4 + elapsed_phase5
    total_ik_calls = n_calls_phase1 + n_calls_phase2 + n_calls_phase3 + n_calls_phase4 + n_calls_phase5
    total_nfev = total_nfev_phase1 + total_nfev_phase3 + total_nfev_phase4 + total_nfev_phase5

    print("\n[DIAG] ╔══════════════════════════════════════════════════════╗", flush=True)
    print("[DIAG] ║              诊断结果汇总                             ║", flush=True)
    print("[DIAG] ╠══════════════════════════════════════════════════════╣", flush=True)
    print(f"[DIAG] ║  Phase 1 (pregrasp IK, 1次):     {elapsed_phase1:8.0f} ms  ({elapsed_phase1/total_ms*100:5.1f}%) ║", flush=True)
    print(f"[DIAG] ║  Phase 2 (fallback 插值):        {elapsed_phase2:8.0f} ms  ({elapsed_phase2/total_ms*100:5.1f}%) ║", flush=True)
    print(f"[DIAG] ║  Phase 3 (pre→pick, 80×IK):     {elapsed_phase3:8.0f} ms  ({elapsed_phase3/total_ms*100:5.1f}%) ║", flush=True)
    print(f"[DIAG] ║  Phase 4 (pick→lift, 80×IK):    {elapsed_phase4:8.0f} ms  ({elapsed_phase4/total_ms*100:5.1f}%) ║", flush=True)
    print(f"[DIAG] ║  Phase 5 (lift→chest, 150×IK):  {elapsed_phase5:8.0f} ms  ({elapsed_phase5/total_ms*100:5.1f}%) ║", flush=True)
    print(f"[DIAG] ╠══════════════════════════════════════════════════════╣", flush=True)
    print(f"[DIAG] ║  总计:                            {total_ms:8.0f} ms  (100.0%) ║", flush=True)
    print(f"[DIAG] ║  总 IK 调用次数:  {total_ik_calls}                              ║", flush=True)
    print(f"[DIAG] ║  总函数评估次数:  {total_nfev}                          ║", flush=True)
    print(f"[DIAG] ╚══════════════════════════════════════════════════════╝", flush=True)

    # 瓶颈建议
    print("\n[DIAG] ═══ 瓶颈分析与建议 ═══", flush=True)
    slowest_phase = max([
        (1, elapsed_phase1), (3, elapsed_phase3),
        (4, elapsed_phase4), (5, elapsed_phase5)
    ], key=lambda x: x[1])
    print(f"[DIAG] 最慢阶段: Phase {slowest_phase[0]} ({slowest_phase[1]:.0f} ms)", flush=True)
    print(f"[DIAG] 与 GPU 无关 — scipy.optimize.least_squares 是纯 CPU 计算", flush=True)
    print(f"[DIAG] 建议1: 减少 IK 帧数 (count 从 81/151 降到 30/60)", flush=True)
    print(f"[DIAG] 建议2: 使用缓存 (--use-cache 跳过 IK)", flush=True)
    print(f"[DIAG] 建议3: 用 CuRobo GPU 规划替代 scipy IK 生成路径", flush=True)
    print(f"[DIAG] 建议4: 在 Isaac Sim 环境内运行时 GPU 不参与 scipy 计算", flush=True)

    return {
        "total_ms": total_ms,
        "total_ik_calls": total_ik_calls,
        "total_nfev": total_nfev,
        "phases": {
            "phase1_pregrasp": elapsed_phase1,
            "phase2_fallback": elapsed_phase2,
            "phase3_pre_to_pick": elapsed_phase3,
            "phase4_pick_to_lift": elapsed_phase4,
            "phase5_lift_to_chest": elapsed_phase5,
        }
    }


if __name__ == "__main__":
    result = run_diagnostics()

    # 如果总耗时超过 30 秒，打印警告
    if result["total_ms"] > 30000:
        print(f"\n[DIAG] ⚠️ 总耗时 {result['total_ms']/1000:.1f} 秒 — 远超预期!", flush=True)
    elif result["total_ms"] > 10000:
        print(f"\n[DIAG] ⚡ 总耗时 {result['total_ms']/1000:.1f} 秒 — 偏慢但可接受", flush=True)
    else:
        print(f"\n[DIAG] ✅ 总耗时 {result['total_ms']/1000:.1f} 秒 — 正常", flush=True)
