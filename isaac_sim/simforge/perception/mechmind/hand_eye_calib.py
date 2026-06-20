#!/usr/bin/env python3
"""手眼标定辅助工具 — 收集机器人位姿 + 相机棋盘格观测，求解 cam→base 矩阵。

依赖：
    pip install opencv-python scipy numpy

标定流程：
    1. 将棋盘格固定在机器人可见范围内
    2. 运行此脚本，按提示移动机器人并按 Enter 采集各姿态
    3. 标定完成后结果保存到 hand_eye_result.json

用法：
    python3 hand_eye_calib.py --board 7x6 --square 25  # 7×6 角点，25mm 方格
    python3 hand_eye_calib.py --verify hand_eye_result.json  # 验证已有结果
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def detect_chessboard(image: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """检测图像中的棋盘格角点，返回 (N,2) 图像坐标，失败返回 None。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(gray, pattern)
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return corners.reshape(-1, 2)


def robot_pose_to_matrix(pos_mm: list, quat_wxyz: list) -> np.ndarray:
    """机器人末端位姿（位置 mm + 四元数）→ 4×4 齐次矩阵。"""
    T = np.eye(4)
    T[:3, :3] = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    T[:3, 3]  = np.array(pos_mm) / 1000.0
    return T


def solve_hand_eye(R_gripper2base: list, t_gripper2base: list,
                   R_target2cam: list,  t_target2cam: list) -> tuple[np.ndarray, np.ndarray]:
    """调用 OpenCV AX=XB 手眼标定。"""
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    return R_cam2gripper, t_cam2gripper


def build_matrix(R_: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_
    T[:3, 3]  = t.flatten()
    return T


def save_result(path: str, T_cam2base: np.ndarray):
    data = {"matrix": T_cam2base.tolist(),
            "description": "cam_to_base 4x4 homogeneous matrix (units: meters)"}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"结果已保存 → {path}")


def verify(result_path: str, test_pos_mm: list, test_quat: list,
           cam_pos_mm: list):
    """验证：给定末端位姿和相机坐标系下的点，输出机器人基坐标系下的位置。"""
    T_cam2base = np.array(json.load(open(result_path))["matrix"])
    T_ee2base  = robot_pose_to_matrix(test_pos_mm, test_quat)
    p_cam = np.array([*np.array(cam_pos_mm) / 1000.0, 1.0])
    p_base = (T_cam2base @ p_cam)[:3]
    print(f"相机系点 {cam_pos_mm} mm → 基坐标系 {np.round(p_base * 1000, 1)} mm")


def main():
    parser = argparse.ArgumentParser(description="手眼标定辅助")
    parser.add_argument("--board",   default="7x6", help="棋盘格角点数 列x行")
    parser.add_argument("--square",  type=float, default=25.0, help="方格尺寸 mm")
    parser.add_argument("--output",  default="hand_eye_result.json")
    parser.add_argument("--verify",  help="验证模式：传入已有结果 JSON")
    args = parser.parse_args()

    if args.verify:
        print(f"验证模式：{args.verify}")
        print("请输入末端位姿 (pos_mm x y z) (quat w x y z) 和相机系点 (x y z mm):")
        pos   = list(map(float, input("  末端位置 mm (x y z): ").split()))
        quat  = list(map(float, input("  末端姿态 wxyz: ").split()))
        cpt   = list(map(float, input("  相机系点 mm (x y z): ").split()))
        verify(args.verify, pos, quat, cpt)
        return

    cols, rows = map(int, args.board.split("x"))
    pattern    = (cols, rows)
    sq_mm      = args.square

    # 棋盘格世界坐标（Z=0 平面）
    obj_pts = np.zeros((cols * rows, 3), dtype=np.float32)
    obj_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_mm

    print(f"棋盘格：{cols}×{rows} 角点，方格 {sq_mm} mm")
    print("按 Enter 采集每个姿态，输入 'q' 完成采集并计算。\n")

    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []

    # 相机内参（需提前标定或从相机 SDK 读取）
    print("请输入相机内参（从 Mech-Eye SDK 或 Mech-Vision 获取）：")
    fx = float(input("  fx (px): "))
    fy = float(input("  fy (px): "))
    cx = float(input("  cx (px): "))
    cy = float(input("  cy (px): "))
    K  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)  # 假设已校正；如有畸变系数可在此填入

    n = 0
    while True:
        cmd = input(f"\n姿态 {n+1} — 移动机器人后按 Enter 采集，q 完成: ").strip()
        if cmd.lower() == "q":
            break

        # 采集图像（从相机 SDK 获取 color_map，此处用 OpenCV 摄像头做占位）
        print("  采集彩色图（请在 connect.py 里替换为 mecheye 采集）...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("  图像采集失败，跳过。")
            continue

        corners = detect_chessboard(frame, pattern)
        if corners is None:
            print("  未检测到棋盘格，调整位置后重试。")
            continue

        retval, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist)
        R_t2c, _ = cv2.Rodrigues(rvec)
        R_t2c_list.append(R_t2c)
        t_t2c_list.append(tvec)

        print("  请输入当前机器人末端位姿：")
        pos_mm  = list(map(float, input("  末端位置 mm (x y z): ").split()))
        quat    = list(map(float, input("  末端姿态 wxyz: ").split()))
        T_g2b   = robot_pose_to_matrix(pos_mm, quat)
        R_g2b_list.append(T_g2b[:3, :3])
        t_g2b_list.append(T_g2b[:3, 3:])
        n += 1
        print(f"  ✓ 姿态 {n} 已记录")

    if n < 3:
        print("至少需要 3 个姿态，退出。")
        return

    R_c2g, t_c2g = solve_hand_eye(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
    T_cam2base = build_matrix(R_c2g, t_c2g)
    print(f"\n标定完成（{n} 个姿态）：")
    print(T_cam2base)
    save_result(args.output, T_cam2base)


if __name__ == "__main__":
    main()
