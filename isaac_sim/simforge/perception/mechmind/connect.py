#!/usr/bin/env python3
"""Mech-Eye Log S 相机连接、状态检查与基础采集。

依赖：
    python3.10 -m pip install MechEyeApi   # 安装 mecheye SDK
    SDK .deb 需已安装（见 sdk/ 目录）

用法：
    python3.10 connect.py                    # 自动发现并连接第一台相机
    python3.10 connect.py --ip 192.168.x.x  # 指定 IP 连接
    python3.10 connect.py --list             # 仅列出可用相机，不连接
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

try:
    from mecheye.shared import *
    from mecheye.area_scan_3d_camera import Camera, CameraInfo, Frame2D, Frame3D
    from mecheye.area_scan_3d_camera_utils import find_and_connect, print_camera_info
except ImportError:
    print(
        "mecheye SDK 未安装。\n"
        "  python3.10 -m pip install MechEyeApi\n"
        "另需安装 SDK .deb（perception/mechmind/sdk/ 目录下）。",
        file=sys.stderr,
    )
    sys.exit(1)


def list_cameras() -> list:
    """返回网络中所有可发现的 Mech-Eye 相机列表。"""
    return Camera.discover_cameras()


def connect(ip: str | None = None) -> Camera:
    """连接相机并返回 Camera 对象。ip=None 时自动选第一台。"""
    infos = list_cameras()
    if not infos:
        raise RuntimeError("未发现任何 Mech-Eye 相机，请检查网络连接和子网设置。")

    if ip:
        matched = [i for i in infos if i.ip_address == ip]
        if not matched:
            available = [i.ip_address for i in infos]
            raise RuntimeError(f"未找到 IP={ip} 的相机。已发现: {available}")
        target = matched[0]
    else:
        target = infos[0]
        print(f"自动选择第一台相机: {target.model}  IP={target.ip_address}")

    camera = Camera()
    status = camera.connect(target)
    if not status.is_ok():
        raise RuntimeError(f"连接失败: {status.description()}")

    print(f"已连接 ✓  型号={target.model}  IP={target.ip_address}  "
          f"固件={target.firmware_version}  SN={target.serial_number}")
    return camera


def capture_once(camera: Camera, save_dir: str = "/tmp/mechmind") -> dict:
    """采集一帧（彩色图 + 深度图 + 点云），保存到 save_dir，返回文件路径。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2D 采集
    frame2d = Frame2D()
    status = camera.capture_2d(frame2d)
    if not status.is_ok():
        raise RuntimeError(f"2D 采集失败: {status.description()}")
    color = frame2d.get_color_image()
    color_path = str(save_dir / "color.png")
    color.save(color_path)
    print(f"  彩色图: {color.width()}×{color.height()}  → {color_path}")

    # 3D 采集
    frame3d = Frame3D()
    status = camera.capture_3d(frame3d)
    if not status.is_ok():
        raise RuntimeError(f"3D 采集失败: {status.description()}")

    depth = frame3d.get_depth_map()
    depth_path = str(save_dir / "depth.png")
    depth.save(depth_path)
    print(f"  深度图: {depth.width()}×{depth.height()}  → {depth_path}")

    cloud = frame3d.get_untextured_point_cloud()
    cloud_path = str(save_dir / "cloud.ply")
    cloud.save(cloud_path)
    print(f"  点  云: {cloud.width()}×{cloud.height()} pts  → {cloud_path}")

    return {"color": color_path, "depth": depth_path, "cloud": cloud_path}


def main():
    parser = argparse.ArgumentParser(description="Mech-Eye 相机连接与采集")
    parser.add_argument("--ip",   help="相机 IP（留空则自动发现）")
    parser.add_argument("--list", action="store_true", help="仅列出可用相机")
    parser.add_argument("--save", default="/tmp/mechmind", help="保存采集结果的目录")
    args = parser.parse_args()

    if args.list:
        infos = list_cameras()
        if not infos:
            print("未发现相机。")
        for i, info in enumerate(infos):
            print(f"  [{i}] {info.model}  IP={info.ip_address}  SN={info.serial_number}  固件={info.firmware_version}")
        return

    camera = connect(ip=args.ip)
    try:
        capture_once(camera, save_dir=args.save)
    finally:
        camera.disconnect()
        print("已断开连接。")


if __name__ == "__main__":
    main()
