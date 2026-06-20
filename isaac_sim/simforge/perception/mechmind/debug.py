#!/usr/bin/env python3
"""Mech-Eye Log S 一键调试诊断 — 网络、SDK、连接、帧率、点云质量。

用法：
    python3.10 debug.py                              # 完整诊断（自动发现相机）
    python3.10 debug.py --ip 192.168.x.x             # 指定相机 IP
    python3.10 debug.py --ip 192.168.x.x --ping-only # 仅测试网络
"""

from __future__ import annotations
import argparse
import subprocess
import socket
import sys
import time


def _sep(title: str = ""):
    if title:
        print(f"\n── {title} {'─' * max(0, 46 - len(title))}")
    else:
        print("─" * 50)


# ── 1. 网络 Ping ──────────────────────────────────────────────────────────────

def check_ping(ip: str, count: int = 4) -> bool:
    _sep(f"[1/5] Ping {ip}")
    result = subprocess.run(
        ["ping", "-c", str(count), "-W", "1", ip],
        capture_output=True, text=True,
    )
    # show only summary line
    for line in result.stdout.splitlines():
        if "packet" in line or "ms" in line:
            print(" ", line)
    ok = result.returncode == 0
    print("  ✓ 网络连通" if ok else "  ✗ Ping 失败 — 检查网线/IP/防火墙")
    return ok


# ── 2. Mech-Vision 端口 ───────────────────────────────────────────────────────

def check_port(ip: str, port: int = 50004) -> bool:
    _sep(f"[2/5] Mech-Vision Robot Interface 端口 {port}")
    try:
        with socket.create_connection((ip, port), timeout=3):
            print(f"  ✓ {ip}:{port} 可达")
            return True
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"  ~ {ip}:{port} 不可达 ({e})")
        print("    → 若暂未部署 Mech-Vision，可忽略此项")
        return False


# ── 3. SDK 安装 ───────────────────────────────────────────────────────────────

def check_sdk() -> bool:
    _sep("[3/5] SDK 安装检查")
    # Check .deb
    import subprocess as sp
    deb = sp.run(["dpkg", "-l", "mecheyeapi"], capture_output=True, text=True)
    if "mecheyeapi" in deb.stdout:
        ver_line = [l for l in deb.stdout.splitlines() if "mecheyeapi" in l]
        print(f"  ✓ Mech-Eye SDK .deb: {ver_line[0].split()[2] if ver_line else 'installed'}")
    else:
        print("  ✗ Mech-Eye SDK .deb 未安装")
        print("    → sudo dpkg -i perception/mechmind/sdk/Mech-Eye_API_2.6.0_amd64.deb")

    # Check Python package
    try:
        from mecheye.area_scan_3d_camera import Camera
        import importlib.metadata
        try:
            ver = importlib.metadata.version("MechEyeAPI")
        except Exception:
            ver = "已安装"
        print(f"  ✓ MechEyeApi Python 包: {ver}")
        return True
    except ImportError:
        print("  ✗ MechEyeApi Python 包未安装")
        print("    → python3.10 -m pip install MechEyeApi")
        return False


# ── 4. 相机发现与连接 ─────────────────────────────────────────────────────────

def check_camera(ip: str | None) -> object | None:
    _sep(f"[4/5] 相机连接 {'IP=' + ip if ip else '(自动发现)'}")
    try:
        from mecheye.area_scan_3d_camera import Camera

        infos = Camera.discover_cameras()
        print(f"  网络扫描: 发现 {len(infos)} 台相机")
        for i, info in enumerate(infos):
            print(f"    [{i}] {info.model}  {info.ip_address}  SN={info.serial_number}")

        if not infos:
            print("  ✗ 未发现相机 — 检查同一子网、防火墙放行 UDP 5787")
            return None

        target = next((i for i in infos if i.ip_address == ip), infos[0]) if ip else infos[0]
        camera = Camera()
        status = camera.connect(target)
        if not status.is_ok():
            print(f"  ✗ 连接失败: {status.description()}")
            return None

        print(f"  ✓ 已连接: {target.model}  固件={target.firmware_version}")
        return camera

    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return None


# ── 5. 采集帧率与点云质量 ─────────────────────────────────────────────────────

def check_capture(camera, n_frames: int = 3):
    _sep(f"[5/5] 采集 {n_frames} 帧（帧率 + 点云密度）")
    from mecheye.area_scan_3d_camera import Frame2D, Frame3D

    times_2d, times_3d = [], []
    for i in range(n_frames):
        # 2D
        t0 = time.perf_counter()
        f2 = Frame2D()
        s = camera.capture_2d(f2)
        dt2 = time.perf_counter() - t0
        # 3D
        t0 = time.perf_counter()
        f3 = Frame3D()
        s3 = camera.capture_3d(f3)
        dt3 = time.perf_counter() - t0

        if s.is_ok() and s3.is_ok():
            times_2d.append(dt2)
            times_3d.append(dt3)
            cloud = f3.get_untextured_point_cloud()
            total_pts = cloud.width() * cloud.height()
            valid_pts = sum(
                1 for j in range(min(total_pts, 10000))
                if cloud[j].z > 0
            )
            valid_pct = valid_pts / min(total_pts, 10000) * 100
            print(f"  帧{i+1}: 2D={dt2*1000:.0f}ms  3D={dt3*1000:.0f}ms  "
                  f"点云={cloud.width()}×{cloud.height()}  有效点≈{valid_pct:.0f}%")
        else:
            print(f"  帧{i+1}: 采集失败 2D={s.description()} 3D={s3.description()}")

    if times_2d:
        print(f"  平均: 2D={sum(times_2d)/len(times_2d)*1000:.0f}ms  "
              f"3D={sum(times_3d)/len(times_3d)*1000:.0f}ms")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mech-Eye 一键诊断")
    parser.add_argument("--ip",        help="相机 IP（留空则自动发现）")
    parser.add_argument("--vision-ip", help="Mech-Vision 主机 IP（用于端口检查）")
    parser.add_argument("--ping-only", action="store_true")
    args = parser.parse_args()

    _sep()
    print("  Mech-Eye Log S 调试诊断")
    _sep()

    target_ip = args.ip
    vision_ip = args.vision_ip or target_ip

    if vision_ip:
        ping_ok = check_ping(vision_ip)
        if args.ping_only:
            return
        if not ping_ok:
            print("\n网络不通，终止诊断。")
            sys.exit(1)
        check_port(vision_ip)
    else:
        if args.ping_only:
            print("--ping-only 需要 --ip 参数")
            sys.exit(1)

    sdk_ok = check_sdk()
    if not sdk_ok:
        print("\nSDK 未就绪，终止诊断。")
        sys.exit(1)

    camera = check_camera(target_ip)
    if camera is None:
        print("\n相机未连接，跳过采集测试。")
        _sep()
        print("  诊断完成（无相机）")
        _sep()
        return

    check_capture(camera)
    camera.disconnect()

    _sep()
    print("  诊断完成 ✓")
    _sep()


if __name__ == "__main__":
    main()
