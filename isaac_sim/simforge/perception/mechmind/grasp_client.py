#!/usr/bin/env python3
"""从 Mech-Vision 接收抓取位姿，转换到机器人基坐标系。

流程：
    Mech-Eye Log S → Mech-Vision（点云处理 + 抓取检测）
                   → 本脚本（TCP 接收位姿）
                   → 机器人基坐标系下的 pick pose

依赖：
    Mech-Vision 已部署并启动对应工程，开放了 Robot Interface 端口（默认 50004）

用法：
    python3 grasp_client.py                      # 连接默认地址，请求一次抓取位姿
    python3 grasp_client.py --host 192.168.x.x   # 指定 Mech-Vision 主机 IP
    python3 grasp_client.py --loop               # 持续轮询，Ctrl+C 停止
"""

from __future__ import annotations
import argparse
import json
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np


MECHEYE_DEFAULT_HOST = "127.0.0.1"
MECHEYE_ROBOT_PORT   = 50004      # Mech-Vision Robot Interface 默认端口


@dataclass
class GraspPose:
    """一个抓取候选位姿（相机坐标系或机器人基坐标系）。"""
    position: np.ndarray      # [x, y, z]  单位 mm
    quaternion: np.ndarray    # [w, x, y, z]
    score: float = 0.0
    label: str = ""

    def to_robot_base(self, cam_to_base: np.ndarray) -> "GraspPose":
        """将位姿从相机坐标系变换到机器人基坐标系。

        cam_to_base: 4×4 齐次变换矩阵（手眼标定结果）
        """
        from scipy.spatial.transform import Rotation as R

        p_cam = np.array([*self.position, 1.0])
        p_base = (cam_to_base @ p_cam)[:3]

        rot_cam = R.from_quat([self.quaternion[1], self.quaternion[2],
                                self.quaternion[3], self.quaternion[0]])
        rot_base = R.from_matrix(cam_to_base[:3, :3]) * rot_cam
        q_base_xyzw = rot_base.as_quat()
        q_base_wxyz = np.array([q_base_xyzw[3], *q_base_xyzw[:3]])

        return GraspPose(position=p_base, quaternion=q_base_wxyz,
                         score=self.score, label=self.label)

    def position_m(self) -> np.ndarray:
        """位置转米。"""
        return self.position / 1000.0


class MechVisionClient:
    """Mech-Vision Robot Interface TCP 客户端（JSON 协议）。"""

    def __init__(self, host: str = MECHEYE_DEFAULT_HOST, port: int = MECHEYE_ROBOT_PORT,
                 timeout: float = 10.0):
        self.host    = host
        self.port    = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        print(f"[MechVision] 已连接 {self.host}:{self.port}")

    def disconnect(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send(self, cmd: dict):
        data = (json.dumps(cmd) + "\n").encode()
        self._sock.sendall(data)

    def _recv(self) -> dict:
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("连接断开")
            buf += chunk
        return json.loads(buf.decode())

    def get_grasp_poses(self, job_id: int = 1, max_poses: int = 5) -> list[GraspPose]:
        """触发 Mech-Vision 工程并获取抓取位姿列表。"""
        self._send({"command": "get_planned_grasp_poses",
                    "job_id": job_id, "max_poses": max_poses})
        resp = self._recv()

        if resp.get("status") != "OK":
            raise RuntimeError(f"Mech-Vision 返回错误：{resp}")

        poses = []
        for item in resp.get("poses", []):
            pos = np.array(item["position"])       # [x, y, z] mm
            quat = np.array(item["quaternion"])    # [w, x, y, z]
            poses.append(GraspPose(
                position=pos,
                quaternion=quat,
                score=item.get("score", 0.0),
                label=item.get("label", ""),
            ))
        return poses


def load_hand_eye_matrix(path: str) -> np.ndarray:
    """从 JSON 文件加载手眼标定矩阵（4×4）。

    文件格式：{"matrix": [[r00,r01,...,t0], ..., [0,0,0,1]]}
    """
    with open(path) as f:
        data = json.load(f)
    return np.array(data["matrix"], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Mech-Vision 抓取位姿客户端")
    parser.add_argument("--host",      default=MECHEYE_DEFAULT_HOST, help="Mech-Vision 主机 IP")
    parser.add_argument("--port",      type=int, default=MECHEYE_ROBOT_PORT)
    parser.add_argument("--job",       type=int, default=1, help="Mech-Vision 工程 ID")
    parser.add_argument("--max-poses", type=int, default=5)
    parser.add_argument("--hand-eye",  help="手眼标定矩阵 JSON 文件路径（可选）")
    parser.add_argument("--loop",      action="store_true", help="持续轮询")
    args = parser.parse_args()

    cam_to_base = None
    if args.hand_eye:
        cam_to_base = load_hand_eye_matrix(args.hand_eye)
        print(f"[grasp_client] 手眼矩阵已加载：{args.hand_eye}")

    client = MechVisionClient(args.host, args.port)
    client.connect()

    try:
        while True:
            poses = client.get_grasp_poses(job_id=args.job, max_poses=args.max_poses)
            print(f"\n获取到 {len(poses)} 个抓取位姿：")
            for i, p in enumerate(poses):
                if cam_to_base is not None:
                    p = p.to_robot_base(cam_to_base)
                print(f"  [{i}] pos={np.round(p.position_m(), 4)} m  "
                      f"quat(wxyz)={np.round(p.quaternion, 4)}  score={p.score:.3f}")

            if not args.loop:
                break
            time.sleep(1.0)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
