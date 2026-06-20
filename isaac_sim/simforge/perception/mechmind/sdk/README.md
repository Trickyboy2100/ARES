# SDK 文件说明

本目录存放 Mech-Eye SDK 安装文件（.deb / .zip 不进 git，太大）。

## 已下载文件（本机）

| 文件 | 版本 | 大小 | 说明 |
|------|------|------|------|
| `Mech-Eye_API_2.6.0_amd64.deb` | 2.6.0 | 150 MB | Ubuntu AMD64 SDK 主包 |
| `Mech-Eye_API_2.6.0_amd64.zip` | 2.6.0 | 150 MB | 同上的 zip 包装 |
| `mecheye_python_samples/` | 2.5.4+ | — | 官方 Python 示例（git clone） |

## 新机器安装步骤

```bash
# 1. 下载 SDK（从官方下载中心）
#    https://downloads.mech-mind.com/?tab=tab-sdk
#    选择 Mech-Eye SDK → Linux → AMD64 → 下载 zip

# 2. 解压（zip 格式需用 7z）
sudo apt-get install -y p7zip-full
7z e Mech-Eye_API_2.6.0_amd64.zip

# 3. 安装 .deb（SDK 装到 /opt/mech-mind/mech-eye-sdk/）
sudo dpkg -i Mech-Eye_API_2.6.0_amd64.deb

# 4. 安装 Python 包（需 Python 3.7–3.11，推荐系统 python3.10）
python3.10 -m pip install MechEyeApi

# 5. 安装 g++12（Python 包运行时依赖）
sudo apt-get install -y g++-12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 20

# 6. 验证
python3.10 -c "from mecheye.area_scan_3d_camera import Camera; print('OK')"
```

## 本机已安装版本

- SDK .deb: `mecheyeapi 2.6.0`（`/opt/mech-mind/mech-eye-sdk/`）
- Python 包: `MechEyeApi 2.5.4`（`python3.10`）
- g++: `12.3.0`

## 重要提示

- Python 版本必须是 3.7–3.11，**Python 3.12/3.13 不支持**（.so 编译限制）
- 使用 `python3.10`，不是 `python3`（miniconda 默认是 3.13）
- 相机与主机需在**同一子网**，防火墙放行 UDP 5787（发现）和 TCP 5577（数据）
