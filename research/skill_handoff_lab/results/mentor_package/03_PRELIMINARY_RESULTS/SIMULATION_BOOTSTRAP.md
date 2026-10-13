# Isaac Sim 5.1：190真实静态渲染验收

2026-10-13。216未获授权，未连接；全部新增GPU任务在190个人研究目录。未修改宿主驱动、内核、共享容器或重启。

## 实际完成

个人Python 3.11.17环境安装官方Isaac Sim 5.1.0及最小core/replicator/asset/storage/utils组件；Torch 2.7.0+cu126。环境约6.5GB，Kit运行扩展缓存另约4GB；最终宿主磁盘可用144GB。大型完整镜像未下载。曾尝试同步本地安装，因LAN慢停止本研究自己的rsync，保留部分文件；一次核心下载超时后仅进行一次增加HTTP超时的有界重试。只清理本研究uv安装缓存，保留研究数据与Kit缓存。

第一次兼容检查发现重复Vulkan ICD和CUDA_VISIBLE_DEVICES映射问题。修复仅作用于本研究子进程：指定`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`、取消CUDA重映射、显式选择物理GPU1、关闭multi_gpu。第二次兼容检查通过；第一轮冷启动捕获240秒超时并退出自己的进程组，原日志保留；第二次同参数捕获成功。

实际运行`SimulationApp`，headless、RayTracedLighting，打开ARES原main.usd，Replicator RGB输出1280×960，非空且人工检视可见双臂机器人及工位。[展示截图](fig01_isaac_scene.png)仅添加证据等级说明，未经生成式处理；原始像素另存[原图](../../e0_20261013/isaac51_static_capture_v2/isaac51_scene.png)。进程返回0，wall=227.747秒。完整argv、PID、版本、场景SHA、相机位姿、输出日志见`../../e0_20261013/isaac51_static_capture_v2`。场景SHA `0d0451c2cdc72f11cd2ec15be0900571771c5ac14618f518090be52d580bb9a2`。

最终四卡均2MiB、0%占用。启动期间其他卡有约数百MiB CUDA上下文，不代表多个Oracle或物理环境。未长期占用多卡。

## 严格限制

- 证据等级 **ISAAC_5_1_STATIC_RENDER**，不是任务物理执行、不是接触抓取、不是训练视频。
- 原USD左右臂安装与ARES-R BODY映射尚未共同验收；未偷偷修改原USD标定。
- PhysX日志明确报告`/World/Tray/Mesh`动态triangle mesh不能使用当前碰撞设置，回退convex hull；必须在独立research层修正/验证后才可评价物理碰撞。
- Isaac Lab 2.3.0及DirectRLEnv尚未部署验收；V0任务、V1 16个独立物理实例、V2–V4均NOT_TESTED。没有16进程冒充向量化。
- 560.35.05实际完成启动和截图，不能继续把驱动版本写成静态渲染绝对阻塞；但它不是[Isaac 5.1官方测试驱动580.65.06](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。兼容检查器内部较低最低版本不替代官方测试要求，也不证明所有动力学功能稳定。

## 下一实验

先在独立research层核验共同BODY坐标变换、关节映射、固定托盘附着和静态/凸分解碰撞几何，再运行单环境持物轨迹并检查姿态、接触、掉落与目标误差。只有V0通过后，按[Isaac Lab 2.3安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)采用scene cloning/physics replication构建4×4独立实例。此轮尚无物理反馈或仿真RL结果。
