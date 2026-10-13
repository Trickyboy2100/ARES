# 190 GPU实际运行记录

2026-10-13，4×RTX4090 24GB，驱动560.35.05。启动前四卡均2MiB/0%，结束后恢复2MiB/0%；只用GPU0。根磁盘159→158GB空闲，独立依赖504MB+新版Warp349MB，另有下载归档；没有大型镜像下载、共享镜像覆盖或驱动改动。

真实PyTorch 2.7.1+cu128矩阵计算PASS；cuda-core 0.7/NVRTC编译sm_89 cubin PASS；固定cuRobo实际规划PASS（仅诊断级路径）。CUDA12.8 Docker镜像启动被cuda>=12.8容器门槛拒绝，未禁用门槛。现有宿主PyTorch与用户依赖路径可用。

cuRobo官方[固定源码安装要求](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/docs/getting-started/installation.rst)仍要求580.65.06；本轮实际可运行不等于受支持的正式配置。CUDA[小版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)解释为何部分cu12代码可在较旧驱动运行，但不能替代实测。

诊断初始化1.994s、warm-up5.145s；三次求解2.855/3.025/3.052s，最后一条成功。Cabinet第二候选持久Worker端到端0.854s，但为碰撞拒绝请求。**四卡总吞吐NOT_TESTED**，不能外推1°诊断任务为工业规划性能。
