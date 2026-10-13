# Isaac Sim 5.1状态

BLOCKED：190研究/软件目录未发现5.1安装，现有OmniGibson容器标明Isaac4.1；未运行旧版。未下载大型Isaac镜像，未改驱动。216使用许可已询问，尚未切换。

fig01是既有USD真实几何的CPU静态预览，**STATIC_USD_CPU_GEOMETRY_RENDER**，不是Isaac运行画面。原main.usd保持不变。原USD左右臂与ARES-R BODY映射不能由同一个刚体变换满足（既有审计平移差1.498m），新的research层尚未完成，禁止把旧USD世界图与BODY规划图当作已标定同一场景。

现有5.1捕获脚本保留独立进程、日志与限时退出，不调用全局pkill。STATIC_RENDER、规划预览、物理执行必须继续分别标注。
