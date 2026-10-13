# R1实际结果与未完成项

190个人研究目录运行，基线f93cd24，seed31。未调用实机/216，未修改ARES-R生产配置、驱动或重启。2026-10-14本地凌晨运行，UTC日志2026-10-13。

|项目|状态|实测证据/边界|
|---|---|---|
|190 Isaac5.1 + Lab2.3|PASS|tag源码3c6e67b；pip isaaclab0.47.2为此tag内部扩展版本；torch2.7/SB3 2.7/gym1.2/warp1.9.1|
|单进程1→4→16 PhysX|PASS|scene cloning / replicated physics；独立origins、q、target；world dt1/120|
|在线标准PPO smoke|PASS|256/1024/4096交互；32/128/512 Adam更新；checkpoint参数变化|
|16环境实际截图/在线GIF|PASS|8个训练期间帧，各附实际关节状态；不是16个应用或固定轨迹播放|
|同起点Prepare规划|PASS|5/5真实路径通过代理验收；三Option均可行，Nominal更短|
|三Option同起点PhysX|PASS（独立右臂采样范围）|同q0、同目标、同反馈保持；684/848/928步，三条均通过，无Prepare独有优势|
|名义带物PhysX精度|FAIL|81规划节点，444物理步；4.23mm > 冻结2mm门限|
|有界普通反馈保持|PASS（局部采样范围）|额外240步，endpoint低于2mm；普通控制修正，无RL|
|完整双臂/底盘/柜口任务|NOT_TESTED|目前独立右臂；完整碰撞、托盘惯量与实际任务随机化不足|
|A的非短视RL必要性|NOT_TESTED|仅一个开放案例，负机制证据；不支持宣称需要RL|
|B真正恢复Option-RL|NOT_TESTED|尚无经确认的E2可恢复失败；软件错误不是恢复样本|
|正式RL与强基线/泛化|NOT_TESTED|smoke不是方法优势；R2尚未完成|
|文献全部精读|NOT_TESTED|已读深度见对应表，未完成项明确保留|

16环境总墙时42.72s含Kit启动、训练、渲染及退出；训练CSV约409环境交互/s（小规模单seed，含周期摄像），不能推算四卡吞吐。使用GPU1，规划批次GPU0有界执行；非4卡独占。末次运行前磁盘143GB可用，未下载大镜像。

物理原JAKA实例原型中已有真实网格/凸包，初次普通Traverse遗漏instance proxies造成误判；最终保留原碰撞几何，并用PrimRange TraverseInstanceProxies核查。保留固定EG2源STL，焊接托盘使用既有USD AABB与Link6变换。零姿态安装帧误差3.673e-6，实际动态Link6-FK最大位置差0.60微米/旋转差7.66e-5度。该数字仅软件模型一致性，不是实机标定精度。

故障保留：依赖缺失、材质引用警告、同步Replicator捕获挂起（仅终止自有进程）、全局1-camera被16env reset索引导致CUDA assert、轨迹dtype错误。最终改用Lab Camera、不将全局camera放入N-env sensor集合、显式float32，并要求final manifest而非exit0判定成功。全局pkill未使用。

当前最值得投入：完整安装/碰撞验收后冻结E1/E2参数；同起点完整物理Prepare与充分预算恢复搜索；比较普通控制/重新规划能否解决误差。只有普通规划/控制不足时选择匹配Option-RL。不要继续E0反复刷训练步数。

验证：现有研究测试18 passed（2个Gym无限Box提示）；新工具全部py_compile通过；git diff --check通过。原始执行验收按节点采样，未验证每步接触/加速度，不扩大安全结论。

重现：tools/run_r1_physx.py（独立进程组与限时）；tools/isaac_lab_smoke.py（PhysX/PPO/原始规划回放）；tools/prepare_boundary_probe.py（冻结同起点请求）；tools/validate_physx_replay.py（实际FK/末端/姿态验收）；tools/package_r1.py（checkpoint更新计数/真实曲线/哈希）。R1/raw各实验保留当时runtime_source.py；不要用最新版源码哈希代替旧运行来源。
