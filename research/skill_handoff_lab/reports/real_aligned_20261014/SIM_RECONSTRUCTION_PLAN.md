# 下一轮最小real-aligned Isaac工作单

旧3c20dcd及f5e9eb0训练代码/数据保留；新增独立`site_reference`配置和USD层。当前STATIC模板不允许直接作为物理训练环境。

|阶段|执行内容|可验收产物|证据/依赖|
|---|---|---|---|
|R0资料冻结|导入10月tar/dirty config/捕获manifest，逐文件SHA；区分object4与object3|请求timeline、约束指纹、版本表|UNVERIFIED→原档；不调用现场|
|R1坐标/对象|两臂按robot_world+独立校准装配；每臂active TCP；CAD root→bbox/ears→grasp→TCP→object链；修正144mm数字几何错位|28+姿态数值核对；对象/夹爪碰撞一致、真实耳片/附着坐标|HISTORICAL模型先做；实物身份/Oct配置需导出|
|R2单env持物|实测right Preland q/合法完整路径，左真实HOLD_CURRENT，实际/显式代理托盘、确认工位；native限位/速度/加速度|Isaac5.1 PhysX步进、受控姿态保持、对象包络/接触/净距、成功失败|非静态replay；原mesh碰撞保留，不能删障碍求成功|
|R3最后操作|核验Preland→Place动作方向/目标语义/接触/释放；先名义控制再RL|成功/失败演示和独立完整过程验收|仅已附着可先做，抓取/释放未实现必须声明|
|R4向量化|clone1→4→16，各自left/right q、物体、station/scene epoch、几何/载荷/目标；env独立 reset/step|同一应用物理阵列截图、独立episode/physics日志|复用既有Lab2.3 DirectRLEnv，不是静态复制或16进程|
|R5研究训练|预声明场景组/难度/seed，A或B及强基线，同预算对照|真实模型、曲线、冻结测试，失败/UNKNOWN保留|开始正式训练前确认任务与数据层级，不重刷WideCabinet|

## 立即可做的最小版本

本轮构建SITE_REFERENCE_STATIC：历史两臂安装/q、固定EG2 CAD、旧金属Tray展示、旧chassis外观（仅base-position-fit，旋转/形状未现场验收）、9月点云代理wireframes、reported Oct目标和Preland。WORLD=BODY是研究坐标选择，不是旧main全局标定。报告STATIC_RENDER/HISTORICAL/PROXY_ONLY，0物理步，0任务成功。

旧metal mesh重新居中只用于对照展示，**没有捏造右手T_Link6_object**。原PPO焊接盒隐藏显示但不删其源碰撞，STATIC模板未做动态质量/抓持验收；需R1/R2重新生成物理对象。旧chassis取完整源网格，位置拟合不冒充实际安装。不能把static USD直接启用physics后当16环境训练。

## 场景结构与泛化

每env一个BODY-root：左右完整安装、左停放碰撞几何、右Articulation、真实/声明代理对象、工位实体与扫描surface/unknown/free分层。Env_grid_origin仅显示/物理隔离，不改变BODY标定。底盘实际换站使PICK scene/轨迹失效；PLACE新观测重新绑定。第一版允许固定真实PLACE站，后续再模拟AMR移动与观测规则。

训练参数需以包络和量测限定：开口宽/高/深、法向与目标、夹持偏差/物体尺寸、接触材料/载荷；宽窄序列先CAD/碰撞/IK可行性验收，再预声明随机化。real-aligned原例作为冻结回归/测试组，不能不断挑有利seed。

## 最小用户材料包

优先仅给两份：①`ARES-R_2026-10-12_Preland_Planning_Records`完整原归档（含run配置/读回/识别/拒绝）；②对应点云/scene manifest＋物体CAD或带尺/夹持照片。若打包困难，先给路径和原始request/planning/native拒绝三个文件及robot_world/system/motion/tool导出，就能先继续A和坐标审计。不要求密码、私钥、重启或新扫描。
